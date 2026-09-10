"""Evaluate the auto-cashout decisions against the COUNTERFACTUAL of
holding each bet to the final whistle.

A cashout decision can only be judged in hindsight by asking: *what would
I have got if I'd held instead?* We have both halves:
  - what we GOT   = the cashout amount (logged in output/auto_cashout_log.jsonl)
  - what we'd GET = the bet's settlement outcome (from output/verification_*.csv):
                    stake × odds if the selection actually won, else 0.

So per executed auto-cashout:
    held_return = stake × odds   if selection won at full-time, else 0
    cash_return = cashout amount
    delta       = cash_return − held_return     (>0 ⇒ cashing beat holding)

The headline is the AGGREGATE Σcash − Σheld: positive ⇒ the rule made more
than holding everything to settlement; negative ⇒ it cost value (the price
paid for variance reduction wasn't repaid by the losses it dodged). A single
bet beaten by holding isn't a "wrong" decision — only the aggregate over many
decisions is meaningful (variance). Intuition split:
  - stop_loss "saves"  = cash banked on bets that went on to LOSE (vs 0)
  - lock_in  "gives up" = profit forgone on bets that went on to WIN
                          (held_return − cash_return)

CAVEAT: cash_return is the SYNTHETIC fair-value estimate (no real bookmaker
haircut), so Σcash is optimistic vs what a real cashout would have paid.

Read-only. Joins the audit log to verification CSVs by (date, match).
Writes output/auto_cashout_eval.json + prints a report.

Usage:
    python3 scripts/evaluate_auto_cashout.py
"""

import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'output')
LOG_PATH = os.path.join(OUTPUT_DIR, 'auto_cashout_log.jsonl')


def _norm(s):
    return ' '.join((s or '').strip().lower().split())


def _load_verifications():
    """{date: {normalized_match: row}} from all verification CSVs."""
    idx = {}
    files = (glob.glob(os.path.join(OUTPUT_DIR, 'verification_*.csv')) +
             glob.glob(os.path.join(OUTPUT_DIR, 'history', 'verification_*.csv')))
    for f in files:
        base = os.path.basename(f)
        # verification_YYYY-MM-DD.csv  (ignore any .timestamp suffix)
        date = base.replace('verification_', '')[:10]
        try:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    m = _norm(r.get('Match'))
                    if m:
                        idx.setdefault(date, {})[m] = r
                        # also index by "home vs away" as a fallback key
                        ha = _norm(f"{r.get('Home')} vs {r.get('Away')}")
                        idx[date].setdefault(ha, r)
        except OSError:
            continue
    return idx


def _load_lane_bets():
    """{bet_id: [{lane, stake, odds, cashout_amount}, ...]} from every slip.

    The audit log records ONE representative bet per bet_id, but a cashout
    cascades across every lane holding that same wager. So the per-lane
    bankroll impact can't be read off the log — it has to be recovered from
    the slips, where each lane's own stake and realized amount live.
    """
    idx = {}
    files = (glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json')) +
             glob.glob(os.path.join(OUTPUT_DIR, 'history', 'bets_*.json')))
    for f in files:
        try:
            with open(f) as fh:
                slip = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        for b in slip.get('bets', []):
            bid = b.get('bet_id')
            if not bid:
                continue
            amt = b.get('cashout_amount')
            idx.setdefault(bid, []).append({
                'lane': b.get('lane', 'value'),
                'stake': float(b.get('stake_units', b.get('stake', 0)) or 0),
                'odds': float(b.get('odds', b.get('odd', 0)) or 0),
                'cashout_amount': None if amt is None else float(amt),
            })
    return idx


def _lane_impact(scored, lane_bets):
    """Per-lane bankroll delta for these cashouts.

    Where a lane actually cashed out, use its RECORDED `cashout_amount` —
    that is ground truth. Only fall back to scaling the representative's
    amount by the lane's share of stake when no amount was recorded (i.e.
    the shadow counterfactual, where nothing fired).

    The fallback is exact for shadow because a cascade prices every lane at
    one instant off the same adj_prob, and the price is linear in stake. It
    is NOT valid for historical executed cashouts: lanes there were cashed
    at different times (pre-cascade), so their ratios genuinely differ.
    """
    per = {}
    for x in scored:
        rows = lane_bets.get(x.get('bet_id')) or []
        rep_stake = float(x.get('stake') or 0)
        if not rows or rep_stake <= 0:
            continue
        for r in rows:
            stake = r['stake']
            odds = r['odds'] or float(x.get('odds') or 0)
            if r['cashout_amount'] is not None:
                cash = r['cashout_amount']
                basis = 'recorded'
            else:
                cash = float(x['cash_return']) * (stake / rep_stake)
                basis = 'scaled'
            held = stake * odds if x['won'] else 0.0
            d = per.setdefault(r['lane'], {'n': 0, 'stake': 0.0, 'cash': 0.0,
                                           'held': 0.0, 'recorded': 0, 'scaled': 0})
            d['n'] += 1
            d['stake'] += stake
            d['cash'] += cash
            d['held'] += held
            d[basis] += 1
    for d in per.values():
        d['delta'] = round(d['cash'] - d['held'], 2)
        for k in ('stake', 'cash', 'held'):
            d[k] = round(d[k], 2)
    return per


def _selection_won(bet_type, selection, row):
    """Did this selection win, per the verification row? Returns
    True/False, or None if the row can't decide (missing column)."""
    sel = (selection or '').strip()
    if bet_type == 'O/U' or 'Over' in sel or 'Under' in sel:
        actual = (row.get('Actual O/U') or '').strip()
        if not actual:
            return None
        want = 'Over 2.5' if 'Over' in sel else 'Under 2.5'
        return actual == want
    # 1X2
    actual = (row.get('Actual 1X2') or '').strip()
    if not actual:
        return None
    canon = {'1': '1', 'home': '1', 'x': 'X', 'X': 'X', 'draw': 'X',
             '2': '2', 'away': '2'}.get(sel, sel)
    return actual == canon


def main():
    if not os.path.exists(LOG_PATH):
        print(f"No audit log at {LOG_PATH} — nothing auto-cashed yet.")
        return 0

    # Cashout events, deduped by bet_id keeping the FIRST occurrence.
    #
    # Executed rows: a bet flips to CASHED_OUT after the first firing, so
    # later sweeps skip it (dedupe defensively anyway).
    #
    # Shadow rows: nothing fires, so the bet stays OPEN and is re-evaluated
    # every sweep for the rest of its live window. The first non-hold entry
    # is the would-have-fired moment — later ones are the same decision
    # repeated and must not be counted again.
    fired, shadowed = {}, {}
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = e.get('bet_id')
            if not bid:
                continue
            if e.get('executed'):
                fired.setdefault(bid, e)
            elif e.get('would_fire'):
                shadowed.setdefault(bid, e)

    # A bet that was shadow-logged and later fired for real is counted as
    # executed only — the real event supersedes its own counterfactual.
    shadowed = {k: v for k, v in shadowed.items() if k not in fired}

    if not fired and not shadowed:
        print("Audit log has no cashouts to score yet (only held evaluations).")
        return 0

    verifs = _load_verifications()

    def score(events, mode):
        scored, pending = [], []
        for bid, e in events.items():
            date = bid.split(':', 1)[0] if ':' in bid else ''
            row = (verifs.get(date) or {}).get(_norm(e.get('match')))
            if row is None:
                pending.append(e)
                continue
            won = _selection_won(e.get('type'), e.get('selection'), row)
            if won is None:
                pending.append(e)
                continue
            stake = float(e.get('stake') or 0)
            odds = float(e.get('odds') or 0)
            cash = float(e.get('amount') or 0)
            held = stake * odds if won else 0.0
            scored.append({**e, 'mode': mode, 'won': won, 'score': row.get('Score'),
                           'held_return': round(held, 2), 'cash_return': round(cash, 2),
                           'delta': round(cash - held, 2)})
        return scored, pending

    # Aggregate.
    def agg(items):
        cash = sum(x['cash_return'] for x in items)
        held = sum(x['held_return'] for x in items)
        return {'n': len(items), 'cash': round(cash, 2), 'held': round(held, 2),
                'net_delta': round(cash - held, 2)}

    def summarize(scored):
        return {
            'overall': agg(scored),
            'by_decision': {d: agg([x for x in scored if x['decision'] == d])
                            for d in ('lock_in', 'stop_loss')
                            if any(x['decision'] == d for x in scored)},
            'stop_loss_saved_vs_losing':
                round(sum(x['cash_return'] for x in scored if not x['won']), 2),
            'lock_in_given_up_vs_winning':
                round(sum(x['held_return'] - x['cash_return'] for x in scored if x['won']), 2),
            'cashed_bets_that_would_have_won': sum(1 for x in scored if x['won']),
            'details': sorted(scored, key=lambda x: x['delta']),
        }

    scored, pending = score(fired, 'executed')
    shadow_scored, shadow_pending = score(shadowed, 'shadow')

    lane_bets = _load_lane_bets()
    lane_exec = _lane_impact(scored, lane_bets)
    lane_shadow = _lane_impact(shadow_scored, lane_bets)

    report = {
        'cashouts_executed': len(fired), 'scored': len(scored),
        'pending_settlement': len(pending),
        **summarize(scored),
        'lane_impact_executed': lane_exec,
        'shadow': {
            'would_fire': len(shadowed), 'scored': len(shadow_scored),
            'pending_settlement': len(shadow_pending),
            **summarize(shadow_scored),
            'lane_impact': lane_shadow,
        },
        'caveat': 'cash_return is the SYNTHETIC estimate (no real haircut) → Σcash optimistic.',
    }
    overall = report['overall']
    by_dec = report['by_decision']
    saved = report['stop_loss_saved_vs_losing']
    given_up = report['lock_in_given_up_vs_winning']
    would_win = report['cashed_bets_that_would_have_won']
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'auto_cashout_eval.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Console report.
    print("=== Auto-cashout decision evaluation (cash vs hold-to-settlement) ===")
    print(f"executed cashouts: {len(fired)}   scored: {len(scored)}   "
          f"pending settlement: {len(pending)}")
    if not scored and shadowed:
        print("  (no real cashouts — running in shadow mode; see SHADOW section below)")
    if scored:
        o = overall
        verdict = ("auto-cashout BEAT holding" if o['net_delta'] > 0 else
                   "auto-cashout COST vs holding" if o['net_delta'] < 0 else "even")
        print(f"\nAggregate over {o['n']} scored cashouts:")
        print(f"  Σ cashed  = €{o['cash']:.2f}")
        print(f"  Σ if held = €{o['held']:.2f}")
        print(f"  net Δ     = €{o['net_delta']:+.2f}   → {verdict}")
        print(f"  ({would_win}/{o['n']} cashed bets would have WON if held)")
        print(f"\n  stop_loss saved (on bets that went on to lose): €{saved:+.2f}")
        print(f"  lock_in gave up (on bets that went on to win):  €{-given_up:+.2f}")
        for d, a in by_dec.items():
            print(f"  [{d:<9}] n={a['n']}  cashed=€{a['cash']:.2f}  "
                  f"held=€{a['held']:.2f}  netΔ=€{a['net_delta']:+.2f}")
        print("\n  Worst 5 decisions (held would've beaten cashing):")
        for x in report['details'][:5]:
            if x['delta'] >= 0:
                break
            print(f"    {x['match'][:34]:<34} {x['decision']:<9} {x['selection']:<9} "
                  f"score={x['score']} cash=€{x['cash_return']:.2f} held=€{x['held_return']:.2f} "
                  f"Δ€{x['delta']:+.2f}")
    if shadow_scored:
        s = report['shadow']
        o = s['overall']
        verdict = ("would have BEAT holding" if o['net_delta'] > 0 else
                   "would have COST vs holding" if o['net_delta'] < 0 else "even")
        print(f"\n=== SHADOW (logged only, no money moved) ===")
        print(f"would-fire decisions: {s['would_fire']}   scored: {s['scored']}   "
              f"pending settlement: {s['pending_settlement']}")
        print(f"  Σ would-cash = €{o['cash']:.2f}")
        print(f"  Σ if held    = €{o['held']:.2f}")
        print(f"  net Δ        = €{o['net_delta']:+.2f}   → {verdict}")
        print(f"  ({s['cashed_bets_that_would_have_won']}/{o['n']} would have WON if held)")
        for d, a in s['by_decision'].items():
            print(f"  [{d:<9}] n={a['n']}  would-cash=€{a['cash']:.2f}  "
                  f"held=€{a['held']:.2f}  netΔ=€{a['net_delta']:+.2f}")
        # How the synthetic price we decide on compares to the real offer.
        with_bk = [x for x in shadow_scored if x.get('bookmaker_offer') is not None]
        if with_bk:
            d_sum = sum(float(x['bookmaker_offer']) - float(x['cash_return']) for x in with_bk)
            print(f"\n  real bookmaker offer present on {len(with_bk)}/{len(shadow_scored)}; "
                  f"Σ(real − synthetic) = €{d_sum:+.2f}")
            print("  (negative ⇒ the real offer pays LESS than our fair-value estimate, "
                  "so the synthetic numbers above are optimistic)")

    def _print_lane_impact(per, title, note):
        if not per:
            return
        print(f"\n{title}")
        print("  lane        bets    staked    would-cash    if held      Δ bankroll")
        for lane in sorted(per, key=lambda l: per[l]['delta']):
            d = per[lane]
            print(f"  {lane:<10} {d['n']:>5}  €{d['stake']:>8.2f}  €{d['cash']:>10.2f}  "
                  f"€{d['held']:>8.2f}   €{d['delta']:>+8.2f}")
        print(f"  {note}")

    _print_lane_impact(
        lane_shadow,
        "=== Per-lane bankroll impact HAD live cashout been ON (shadow) ===",
        "Δ is how each lane's bankroll would have moved vs holding to settlement.")
    _print_lane_impact(
        lane_exec,
        "=== Per-lane bankroll impact of cashouts that ACTUALLY fired ===",
        "This money already moved — shown for comparison with the shadow rows.")

    if pending or shadow_pending:
        print(f"\n  {len(pending)} executed + {len(shadow_pending)} shadow cashout(s) await "
              f"settlement (no verification row yet) — re-run after the next verification.")
    print(f"\nReport → {os.path.join(OUTPUT_DIR, 'auto_cashout_eval.json')}")
    print("Caveat: cash_return is the synthetic estimate (no real haircut) → Σcash optimistic.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
