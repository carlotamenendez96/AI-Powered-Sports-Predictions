"""Edge check — is football actually beating the price?

The landing page's counterpart to the cashout backtest: a mechanical run that
re-answers, from whatever has settled so far, the three questions this repo
keeps needing:

1. **Is the EV signal real?** Bucket every settled bet by the EV the model
   claimed and compare with the return actually realised. A working signal
   makes ROI rise with claimed EV. Measured 2026-09-13 it did the opposite
   (Spearman rho -0.038 over 2,354 bets), because the model's probabilities
   are compressed toward 0.5 and `EV = conf x odds - 1` manufactures EV
   exactly where the model is most wrong.

2. **Did disabling per-league calibration help?** Splits settled bets at
   CALIBRATION_OFF_DATE so the before/after ROI is visible as volume accrues.

3. **Does the model beat "just back the favourite"?** Only answerable on days
   whose pre-match odds survive. Verification used to overwrite
   `matches_<date>.json` with results and drop the odds; since 2026-09-13 it
   writes `results_<date>.json` instead, so this number starts working from
   that date forward. The pane shows how many scoreable days exist yet.

Writes `output/edge_checks/<timestamp>.{json,txt}` — JSON for the UI, text for
reading. Never writes outside output/; safe to re-run any time.

Usage:
    python3 scripts/run_edge_check.py
    python3 scripts/run_edge_check.py --since 2026-08-01
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
OUT_DIR = os.path.join(OUTPUT_DIR, 'edge_checks')

# The day per-league Platt calibration was switched off for football
# (commit 203fe11). Bets settle against the model that priced them, so the
# split is on the bet's own date, not on when it settled.
CALIBRATION_OFF_DATE = '2026-09-13'

# EV buckets. The first is deliberately "the model said don't bet" — those
# rows exist because the conviction and model lanes ignore EV, and they are
# the control group that makes the comparison meaningful.
EV_BUCKETS = [(-9e9, 0.0, '<0'), (0.0, 0.05, '0-.05'), (0.05, 0.10, '.05-.10'),
              (0.10, 0.20, '.10-.20'), (0.20, 0.50, '.20-.50'), (0.50, 9e9, '>.50')]


def _f(v):
    """Float or None — bet slips carry EV as '+0.39' strings and conf as str."""
    try:
        return float(str(v).replace('+', '').strip())
    except (TypeError, ValueError):
        return None


def load_settled_bets(since=None):
    """Every WON/LOST bet from active and archived slips.

    CASHED_OUT and VOID are excluded: their P/L reflects a cashout decision or
    a refund, not whether the pick was right, which is what this script asks.
    """
    rows = []
    for bf in (glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json'))
               + glob.glob(os.path.join(OUTPUT_DIR, 'history', 'bets_*.json'))):
        try:
            with open(bf) as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for b in (blob.get('bets') if isinstance(blob, dict) else blob) or []:
            if b.get('status') not in ('WON', 'LOST'):
                continue
            date = str(b.get('date', ''))[:10]
            if since and date < since:
                continue
            ev, conf = _f(b.get('ev')), _f(b.get('conf'))
            odds = _f(b.get('odds') if b.get('odds') is not None else b.get('odd'))
            stake = _f(b.get('stake'))
            # `profit` is the backward-compatible alias and is present on every
            # settled bet; `pnl` is absent on the oldest slips. Reading only
            # `pnl` silently dropped those rows at the filter below (36 of
            # 2,606 as of 2026-09-18) — they are real settled bets.
            pnl = _f(b.get('pnl')) if b.get('pnl') is not None else _f(b.get('profit'))
            if None in (odds, stake, pnl) or stake <= 0:
                continue
            rows.append({
                'date': date, 'lane': b.get('lane', '?'), 'type': b.get('type', '?'),
                'ev': ev, 'conf': conf, 'odds': odds, 'stake': stake, 'pnl': pnl,
                'won': b.get('status') == 'WON',
            })
    return pd.DataFrame(rows)


def _agg(df):
    """n / stake / pnl / roi / winrate for a slice. Empty-safe."""
    if df.empty:
        return {'bets': 0, 'stake': 0.0, 'pnl': 0.0, 'roi': None, 'winrate': None}
    stake = float(df.stake.sum())
    pnl = float(df.pnl.sum())
    return {
        'bets': int(len(df)),
        'stake': round(stake, 2),
        'pnl': round(pnl, 2),
        'roi': round(pnl / stake, 4) if stake else None,
        'winrate': round(float(df.won.mean()), 4),
    }


def ev_buckets(df):
    """Realised ROI per claimed-EV bucket, plus the rank correlation.

    `rho` is Spearman between claimed EV and realised per-euro return. That is
    the headline: positive means the EV signal orders bets correctly, ~0 or
    negative means it does not.
    """
    have_ev = df[df.ev.notna()].copy()
    out = {'rho': None, 'rho_p': None, 'n': int(len(have_ev)), 'buckets': []}
    if have_ev.empty:
        return out
    have_ev['ret'] = have_ev.pnl / have_ev.stake
    try:
        from scipy import stats
        rho, p = stats.spearmanr(have_ev.ev, have_ev.ret)
        out['rho'], out['rho_p'] = round(float(rho), 4), float(p)
    except Exception:
        pass
    for lo, hi, label in EV_BUCKETS:
        sl = have_ev[(have_ev.ev >= lo) & (have_ev.ev < hi)]
        row = _agg(sl)
        row['bucket'] = label
        row['mean_odds'] = round(float(sl.odds.mean()), 2) if not sl.empty else None
        # The strike rate this bucket needed just to break even.
        row['breakeven_winrate'] = (round(1.0 / row['mean_odds'], 4)
                                    if row['mean_odds'] else None)
        out['buckets'].append(row)
    return out


def market_baseline():
    """Model pick accuracy vs backing the bookmaker favourite, same matches.

    Only scoreable for a date that has BOTH a verification CSV and a pre-match
    slate still carrying 1X2 odds. Days verified before 2026-09-13 had their
    odds overwritten by the verification scrape, so they are unrecoverable and
    simply do not appear here.
    """
    result = {'days': 0, 'matches': 0, 'model_correct': 0, 'market_correct': 0,
              'model_acc': None, 'market_acc': None, 'dates': [],
              'unscoreable_days': 0}
    for vf in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'verification_*.csv'))):
        date = os.path.basename(vf)[len('verification_'):-len('.csv')]
        mf = os.path.join(OUTPUT_DIR, f'matches_{date}.json')
        if not os.path.exists(mf):
            result['unscoreable_days'] += 1
            continue
        try:
            with open(mf) as f:
                matches = json.load(f)
            vdf = pd.read_csv(vf)
        except Exception:
            result['unscoreable_days'] += 1
            continue

        odds = {}
        for m in matches:
            try:
                odds[(str(m.get('home_team', '')).strip(),
                      str(m.get('away_team', '')).strip())] = {
                    '1': float(m['interaction_1x2_1']),
                    'X': float(m['interaction_1x2_X']),
                    '2': float(m['interaction_1x2_2'])}
            except (KeyError, TypeError, ValueError):
                continue
        if not odds:
            result['unscoreable_days'] += 1   # slate present but odds stripped
            continue

        day_n = day_model = day_market = 0
        for _, r in vdf.iterrows():
            key = (str(r.get('Home', '')).strip(), str(r.get('Away', '')).strip())
            actual = str(r.get('Actual 1X2', '')).strip()
            pick = str(r.get('Pred 1X2', '')).strip()
            if key not in odds or actual not in ('1', 'X', '2') or pick not in ('1', 'X', '2'):
                continue
            fav = min(odds[key], key=odds[key].get)
            day_n += 1
            day_model += (pick == actual)
            day_market += (fav == actual)
        if not day_n:
            result['unscoreable_days'] += 1
            continue
        result['days'] += 1
        result['matches'] += day_n
        result['model_correct'] += day_model
        result['market_correct'] += day_market
        result['dates'].append(date)

    if result['matches']:
        result['model_acc'] = round(result['model_correct'] / result['matches'], 4)
        result['market_acc'] = round(result['market_correct'] / result['matches'], 4)
    return result


def build_report(since=None):
    df = load_settled_bets(since=since)
    before = df[df.date < CALIBRATION_OFF_DATE]
    after = df[df.date >= CALIBRATION_OFF_DATE]
    return {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'since': since,
        'calibration_off_date': CALIBRATION_OFF_DATE,
        'overall': _agg(df),
        'by_lane': {lane: _agg(sl) for lane, sl in df.groupby('lane')} if not df.empty else {},
        'by_market': {mkt: _agg(sl) for mkt, sl in df.groupby('type')} if not df.empty else {},
        'calibration_split': {'before': _agg(before), 'after': _agg(after)},
        'ev': ev_buckets(df),
        'market_baseline': market_baseline(),
        'date_range': ([str(df.date.min()), str(df.date.max())] if not df.empty else None),
    }


def render_text(rep):
    L = []
    A = L.append
    A('=' * 78)
    A('FOOTBALL EDGE CHECK')
    A('=' * 78)
    A(f"generated {rep['generated_at']}")
    if rep['date_range']:
        A(f"settled bets from {rep['date_range'][0]} to {rep['date_range'][1]}")
    o = rep['overall']
    A('')
    overall_roi = 'n/a' if o['roi'] is None else f"{o['roi']:+.1%}"
    A(f"OVERALL  {o['bets']} bets | stake EUR {o['stake']:,.2f} | "
      f"P/L EUR {o['pnl']:+,.2f} | ROI {overall_roi}")
    A('')
    A('--- by lane ---')
    for lane, a in sorted(rep['by_lane'].items()):
        roi = 'n/a' if a['roi'] is None else f"{a['roi']:+.1%}"
        A(f"  {lane:<11} {a['bets']:>5} bets  stake {a['stake']:>9,.2f}  "
          f"P/L {a['pnl']:>+9,.2f}  ROI {roi:>7}")
    A('')
    A('--- does claimed EV predict realised return? ---')
    ev = rep['ev']
    rho = 'n/a' if ev['rho'] is None else f"{ev['rho']:+.4f}"
    A(f"  Spearman rho(EV, per-euro return) = {rho}   (n={ev['n']})")
    A('  a working EV signal needs rho > 0 and ROI rising down this table')
    A(f"  {'bucket':<9} {'bets':>6} {'stake':>10} {'P/L':>10} {'ROI':>8} "
      f"{'winrate':>8} {'breakeven':>10}")
    for b in ev['buckets']:
        if not b['bets']:
            continue
        roi = 'n/a' if b['roi'] is None else f"{b['roi']:+.1%}"
        wr = 'n/a' if b['winrate'] is None else f"{b['winrate']:.1%}"
        be = 'n/a' if b['breakeven_winrate'] is None else f"{b['breakeven_winrate']:.1%}"
        A(f"  {b['bucket']:<9} {b['bets']:>6} {b['stake']:>10,.0f} {b['pnl']:>+10,.1f} "
          f"{roi:>8} {wr:>8} {be:>10}")
    A('')
    A(f"--- calibration split (off from {rep['calibration_off_date']}) ---")
    for k in ('before', 'after'):
        a = rep['calibration_split'][k]
        roi = 'n/a' if a['roi'] is None else f"{a['roi']:+.1%}"
        A(f"  {k:<7} {a['bets']:>5} bets  stake {a['stake']:>9,.2f}  "
          f"P/L {a['pnl']:>+9,.2f}  ROI {roi:>7}")
    A('')
    A('--- model vs market (days whose pre-match odds survived) ---')
    mb = rep['market_baseline']
    if mb['matches']:
        A(f"  {mb['days']} scoreable days, {mb['matches']} matches")
        A(f"  model  pick accuracy: {mb['model_acc']:.2%}")
        A(f"  favourite accuracy  : {mb['market_acc']:.2%}")
        edge = mb['model_acc'] - mb['market_acc']
        A(f"  difference          : {edge:+.2%}  "
          f"({'model ahead' if edge > 0 else 'market ahead'})")
    else:
        A('  no scoreable days yet — verification before 2026-09-13 overwrote the')
        A('  pre-match slate and dropped the odds. This fills in from that date on.')
    A(f"  unscoreable days: {mb['unscoreable_days']}")
    A('')
    A('=' * 78)
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default=None, help='Only bets on/after YYYY-MM-DD.')
    args = ap.parse_args()

    rep = build_report(since=args.since)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'{ts}.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    text = render_text(rep)
    with open(os.path.join(OUT_DIR, f'{ts}.txt'), 'w') as f:
        f.write(text + '\n')
    print(text)
    print(f"\nSaved -> {os.path.join(OUT_DIR, ts)}.{{json,txt}}")


if __name__ == '__main__':
    main()
