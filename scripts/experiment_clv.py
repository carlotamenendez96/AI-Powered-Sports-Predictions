"""E0 — Closing Line Value against Pinnacle closing.

The question every other experiment is downstream of: when the model picks a
side at the price we could actually have taken, does the market move TOWARD
that side by kickoff? CLV is the leading indicator of edge — it is measurable
per bet at far lower variance than ROI, which at current volume (2,606 settled
bets, detectable edge +/-5.5%) cannot select between models at all.

This runs entirely offline on the existing corpus. `data_loader` exposes
`close_H/D/A` (Pinnacle closing preferred, 90.8% of rows), so no scraping is
needed and the sample is ~100x the live betting record.

METHOD

  taken price   B365H/D/A, the reference price that feeds IP_* - i.e. exactly
                what the model saw and what /auto_wager would have bet into.
  closing price close_H/D/A, devigged.
  CLV_log       log(odds_taken) - log(odds_close) on the model's pick.
                Positive => the price shortened => we took the better side.
                Reported as the headline because it is additive across bets and
                far better behaved than the raw ratio.
  CLV_prob      P_close(pick) - P_taken(pick), both devigged - the same fact in
                probability space, which is easier to sanity-check.

  ROWS ARE RESTRICTED TO odds_is_closing == False. This is not a detail: for
  33.9% of the corpus `B365H` IS a closing price (those files carry no opening
  quote), so "taken vs closing" there would be closing vs closing and CLV would
  be mechanically ~0. Before the 2026-09-18 provenance fix this was invisible
  and any CLV study would have been silently wrong on a third of its sample.

CONTROLS

  favourite   always back the shortest price - the benchmark a model must beat
  random      a uniformly random side, seeded - the zero-skill floor
  model       the production feature set, out-of-fold

A model with no edge still shows CLV ~ 0 against a sharp close. A model that is
systematically taking the wrong side of steam shows NEGATIVE CLV, which is
worse than zero and is the outcome the adverse-selection diagnosis predicts for
the EV-gated subset specifically.

Writes output/experiments/clv_<ts>.{json,txt}. Never writes into models/.

Usage:
    python3 scripts/experiment_clv.py
    python3 scripts/experiment_clv.py --splits 5 --min-league 150
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import (          # noqa: E402
    prepare, add_1x2_market, oof_predictions, stack_test,
)
from experiment_h2h import arm_features, feats_required     # noqa: E402
from model_registry import get_spec                          # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')
CARRY = ['close_H', 'close_D', 'close_A', 'close_source',
         'odds_source', 'odds_is_closing', 'target_1x2']
ODDS_BANDS = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 5.0), (5.0, 99.0)]
EV_BANDS = [(-9, 0, 'EV<=0'), (0, .05, 'EV 0-0.05'), (.05, .10, 'EV 0.05-0.10'),
            (.10, .20, 'EV 0.10-0.20'), (.20, 9, 'EV>=0.20')]
DEV_BANDS = [(-9, -.05, 'dev<-0.05'), (-.05, -.02, '-0.05..-0.02'),
             (-.02, .02, '-0.02..+0.02'), (.02, .05, '+0.02..+0.05'),
             (.05, 9, 'dev>+0.05')]


def devig(mat):
    inv = 1.0 / mat
    return inv / inv.sum(axis=1, keepdims=True)


def boot_ci(x, iters=4000, seed=0):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    return tuple(float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def rps(P, y):
    Y = np.zeros_like(P)
    Y[np.arange(len(P)), y] = 1.0
    cP, cY = np.cumsum(P, 1), np.cumsum(Y, 1)
    return float(((cP[:, :-1] - cY[:, :-1]) ** 2).sum(1).mean() / (P.shape[1] - 1))


def clv_stats(clv_log, clv_prob, won, odds_taken):
    """Headline CLV plus the realised return on the same selection."""
    n = len(clv_log)
    if n == 0:
        return {'n': 0}
    lo, hi = boot_ci(clv_log)
    pnl = np.where(won, odds_taken - 1.0, -1.0)
    rlo, rhi = boot_ci(pnl)
    return {
        'n': int(n),
        'clv_log_mean': float(clv_log.mean()),
        'clv_log_ci': [lo, hi],
        'clv_log_positive_rate': float((clv_log > 0).mean()),
        'clv_prob_mean': float(clv_prob.mean()),
        'flat_roi': float(pnl.mean()),
        'flat_roi_ci': [rlo, rhi],
        'hit_rate': float(won.mean()),
        'beats_zero': bool(lo > 0),
    }


def selection(P_model, taken, close_dv, y, pick):
    """CLV for one selection rule, given a pick index per match."""
    r = np.arange(len(pick))
    odds_taken = taken[r, pick]
    taken_dv = devig(taken)[r, pick]
    p_close = close_dv[r, pick]
    # Closing odds implied by the devigged closing probability, so both sides
    # of the log are margin-free and the comparison is not a vig artifact.
    odds_close_fair = 1.0 / np.clip(p_close, 1e-6, 1.0)
    odds_taken_fair = 1.0 / np.clip(taken_dv, 1e-6, 1.0)
    clv_log = np.log(odds_taken_fair) - np.log(odds_close_fair)
    clv_prob = p_close - taken_dv
    return clv_log, clv_prob, (pick == y), odds_taken


def build(df, splits, min_league, seed=0):
    spec = get_spec('1x2', 'xgboost')
    feats = arm_features('1x2', 'base', spec)
    oof = oof_predictions(df, '1x2', 'base', n_splits=splits, feats=feats,
                          dropna_on=feats_required(feats), carry=CARRY)

    rep = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'oof_rows_all': int(len(oof)), 'features': len(feats), 'splits': splits}

    # --- the restriction that makes this valid -----------------------------
    has_close = oof[['close_H', 'close_D', 'close_A']].notna().all(axis=1)
    is_open = oof['odds_is_closing'] == False          # noqa: E712
    keep = has_close & is_open
    rep['rows_with_close'] = int(has_close.sum())
    rep['rows_opening_price'] = int(is_open.sum())
    rep['rows_scoreable'] = int(keep.sum())
    rep['close_source'] = oof.loc[keep, 'close_source'].value_counts().to_dict()
    d = oof[keep].reset_index(drop=True)
    if len(d) < 500:
        rep['error'] = 'too few scoreable rows'
        return rep

    P = d[['p_H', 'p_D', 'p_A']].values
    taken = d[['B365H', 'B365D', 'B365A']].values
    close = d[['close_H', 'close_D', 'close_A']].values
    close_dv = devig(close)
    taken_dv = devig(taken)
    y = d['y'].values.astype(int)

    # --- how good is the closing line vs the price we take? ----------------
    rep['benchmarks'] = {
        'rps_model': rps(P, y),
        'rps_taken_market': rps(taken_dv, y),
        'rps_closing': rps(close_dv, y),
        'brier_model': float(((P - np.eye(3)[y]) ** 2).sum(1).mean()),
        'brier_taken_market': float(((taken_dv - np.eye(3)[y]) ** 2).sum(1).mean()),
        'brier_closing': float(((close_dv - np.eye(3)[y]) ** 2).sum(1).mean()),
    }

    # Gate 0 of the ladder: nats added on top of the CLOSING price, not the
    # taken one. This is the honest version of the test the repo has been
    # running against B365 all along.
    pick = P.argmax(1)
    r = np.arange(len(P))
    rep['stack_vs_closing'] = stack_test(P[r, pick], close_dv[r, pick], (pick == y).astype(int))
    rep['stack_vs_taken'] = stack_test(P[r, pick], taken_dv[r, pick], (pick == y).astype(int))

    # --- selections --------------------------------------------------------
    rng = np.random.default_rng(seed)
    picks = {
        'model': P.argmax(1),
        'favourite': taken.argmin(1),
        'random': rng.integers(0, 3, len(P)),
    }
    rep['selections'] = {}
    store = {}
    for name, pk in picks.items():
        cl, cp, won, od = selection(P, taken, close_dv, y, pk)
        rep['selections'][name] = clv_stats(cl, cp, won, od)
        store[name] = (cl, cp, won, od, pk)

    # --- strata, model selection only --------------------------------------
    cl, cp, won, od, pk = store['model']
    ev = P[r, pk] * taken[r, pk] - 1.0
    dev = P[r, pk] - taken_dv[r, pk]

    def stratify(label, bands, values):
        out = {}
        for b in bands:
            lo, hi = b[0], b[1]
            nm = b[2] if len(b) > 2 else f'{lo}-{hi}'
            m = (values >= lo) & (values < hi)
            if m.sum() < 100:
                continue
            out[nm] = clv_stats(cl[m], cp[m], won[m], od[m])
        rep['strata_' + label] = out

    stratify('odds', ODDS_BANDS, od)
    stratify('ev', EV_BANDS, ev)
    stratify('deviation', DEV_BANDS, dev)

    # Conviction gate replica: conf >= 0.65 and odds >= 1.40, the one lane with
    # a positive live point estimate.
    conv = (P[r, pk] >= 0.65) & (od >= 1.40)
    if conv.sum() >= 100:
        rep['conviction_gate'] = clv_stats(cl[conv], cp[conv], won[conv], od[conv])

    # --- ARTIFACT CONTROL: same-book vs cross-book -------------------------
    #
    # The odds-band CLV ladder below is monotone, which is exactly what a
    # DEVIGGING ARTIFACT would look like. Proportional devigging understates a
    # favourite's true probability by more when the margin is larger, so
    # comparing a high-margin book (B365 opening, mean overround 1.0609) to a
    # low-margin one (Pinnacle closing, 1.0314) biases CLV POSITIVE on short
    # prices and NEGATIVE on long ones — the pattern being measured.
    #
    # The clean control is the subset whose closing quote is B365's own: same
    # book, same margin structure, so any surviving ladder is real line
    # movement rather than a margin gradient.
    rep['overround'] = {
        'taken_mean': float((1 / taken).sum(axis=1).mean()),
        'closing_mean': float((1 / close).sum(axis=1).mean()),
        'by_close_source': {
            str(k): float((1 / close[(d['close_source'] == k).values]).sum(axis=1).mean())
            for k in d['close_source'].dropna().unique()
        },
    }

    rep['book_split'] = {}
    for src in ('B365C', 'PSC'):
        m = (d['close_source'] == src).values
        if m.sum() < 300:
            continue
        same = (src == 'B365C')          # taken is B365 opening
        entry = {'n': int(m.sum()), 'same_book': same,
                 'overall': clv_stats(cl[m], cp[m], won[m], od[m]), 'by_odds': {}}
        for lo, hi in ODDS_BANDS:
            mm = m & (od >= lo) & (od < hi)
            if mm.sum() < 100:
                continue
            entry['by_odds'][f'{lo}-{hi}'] = clv_stats(cl[mm], cp[mm], won[mm], od[mm])
        conv_m = m & conv
        if conv_m.sum() >= 60:
            entry['conviction'] = clv_stats(cl[conv_m], cp[conv_m], won[conv_m], od[conv_m])
        rep['book_split'][src] = entry

    # --- CONFOUND CONTROL: is the conviction gate just short odds? ---------
    #
    # conviction = conf >= 0.65 AND odds >= 1.40, which lives almost entirely
    # inside the short-price band that already shows positive CLV. Compare it
    # against non-conviction picks AT THE SAME ODDS to see whether the
    # confidence condition adds anything beyond the price.
    rep['conviction_vs_odds_matched'] = {}
    for lo, hi in ((1.40, 1.70), (1.70, 2.00), (1.40, 2.00)):
        band = (od >= lo) & (od < hi)
        a_, b_ = band & conv, band & ~conv
        if a_.sum() < 50 or b_.sum() < 50:
            continue
        rep['conviction_vs_odds_matched'][f'{lo}-{hi}'] = {
            'conviction': clv_stats(cl[a_], cp[a_], won[a_], od[a_]),
            'rest': clv_stats(cl[b_], cp[b_], won[b_], od[b_]),
        }

    lg = defaultdict(list)
    for i, name in enumerate(d['league'].astype(str).values):
        lg[name].append(i)
    rep['strata_league'] = {}
    for name, idx in lg.items():
        if len(idx) < min_league:
            continue
        i = np.array(idx)
        rep['strata_league'][name] = clv_stats(cl[i], cp[i], won[i], od[i])
    return rep


def _fmt(st):
    if not st.get('n'):
        return f'{"(empty)":>12}'
    lo, hi = st['clv_log_ci']
    mark = '  **' if st['beats_zero'] else ('  ++' if hi < 0 else '')
    return (f"{st['n']:7d} {100 * st['clv_log_mean']:+8.2f}% "
            f"[{100 * lo:+6.2f},{100 * hi:+6.2f}] "
            f"{100 * st['clv_log_positive_rate']:6.1f}% "
            f"{100 * st['flat_roi']:+7.1f}% {100 * st['hit_rate']:6.1f}%{mark}")


HDR = (f'{"":26} {"n":>7} {"CLV(log)":>9} {"95% CI":>16} {"CLV>0":>7} '
       f'{"flatROI":>8} {"hit":>7}')


def render(rep):
    L = ['=' * 104,
         f'E0 — CLOSING LINE VALUE vs PINNACLE CLOSING — {rep["generated"]}',
         '=' * 104, '']
    L += [f'OOF rows: {rep["oof_rows_all"]}   with a closing price: {rep["rows_with_close"]}'
          f'   taken price is OPENING: {rep["rows_opening_price"]}',
          f'SCOREABLE (both): {rep["rows_scoreable"]}',
          f'closing source: {rep.get("close_source", {})}',
          '',
          'Rows whose taken price is itself a closing quote are excluded — CLV there',
          'would be closing-vs-closing and mechanically ~0.', '']

    b = rep['benchmarks']
    L += ['', 'ACCURACY LADDER (scoreable rows)',
          f'{"":22} {"RPS":>9} {"Brier":>9}',
          f'{"model":22} {b["rps_model"]:9.5f} {b["brier_model"]:9.5f}',
          f'{"taken price (opening)":22} {b["rps_taken_market"]:9.5f} {b["brier_taken_market"]:9.5f}',
          f'{"CLOSING price":22} {b["rps_closing"]:9.5f} {b["brier_closing"]:9.5f}', '']
    for nm, k in (('vs taken price', 'stack_vs_taken'), ('vs CLOSING price', 'stack_vs_closing')):
        s = rep[k]
        L.append(f'  nats added {nm:18} {s["nats_added_over_market"]:+.5f}   '
                 f'(blend weight on model {100 * s["blend_weight_model"]:+.0f}%)')
    L += ['', '  Gate 0 threshold: nats added over CLOSING > +0.005', '']

    L += ['', '=' * 104, 'SELECTION RULES', '=' * 104, HDR]
    for nm in ('model', 'favourite', 'random'):
        if nm in rep['selections']:
            L.append(f'{nm:26} {_fmt(rep["selections"][nm])}')
    if 'conviction_gate' in rep:
        L.append(f'{"conviction gate (replica)":26} {_fmt(rep["conviction_gate"])}')

    for key, title in (('strata_odds', 'BY ODDS BAND'), ('strata_ev', 'BY CLAIMED EV'),
                       ('strata_deviation', 'BY MODEL-MINUS-MARKET DEVIATION')):
        if rep.get(key):
            L += ['', '', '=' * 104, f'MODEL CLV — {title}', '=' * 104, HDR]
            for k, v in rep[key].items():
                L.append(f'{k:26} {_fmt(v)}')

    if rep.get('book_split'):
        ov = rep['overround']
        L += ['', '', '=' * 104,
              'ARTIFACT CONTROL — SAME-BOOK vs CROSS-BOOK',
              '=' * 104,
              f'  mean overround: taken {ov["taken_mean"]:.4f}   closing {ov["closing_mean"]:.4f}',
              '  by closing source: ' + '  '.join(
                  f'{k} {v:.4f}' for k, v in ov['by_close_source'].items()),
              '',
              '  Proportional devigging distorts favourites more at higher margin, so a',
              '  high-margin taken price vs a low-margin close manufactures exactly the',
              '  monotone ladder above. B365C rows are same-book and margin-matched:',
              '  whatever ladder survives THERE is real line movement.', '']
        for src, e in rep['book_split'].items():
            tag = 'SAME BOOK (B365 open -> B365 close)' if e['same_book'] \
                  else 'CROSS BOOK (B365 open -> Pinnacle close)'
            L += ['', f'  --- closing source {src}: {tag}, n={e["n"]} ---', HDR,
                  f'{"  overall":26} {_fmt(e["overall"])}']
            for k, v in e['by_odds'].items():
                L.append(f'{"  " + k:26} {_fmt(v)}')
            if 'conviction' in e:
                L.append(f'{"  conviction gate":26} {_fmt(e["conviction"])}')

    if rep.get('conviction_vs_odds_matched'):
        L += ['', '', '=' * 104,
              'CONFOUND CONTROL — CONVICTION GATE vs ODDS-MATCHED REST',
              '=' * 104,
              '  If the gate only works because it picks short prices, these two rows',
              '  match within each band and the confidence condition adds nothing.', '', HDR]
        for band, e in rep['conviction_vs_odds_matched'].items():
            L.append(f'{"odds " + band + " conviction":26} {_fmt(e["conviction"])}')
            L.append(f'{"odds " + band + " rest":26} {_fmt(e["rest"])}')
            L.append('')

    if rep.get('strata_league'):
        L += ['', '', '=' * 104, 'MODEL CLV — BY LEAGUE', '=' * 104, HDR]
        for k, v in sorted(rep['strata_league'].items(),
                           key=lambda kv: -kv[1]['clv_log_mean']):
            L.append(f'{k[:25]:26} {_fmt(v)}')
        n = len(rep['strata_league'])
        sig = sum(1 for v in rep['strata_league'].values() if v['beats_zero'])
        L += ['', f'  leagues: {n};  CLV CI excludes zero (positive): {sig};  '
                  f'expected by chance: {0.05 * n:.1f}',
              '  ** = CI excludes 0 positive, ++ = CI excludes 0 negative.']

    L += ['', '=' * 104,
          'READ: CLV is the leading indicator. Positive CLV with negative ROI is',
          'variance; negative CLV with positive ROI will regress. A model with no',
          'edge shows CLV ~ 0; a model systematically on the wrong side of steam',
          'shows NEGATIVE CLV, which is the adverse-selection prediction for the',
          'EV-gated subset specifically — check the EV and deviation strata for that.',
          '=' * 104]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--min-league', type=int, default=150,
                    help='Minimum scoreable rows for a league to get its own row.')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = prepare(cache=not args.no_cache)
    missing = [c for c in ('close_H', 'odds_is_closing') if c not in df.columns]
    if missing:
        sys.exit(f'Prepared frame lacks {missing}. Delete '
                 f'output/experiments/_prepared.pkl and re-run (the loader changed).')
    df = add_1x2_market(df)

    rep = build(df, args.splits, args.min_league)
    if 'error' in rep:
        sys.exit(f'{rep["error"]} (scoreable={rep.get("rows_scoreable")})')
    report = render(rep)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'clv_{ts}.json'), 'w') as f:
        json.dump(rep, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, f'clv_{ts}.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote output/experiments/clv_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
