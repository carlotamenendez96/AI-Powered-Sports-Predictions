"""Settled-bet performance report — overall, by lane, market, odds, league, month.

The counterpart to `run_edge_check.py`. Where that script asks "is the EV
signal real", this one asks the plainer question: where has the money actually
gone, and which of those differences survive their own error bars.

Every number carries a bootstrap 95% CI, because the per-bet return
distribution is heavy-tailed (one 5.0 winner moves a small league's ROI by
tens of points) and point estimates on 30-50 bets are close to meaningless.
Two findings from the first full run (2026-09-18, 2,606 settled bets) are why
the CIs are not optional:

  - The per-league table is mostly noise. 32 leagues had >=20 bets; 3 had a CI
    excluding zero against ~1.6 expected by chance, and none of those 3 were
    positive. League ROI also correlates with each league's ODDS MIX (Spearman
    -0.43), so part of the apparent league spread is the odds ladder below
    reappearing in disguise.
  - The odds ladder is real and monotone. Model confidence runs 61.6% -> 45.0%
    across bands where the actual hit rate runs 78.7% -> 4.3%, while the
    market's implied probability tracks reality at every band. That is the
    probability compression CLAUDE.md describes, measured on settled money,
    and it is why `EV = conf x odds - 1` manufactures the most "value" exactly
    where the model is most wrong.

Settled means WON/LOST only. CASHED_OUT and VOID are excluded — their P/L
reflects a cashout price or a refund, not whether the pick was right.

Reading the output: `**` marks a slice whose CI excludes zero on the upside,
`++` on the downside. Treat both with the multiple-comparisons warning the
league section prints; a slice singled out BECAUSE it was extreme will still
look extreme when you re-test it on a subset.

Writes `output/performance/<timestamp>.{json,txt}`. Never writes outside
output/; safe to re-run any time.

Usage:
    python3 scripts/report_performance.py
    python3 scripts/report_performance.py --since 2026-08-01 --min-bets 30
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
OUT_DIR = os.path.join(OUTPUT_DIR, 'performance')

ODDS_BANDS = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0),
              (3.0, 4.0), (4.0, 5.0), (5.0, 99.0)]
ODDS_CAPS = [10.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0]
LANES = ('value', 'conviction', 'model')
BOOTSTRAP_ITERS = 4000


def _f(v):
    try:
        return float(str(v).replace('%', '').replace('+', '').strip())
    except (TypeError, ValueError):
        return None


def load_settled_bets(since=None):
    """Every WON/LOST bet from active and archived slips."""
    rows = []
    for bf in (sorted(glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json')))
               + sorted(glob.glob(os.path.join(OUTPUT_DIR, 'history', 'bets_*.json')))):
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
            odds = _f(b.get('odds') if b.get('odds') is not None else b.get('odd'))
            stake = _f(b.get('stake'))
            # `profit` is the backward-compatible alias and is present on every
            # settled bet; `pnl` is missing on the oldest slips. Prefer whichever
            # exists rather than dropping those rows (36 of 2,606 as of
            # 2026-09-18) — they are real settled bets.
            pnl = _f(b.get('pnl')) if b.get('pnl') is not None else _f(b.get('profit'))
            if None in (odds, stake, pnl) or stake <= 0:
                continue
            rows.append({
                'date': date, 'league': b.get('league', '?'),
                'lane': b.get('lane', '?'), 'type': b.get('type', '?'),
                'conf': _f(b.get('conf')), 'ev': _f(b.get('ev')),
                'odds': odds, 'stake': stake, 'pnl': pnl,
                'won': b.get('status') == 'WON',
            })
    return rows


def agg(rows):
    if not rows:
        return {'bets': 0, 'stake': 0.0, 'pnl': 0.0, 'roi': None, 'winrate': None}
    stake = sum(r['stake'] for r in rows)
    pnl = sum(r['pnl'] for r in rows)
    return {
        'bets': len(rows),
        'stake': round(stake, 2),
        'pnl': round(pnl, 2),
        'roi': (pnl / stake) if stake else None,
        'winrate': sum(r['won'] for r in rows) / len(rows),
    }


def roi_ci(rows, iters=BOOTSTRAP_ITERS, seed=0):
    """Bootstrap 95% CI on ROI.

    Resamples bets, not days. A normal approximation on per-bet returns
    understates the interval badly at long odds, where the return is a rare
    large positive against many -1s.
    """
    if len(rows) < 2:
        return (None, None)
    st = np.array([r['stake'] for r in rows])
    pl = np.array([r['pnl'] for r in rows])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(iters, len(rows)))
    tot = st[idx].sum(axis=1)
    rois = np.where(tot > 0, pl[idx].sum(axis=1) / np.where(tot > 0, tot, 1), np.nan)
    return tuple(float(x) for x in np.nanpercentile(rois, [2.5, 97.5]))


def slice_stats(rows):
    a = agg(rows)
    lo, hi = roi_ci(rows)
    a['roi_ci_low'], a['roi_ci_high'] = lo, hi
    a['significant'] = (None if lo is None or hi is None
                        else ('positive' if lo > 0 else 'negative' if hi < 0 else None))
    return a


def line(label, st, width=34):
    if not st['bets']:
        return f'{label:<{width}} {"(no bets)":>12}'
    lo, hi = st['roi_ci_low'], st['roi_ci_high']
    ci = (f'[{100 * lo:+6.1f},{100 * hi:+6.1f}]' if lo is not None else f'{"":>15}')
    mark = {'positive': '  **', 'negative': '  ++', None: ''}[st['significant']]
    return (f'{label:<{width}} {st["bets"]:5d} {st["stake"]:9.0f} {st["pnl"]:+9.2f} '
            f'{100 * st["roi"]:+7.1f}% {ci} {100 * st["winrate"]:5.1f}%{mark}')


HDR = (f'{"":34} {"bets":>5} {"staked":>9} {"P/L":>9} {"ROI":>8} '
       f'{"95% CI":>16} {"hit":>6}')


def build(rows, min_bets):
    rep = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'settled_bets': len(rows), 'min_bets_for_league': min_bets}
    rep['overall'] = slice_stats(rows)
    rep['by_lane'] = {l: slice_stats([r for r in rows if r['lane'] == l]) for l in LANES}
    rep['by_market'] = {t: slice_stats([r for r in rows if r['type'] == t])
                        for t in sorted({r['type'] for r in rows})}
    rep['by_month'] = {m: slice_stats([r for r in rows if r['date'][:7] == m])
                       for m in sorted({r['date'][:7] for r in rows if r['date']})}

    # Odds bands, with the calibration read that makes them interpretable.
    bands = {}
    for lo, hi in ODDS_BANDS:
        sub = [r for r in rows if lo <= r['odds'] < hi]
        if not sub:
            continue
        st = slice_stats(sub)
        confs = [r['conf'] for r in sub if r['conf'] is not None]
        st['model_conf'] = float(np.mean(confs)) if confs else None
        st['market_implied'] = float(np.mean([1 / r['odds'] for r in sub]))
        st['actual'] = float(np.mean([r['won'] for r in sub]))
        st['model_gap'] = (st['actual'] - st['model_conf']) if confs else None
        st['market_gap'] = st['actual'] - st['market_implied']
        bands[f'{lo}-{hi}'] = st
    rep['by_odds_band'] = bands

    # Leagues, split at min_bets so a 4-bet league cannot headline the table.
    by_lg = defaultdict(list)
    for r in rows:
        by_lg[r['league']].append(r)
    big = {k: v for k, v in by_lg.items() if len(v) >= min_bets}
    rep['by_league'] = {k: slice_stats(v) for k, v in big.items()}
    for k, v in big.items():
        rep['by_league'][k]['mean_odds'] = float(np.mean([r['odds'] for r in v]))
        rep['by_league'][k]['share_odds_ge_3'] = float(np.mean([r['odds'] >= 3 for r in v]))
    small = [r for k, v in by_lg.items() if len(v) < min_bets for r in v]
    rep['leagues_below_threshold'] = {
        'leagues': len(by_lg) - len(big), **slice_stats(small)}
    rep['league_count'] = len(by_lg)

    # Is the league spread just the odds ladder wearing a hat?
    if len(big) >= 5:
        from scipy.stats import spearmanr
        names = list(big)
        y = np.array([rep['by_league'][n]['roi'] for n in names])
        mo = np.array([rep['by_league'][n]['mean_odds'] for n in names])
        sh = np.array([rep['by_league'][n]['share_odds_ge_3'] for n in names])
        r1, p1 = spearmanr(mo, y)
        r2, p2 = spearmanr(sh, y)
        n_sig = sum(1 for n in names if rep['by_league'][n]['significant'])
        rep['league_noise_check'] = {
            'leagues_tested': len(names),
            'ci_excludes_zero': n_sig,
            'expected_by_chance_at_95pct': round(0.05 * len(names), 1),
            'spearman_mean_odds_vs_roi': {'rho': float(r1), 'p': float(p1)},
            'spearman_share_long_odds_vs_roi': {'rho': float(r2), 'p': float(p2)},
        }

    # What an odds ceiling would have done. Post-hoc: the cap is chosen after
    # seeing the data, so these are optimistic by construction.
    rep['odds_cap_counterfactual'] = {
        'none': slice_stats(rows),
        **{str(c): slice_stats([r for r in rows if r['odds'] <= c]) for c in ODDS_CAPS},
    }
    return rep


def render(rep):
    L = ['=' * 100,
         f'SETTLED-BET PERFORMANCE — {rep["settled_bets"]} bets — {rep["generated"]}',
         '=' * 100, '',
         'Settled = WON/LOST. CASHED_OUT and VOID excluded (their P/L is a cashout',
         'price or a refund, not whether the pick was right).',
         '** = CI excludes zero, positive.   ++ = CI excludes zero, negative.', '']
    L += ['OVERALL', HDR, line('ALL SETTLED', rep['overall']), '']
    L += ['BY LANE', HDR] + [line(k, v) for k, v in rep['by_lane'].items()] + ['']
    L += ['BY MARKET', HDR] + [line(k, v) for k, v in rep['by_market'].items()] + ['']
    L += ['BY MONTH', HDR] + [line(k, v) for k, v in rep['by_month'].items()] + ['']

    L += ['', '=' * 100,
          'ODDS LADDER — what the model claimed vs what the market claimed vs what happened',
          '=' * 100,
          f'{"odds":>10} {"bets":>6} {"model":>8} {"market":>8} {"actual":>8} '
          f'{"model gap":>10} {"mkt gap":>9} {"ROI":>8}']
    for k, b in rep['by_odds_band'].items():
        mc = f'{100 * b["model_conf"]:7.1f}%' if b['model_conf'] is not None else f'{"-":>8}'
        mg = f'{100 * b["model_gap"]:+9.1f}pp' if b['model_gap'] is not None else f'{"-":>11}'
        L.append(f'{k:>10} {b["bets"]:6d} {mc} {100 * b["market_implied"]:7.1f}% '
                 f'{100 * b["actual"]:7.1f}% {mg} {100 * b["market_gap"]:+8.1f}pp '
                 f'{100 * b["roi"]:+7.1f}%')
    L += ['',
          'Read: if the model column is flat while actual swings, the model is not',
          'discriminating and the market is doing the work. EV = conf x odds - 1 then',
          'peaks exactly where the model is most overconfident.', '']

    L += ['', '=' * 100,
          f'BY LEAGUE (>= {rep["min_bets_for_league"]} settled bets, sorted by P/L)',
          '=' * 100, HDR]
    for k, v in sorted(rep['by_league'].items(), key=lambda kv: -kv[1]['pnl']):
        L.append(line(k[:33], v))
    sb = rep['leagues_below_threshold']
    L.append(line(f'({sb["leagues"]} leagues below threshold)', sb))

    nc = rep.get('league_noise_check')
    if nc:
        L += ['',
              f'  leagues tested: {nc["leagues_tested"]};  CI excludes zero: '
              f'{nc["ci_excludes_zero"]};  expected by chance: '
              f'{nc["expected_by_chance_at_95pct"]}',
              f'  ROI vs league mean odds:       Spearman rho '
              f'{nc["spearman_mean_odds_vs_roi"]["rho"]:+.3f} '
              f'(p={nc["spearman_mean_odds_vs_roi"]["p"]:.4f})',
              f'  ROI vs share of odds >= 3.0:   Spearman rho '
              f'{nc["spearman_share_long_odds_vs_roi"]["rho"]:+.3f} '
              f'(p={nc["spearman_share_long_odds_vs_roi"]["p"]:.4f})',
              '',
              '  CAUTION: with this many leagues some CIs exclude zero by chance, and a',
              '  negative rho above means part of the league spread is the odds ladder,',
              '  not league-specific skill. Re-testing a league you picked BECAUSE it was',
              '  extreme will confirm it whether or not the effect is real — that is what',
              '  conditioning on an extreme does. Do not size bets off this table.']

    L += ['', '', '=' * 100,
          'ODDS CAP COUNTERFACTUAL — refuse every bet above the cap, stakes unchanged',
          '=' * 100, HDR]
    for k, v in rep['odds_cap_counterfactual'].items():
        L.append(line(f'cap {k}', v))
    L += ['',
          'Post-hoc: the cap is picked after seeing these bets, so this overstates what',
          'a cap would earn forward. Fix a cap in advance and validate on new data.',
          '=' * 100]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default=None, help='Only bets on/after YYYY-MM-DD.')
    ap.add_argument('--min-bets', type=int, default=20,
                    help='Minimum settled bets for a league to get its own row '
                         '(default 20; below that the CI is uninformative).')
    ap.add_argument('--no-write', action='store_true',
                    help='Print the report without writing artifacts.')
    args = ap.parse_args()

    rows = load_settled_bets(args.since)
    if not rows:
        print('No settled bets found.')
        return
    rep = build(rows, args.min_bets)
    if args.since:
        rep['since'] = args.since
    report = render(rep)
    print(report)

    if not args.no_write:
        os.makedirs(OUT_DIR, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(OUT_DIR, f'{ts}.json'), 'w') as f:
            json.dump(rep, f, indent=2, default=float)
        with open(os.path.join(OUT_DIR, f'{ts}.txt'), 'w') as f:
            f.write(report)
        print(f'\nWrote output/performance/{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
