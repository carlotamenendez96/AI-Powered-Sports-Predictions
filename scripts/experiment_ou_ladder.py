"""E6 — is football's Over/Under worth betting as a LADDER (0.5 … 4.5)?

Euroleague bets the totals ladder: each book posts several lines and the
strategy picks the highest-EV (line, side) row. Football's O/U market is a
ladder too, and the production head is already a **Poisson regressor on total
goals**, so P(Over k.5) at every line is available analytically from the same
lambda -- no new model, no retrain. The question is whether betting a line
other than 2.5 is better, not whether we *can*.

Two parts, deliberately ordered so the cheap one can kill the expensive one.

PART 1 - CALIBRATION BY LINE (no odds, no assumptions)
    Ladder-shopping picks the line where the model most disagrees with the
    price. If the model's probabilities are only trustworthy near 2.5 -- where
    it was effectively tuned, since `target_ou` is defined at 2.5 -- then
    shopping systematically selects the lines it is *worst* at, and the answer
    is no before any odds are scraped. Measured against outcomes: predicted
    rate vs actual rate, bias, Brier, and a calibration slope per line.

PART 2 - LADDER-EV SIMULATION (one explicit assumption)
    Historical prices exist ONLY at 2.5 (football-data.co.uk carries no other
    totals), so a real ladder backtest is impossible offline. Instead the
    book's ladder is *modelled*: fit a single Poisson lambda_market to the
    devigged market P(Over 2.5), derive fair prices at every line, and re-apply
    the observed vig. That answers a precise question -- "if the book priced
    the ladder consistently with its own 2.5 line, would shopping beat just
    betting 2.5?" -- and isolates our model's disagreement from line-specific
    bookmaker mispricing.

    LIMITATION, stated plainly: real ladders are not Poisson-consistent, and
    the shape error at 0.5/4.5 is exactly where a book makes its margin. This
    part cannot find edge that comes from the book mispricing a specific line;
    only forward-scraping real ladders (the Euroleague mechanism, already
    built in ml_project/euroleague/fetch_euroleague_odds.py) could.

Usage:
    python3 scripts/experiment_ou_ladder.py [--splits 5] [--no-cache]
Writes output/experiments/ou_ladder_<ts>.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import (          # noqa: E402
    prepare, add_ou_market, feature_list, logit,
)
from model_registry import get_spec         # noqa: E402

LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')


def p_over(lam, line):
    """P(total goals > line) under Poisson(lam). Lines are k.5 so no ties."""
    return 1.0 - poisson.cdf(int(np.floor(line)), lam)


def oof_lambda(df, n_splits=5):
    """Out-of-fold Poisson lambda from the PRODUCTION o/u head.

    Mirrors experiment_odds_free.oof_predictions but keeps lambda instead of
    collapsing it to P(Over 2.5) -- the whole point here is the other lines.
    """
    spec = get_spec('ou', 'xgboost')
    feats = feature_list('ou', 'with_odds', spec)
    need = feats + ['total_goals', 'mkt_over', 'ou_over_odds', 'ou_under_odds']
    d = df.dropna(subset=need).copy().sort_values('date')
    if spec.uses_categorical and 'league_cat' in d.columns:
        d['league_cat'] = d['league_cat'].astype('category')
    print(f"  [ou] {len(d)} rows, {len(feats)} features")

    parts = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(d)):
        model = spec.build()
        model.fit(d.iloc[tr][feats], d.iloc[tr]['total_goals'])
        out = d.iloc[te][['date', 'league', 'total_goals', 'mkt_over',
                          'ou_over_odds', 'ou_under_odds']].copy()
        out['lam'] = model.predict(d.iloc[te][feats])
        out['fold'] = fold
        parts.append(out)
        print(f"    fold {fold + 1}/{n_splits} done ({len(te)} rows)")
    return pd.concat(parts, ignore_index=True)


def calibration_by_line(oof):
    """Part 1: does the Poisson lambda hold up away from 2.5?"""
    rows = []
    for line in LINES:
        p = np.clip(p_over(oof['lam'].values, line), 1e-6, 1 - 1e-6)
        y = (oof['total_goals'].values > line).astype(int)
        # Calibration slope: unregularised logistic fit of y on logit(p).
        # 1.0 = perfectly calibrated; <1 = over-spread (model more confident
        # than reality), the compression signature documented for the 1X2
        # head. sklearn rather than statsmodels, which is not a project dep.
        try:
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            lr.fit(logit(p).reshape(-1, 1), y)
            slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
        except Exception:
            slope = intercept = float('nan')
        rows.append({
            'line': line, 'n': int(len(y)),
            'pred_rate': float(p.mean()), 'actual_rate': float(y.mean()),
            'bias': float(p.mean() - y.mean()),
            'brier': float(np.mean((p - y) ** 2)),
            'slope': slope, 'intercept': intercept,
        })
    return rows


def _lambda_from_market(mkt_over):
    """Single Poisson lambda reproducing the book's devigged P(Over 2.5).

    P(Over 2.5) is monotone in lambda, so a bisection inverts it exactly.
    """
    lo, hi = np.full(len(mkt_over), 0.05), np.full(len(mkt_over), 8.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_low = p_over(mid, 2.5) < mkt_over
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return (lo + hi) / 2


def ladder_simulation(oof):
    """Part 2: shop the (modelled) ladder by EV vs always betting 2.5."""
    lam_mkt = _lambda_from_market(oof['mkt_over'].values)
    # Observed vig at 2.5, re-applied at every simulated line.
    overround = (1 / oof['ou_over_odds'].values) + (1 / oof['ou_under_odds'].values)
    total = oof['total_goals'].values
    lam = oof['lam'].values

    # Three arms: shop everything, shop only the lines adjacent to 2.5, or
    # never shop. The middle arm matters because the extreme lines are where
    # a small probability error is multiplied by the largest price.
    ARMS = {'ladder_all': LINES, 'ladder_near': [1.5, 2.5, 3.5], 'fixed_2_5': [2.5]}
    acc = {k: {'ev': [], 'ret': [], 'line': []} for k in ARMS}
    for i in range(len(oof)):
        cands = []
        for line in LINES:
            pm = np.clip(p_over(lam_mkt[i], line), 1e-4, 1 - 1e-4)
            pmod_o = np.clip(p_over(lam[i], line), 1e-6, 1 - 1e-6)
            # Fair price from the modelled book, then vig applied as at 2.5.
            odds_o, odds_u = 1.0 / (pm * overround[i]), 1.0 / ((1 - pm) * overround[i])
            won_o = total[i] > line
            for side, pmod, odds, won in (('Over', pmod_o, odds_o, won_o),
                                          ('Under', 1 - pmod_o, odds_u, not won_o)):
                cands.append((pmod * odds - 1.0, line, side, (odds - 1.0) if won else -1.0))
        for name, allowed in ARMS.items():
            pool = sorted([c for c in cands if c[1] in allowed], key=lambda c: -c[0])
            ev, line, _side, ret = pool[0]
            acc[name]['ev'].append(ev)
            acc[name]['ret'].append(ret)
            acc[name]['line'].append(line)

    out = {'n': int(len(oof))}
    for name in ARMS:
        ret = np.array(acc[name]['ret'])
        se = ret.std(ddof=1) / np.sqrt(len(ret))
        out[name] = {
            'mean_claimed_ev': float(np.mean(acc[name]['ev'])),
            'realised_roi': float(ret.mean()),
            'roi_se': float(se),
            'roi_ci95': [float(ret.mean() - 1.96 * se), float(ret.mean() + 1.96 * se)],
            'line_mix': {str(l): int(np.sum(np.array(acc[name]['line']) == l)) for l in LINES},
            'pct_off_2_5': float(np.mean(np.array(acc[name]['line']) != 2.5)),
        }
    return out


def render(cal, sim, meta):
    L = []
    A = L.append
    A("=" * 78)
    A("E6 — Over/Under as a LADDER (football)")
    A("=" * 78)
    A(f"rows: {meta['n']}   folds: {meta['splits']}   generated: {meta['ts']}")
    A("")
    A("PART 1 — calibration of the Poisson lambda at each line (no odds used)")
    A("-" * 78)
    A(f"{'line':>6} {'n':>7} {'pred':>8} {'actual':>8} {'bias':>8} {'brier':>8} {'slope':>7}")
    for r in cal:
        A(f"{r['line']:>6} {r['n']:>7} {r['pred_rate']:>8.3f} {r['actual_rate']:>8.3f} "
          f"{r['bias']:>+8.3f} {r['brier']:>8.4f} {r['slope']:>7.2f}")
    A("")
    A("  slope 1.0 = calibrated; <1 = over-spread (model more confident than reality).")
    A("")
    A("PART 2 — EV-shopping the modelled ladder vs always betting 2.5")
    A("-" * 78)
    A(f"{'arm':>12} {'claimed EV':>11} {'realised ROI':>13} {'95% CI':>22} {'off 2.5':>8}")
    for name in ('ladder_all', 'ladder_near', 'fixed_2_5'):
        r = sim[name]
        ci = f"[{r['roi_ci95'][0]:+.3f}, {r['roi_ci95'][1]:+.3f}]"
        A(f"{name:>12} {r['mean_claimed_ev']:>+11.4f} {r['realised_roi']:>+13.4f} "
          f"{ci:>22} {r['pct_off_2_5']:>7.0%}")
    A("")
    A(f"  line mix (shop-all): {sim['ladder_all']['line_mix']}")
    A("")
    A("  NOTE 1: the book's ladder is MODELLED as Poisson-consistent with its")
    A("  own 2.5 line (no historical prices exist at other lines). This cannot")
    A("  detect a book mispricing a specific line; only forward-scraped real")
    A("  ladders could.")
    A("  NOTE 2: because BOTH sides are Poisson, model and market differ only")
    A("  in lambda, and the probability RATIO is largest in the tail — so the")
    A("  shop-all arm lands on an extreme line by construction. Its line mix")
    A("  is an artifact of that; its realised ROI, scored on real outcomes,")
    A("  is not.")
    A("=" * 78)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    df = add_ou_market(prepare(cache=not a.no_cache))
    oof = oof_lambda(df, n_splits=a.splits)
    cal = calibration_by_line(oof)
    sim = ladder_simulation(oof)
    meta = {'n': int(len(oof)), 'splits': a.splits,
            'ts': datetime.now().isoformat(timespec='seconds'),
            'elapsed_s': round(time.time() - t0, 1)}

    text = render(cal, sim, meta)
    print("\n" + text)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'ou_ladder_{stamp}.json'), 'w') as f:
        json.dump({'meta': meta, 'calibration': cal, 'simulation': sim}, f, indent=2)
    with open(os.path.join(OUT_DIR, f'ou_ladder_{stamp}.txt'), 'w') as f:
        f.write(text + "\n")
    print(f"\nwrote output/experiments/ou_ladder_{stamp}.{{json,txt}}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
