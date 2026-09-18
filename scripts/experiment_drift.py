"""E2 stage 1 — does open-to-close line movement carry usable information?

The only test left in the E programme aimed at information the price does not
already hold. Three questions, deliberately separated, because the first is
nearly tautological and only the second is actionable.

  A. HOW BIG IS THE PRIZE?  Fit `y ~ logit(P_open) + drift`, where
     `drift = logit(P_close) - logit(P_open)`. E0 already established the close
     is more accurate than the open (Brier 0.59657 vs 0.59870), so a non-zero
     drift coefficient is guaranteed by construction — this part measures its
     SIZE, i.e. the ceiling if drift could be anticipated perfectly.

  B. IS DRIFT PREDICTABLE AT SERVE TIME?  The crux. At prediction time we hold
     the opening price and no closing quote, so drift is only exploitable if it
     can be forecast from information we already have. Train an out-of-fold
     regressor on the production feature set with drift as the target, then ask
     whether acting on the forecast produces positive same-book CLV.

  C. DOES THE EXISTING MODEL ANTICIPATE DRIFT?  E0 found the production model
     has negative CLV, implying it leans against the market's movement. Measured
     directly here as corr(model deviation, drift).

SAME BOOK, ALWAYS. E0's controls showed that comparing an opening price from
one book to a closing price from another manufactures a monotone odds ladder
out of nothing but the margin gap. Both pairs here are within-book and
margin-stable: B365 overround 1.0677 -> 1.0671 (0.06pp), Pinnacle 1.0397 ->
1.0349 (0.5pp). Both are run; agreement between them is the robustness check.

CAVEAT ON THE PRIZE. football-data's "opening" quote is the earliest price of
the market, while this system predicts the night before kickoff — much closer
to the close. The movement still available to us is a fraction of what part A
measures, so part A is an UPPER BOUND, not a forecast of attainable edge.

Writes output/experiments/drift_<ts>.{json,txt}. Never writes into models/.

Usage:
    python3 scripts/experiment_drift.py
    python3 scripts/experiment_drift.py --books B365 --splits 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import prepare, logit          # noqa: E402
from experiment_h2h import arm_features, feats_required  # noqa: E402
from model_registry import get_spec                      # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')

BOOKS = {
    'B365': (['B365H', 'B365D', 'B365A'], ['B365CH', 'B365CD', 'B365CA']),
    'PS':   (['PSH', 'PSD', 'PSA'],       ['PSCH', 'PSCD', 'PSCA']),
}
OUTCOMES = ('home', 'draw', 'away')


def devig(mat):
    inv = 1.0 / mat
    return inv / inv.sum(axis=1, keepdims=True)


def boot_ci(x, iters=4000, seed=0):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    return tuple(float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def fit_logit_se(X, y):
    """Unregularised logistic fit + SEs from observed Fisher information."""
    lr = LogisticRegression(C=1e8, max_iter=5000).fit(X, y)
    Xd = np.column_stack([np.ones(len(X)), X])
    p = lr.predict_proba(X)[:, 1]
    W = p * (1 - p)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(Xd.T @ (Xd * W[:, None]))))[1:]
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return lr.coef_[0], se


def crossfit_ll(X, y, splits=5, seed=0):
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        pred[te] = LogisticRegression(C=1e8, max_iter=5000).fit(
            X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))


def part_a(popen, pclose, y_onehot, splits):
    """Size of the prize, per outcome: what drift adds over the opening price."""
    out = {}
    for i, name in enumerate(OUTCOMES):
        lo = logit(popen[:, i])
        dr = logit(pclose[:, i]) - lo
        yy = y_onehot[:, i].astype(int)
        coef, se = fit_logit_se(np.column_stack([lo, dr]), yy)
        ll_base = crossfit_ll(lo[:, None], yy, splits)
        ll_both = crossfit_ll(np.column_stack([lo, dr]), yy, splits)
        out[name] = {
            'n': int(len(yy)),
            'drift_sd': float(dr.std()),
            'coef_open': float(coef[0]),
            'coef_drift': float(coef[1]), 'se_drift': float(se[1]),
            'z_drift': float(coef[1] / se[1]) if se[1] else float('nan'),
            'logloss_open_only': float(ll_base),
            'logloss_open_plus_drift': float(ll_both),
            'nats_added_by_drift': float(ll_base - ll_both),
        }
    return out


def oof_drift_forecast(d, feats, target, splits, spec):
    """Out-of-fold prediction of the drift target from serve-time features."""
    import xgboost as xgb
    tscv = TimeSeriesSplit(n_splits=splits)
    pred = np.full(len(d), np.nan)
    X, yv = d[feats], d[target].values
    for tr, te in tscv.split(d):
        m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300,
                             learning_rate=0.05, max_depth=4, subsample=0.9,
                             colsample_bytree=0.8, tree_method='hist',
                             enable_categorical=spec.uses_categorical, seed=42)
        m.fit(X.iloc[tr], yv[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def analyse_book(df, book, splits, seed=0):
    ocols, ccols = BOOKS[book]
    need = ocols + ccols + ['target_1x2']
    if not all(c in df.columns for c in need):
        return {'error': f'{book}: missing columns'}
    d = df.copy()
    for c in ocols + ccols:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    m = (d[ocols].notna().all(1) & d[ccols].notna().all(1)
         & (d[ocols] > 1).all(1) & (d[ccols] > 1).all(1) & d['target_1x2'].notna())
    d = d[m].sort_values('date').reset_index(drop=True)
    if len(d) < 2000:
        return {'error': f'{book}: only {len(d)} rows'}

    O, C = d[ocols].values, d[ccols].values
    popen, pclose = devig(O), devig(C)
    y = d['target_1x2'].values.astype(int)
    Y = np.eye(3)[y]

    rep = {
        'book': book, 'n': int(len(d)),
        'overround_open': float((1 / O).sum(1).mean()),
        'overround_close': float((1 / C).sum(1).mean()),
        'brier_open': float(((popen - Y) ** 2).sum(1).mean()),
        'brier_close': float(((pclose - Y) ** 2).sum(1).mean()),
    }

    # ---- A: the prize -----------------------------------------------------
    rep['part_a'] = part_a(popen, pclose, Y, splits)

    # ---- B: is drift forecastable from serve-time information? ------------
    spec = get_spec('1x2', 'xgboost')
    feats = arm_features('1x2', 'base', spec)
    dense = feats_required(feats)
    d['_drift_H'] = logit(pclose[:, 0]) - logit(popen[:, 0])
    d['_drift_A'] = logit(pclose[:, 2]) - logit(popen[:, 2])
    keep = d[dense].notna().all(axis=1)
    dd = d[keep].reset_index(drop=True)
    if spec.uses_categorical and 'league_cat' in dd.columns:
        dd['league_cat'] = dd['league_cat'].astype('category')

    rep['part_b'] = {'n': int(len(dd)), 'features': len(feats)}
    if len(dd) >= 2000:
        po, pc = devig(dd[ocols].values), devig(dd[ccols].values)
        for tgt in ('_drift_H', '_drift_A'):
            pr = oof_drift_forecast(dd, feats, tgt, splits, spec)
            ok = np.isfinite(pr)
            act = dd[tgt].values
            rep['part_b'][tgt.strip('_')] = {
                'pearson': float(pearsonr(pr[ok], act[ok])[0]),
                'spearman': float(spearmanr(pr[ok], act[ok])[0]),
                'r2': float(1 - ((act[ok] - pr[ok]) ** 2).sum()
                            / ((act[ok] - act[ok].mean()) ** 2).sum()),
                'pred_sd': float(pr[ok].std()), 'actual_sd': float(act[ok].std()),
            }
        # Acting on the forecast: back home when predicted home-drift exceeds
        # predicted away-drift, else away. CLV is same-book by construction.
        prH = oof_drift_forecast(dd, feats, '_drift_H', splits, spec)
        prA = oof_drift_forecast(dd, feats, '_drift_A', splits, spec)
        ok = np.isfinite(prH) & np.isfinite(prA)
        pick = np.where(prH[ok] >= prA[ok], 0, 2)
        r = np.arange(ok.sum())
        # CLV on the picked side: how far the devigged probability moved our way.
        clv = np.log(pc[ok][r, pick]) - np.log(po[ok][r, pick])
        won = (pick == dd['target_1x2'].values[ok].astype(int))
        odds = dd[ocols].values[ok][r, pick]
        lo, hi = boot_ci(clv)
        pnl = np.where(won, odds - 1.0, -1.0)
        rlo, rhi = boot_ci(pnl)
        rep['part_b']['forecast_strategy'] = {
            'n': int(ok.sum()), 'clv_log_mean': float(clv.mean()),
            'clv_log_ci': [lo, hi], 'clv_positive_rate': float((clv > 0).mean()),
            'flat_roi': float(pnl.mean()), 'flat_roi_ci': [rlo, rhi],
            'hit_rate': float(won.mean()), 'beats_zero': bool(lo > 0),
        }
        # Perfect-foresight ceiling on the same rows, for scale.
        pick_o = (np.log(pc[ok]) - np.log(po[ok]))[:, [0, 2]].argmax(1) * 2
        clv_o = np.log(pc[ok][r, pick_o]) - np.log(po[ok][r, pick_o])
        rep['part_b']['perfect_foresight_clv'] = float(clv_o.mean())

    # ---- C: does the production model anticipate drift? -------------------
    from experiment_odds_free import oof_predictions
    try:
        oof = oof_predictions(d, '1x2', 'base', n_splits=splits, feats=feats,
                              dropna_on=dense, carry=['_drift_H'])
        dev = oof['p_H'].values - oof['mkt_H'].values
        dr = oof['_drift_H'].values
        ok = np.isfinite(dev) & np.isfinite(dr)
        rep['part_c'] = {
            'n': int(ok.sum()),
            'pearson_dev_vs_drift': float(pearsonr(dev[ok], dr[ok])[0]),
            'spearman_dev_vs_drift': float(spearmanr(dev[ok], dr[ok])[0]),
        }
    except Exception as e:                       # pragma: no cover
        rep['part_c'] = {'error': str(e)}
    return rep


def render(reps):
    L = ['=' * 96, 'E2 STAGE 1 — IS OPEN-TO-CLOSE LINE MOVEMENT USABLE?', '=' * 96, '',
         'Same-book throughout (E0 showed cross-book devigging fabricates results).',
         'Part A is an UPPER BOUND: football-data "opening" is the market open, while',
         'this system bets the night before, so far less movement remains available.', '']
    for rep in reps:
        if 'error' in rep:
            L += ['', f'### {rep.get("book", "?")}: {rep["error"]}']
            continue
        L += ['', '=' * 96,
              f'### BOOK {rep["book"]}   n={rep["n"]}   '
              f'overround {rep["overround_open"]:.4f} -> {rep["overround_close"]:.4f}',
              f'    Brier: open {rep["brier_open"]:.5f} -> close {rep["brier_close"]:.5f}',
              '=' * 96, '',
              'A. SIZE OF THE PRIZE — what drift adds on top of the opening price',
              f'   {"outcome":8} {"drift sd":>9} {"coef":>9} {"se":>8} {"z":>8} {"nats added":>12}']
        for nm, v in rep['part_a'].items():
            L.append(f'   {nm:8} {v["drift_sd"]:9.4f} {v["coef_drift"]:+9.4f} '
                     f'{v["se_drift"]:8.4f} {v["z_drift"]:+8.1f} '
                     f'{v["nats_added_by_drift"]:+12.5f}')
        b = rep.get('part_b', {})
        L += ['', 'B. IS DRIFT FORECASTABLE FROM SERVE-TIME FEATURES? (the crux)',
              f'   rows {b.get("n")}, features {b.get("features")}']
        for k in ('drift_H', 'drift_A'):
            if k in b:
                v = b[k]
                L.append(f'   {k:8} pearson {v["pearson"]:+.4f}  spearman {v["spearman"]:+.4f}'
                         f'  R2 {v["r2"]:+.4f}  pred sd {v["pred_sd"]:.4f}'
                         f' vs actual {v["actual_sd"]:.4f}')
        if 'forecast_strategy' in b:
            f = b['forecast_strategy']
            lo, hi = f['clv_log_ci']
            mark = '  **' if f['beats_zero'] else ''
            L += ['',
                  f'   acting on the forecast: n={f["n"]}  '
                  f'CLV {100 * f["clv_log_mean"]:+.3f}% [{100 * lo:+.3f},{100 * hi:+.3f}]'
                  f'  CLV>0 {100 * f["clv_positive_rate"]:.1f}%'
                  f'  flatROI {100 * f["flat_roi"]:+.1f}%  hit {100 * f["hit_rate"]:.1f}%{mark}',
                  f'   perfect-foresight ceiling on the same rows: '
                  f'CLV {100 * b["perfect_foresight_clv"]:+.3f}%']
        c = rep.get('part_c', {})
        if 'pearson_dev_vs_drift' in c:
            L += ['', 'C. DOES THE PRODUCTION MODEL ANTICIPATE DRIFT?',
                  f'   corr(model deviation from market, actual drift) = '
                  f'{c["pearson_dev_vs_drift"]:+.4f} pearson / '
                  f'{c["spearman_dev_vs_drift"]:+.4f} spearman   (n={c["n"]})']
    L += ['', '=' * 96,
          'READ: A being large is expected and proves nothing — the close is more',
          'accurate than the open by construction. The decision rests on B: unless the',
          'forecast captures a real share of the perfect-foresight ceiling AND its CLV',
          'CI excludes zero, there is nothing to act on and stage 2 should not be built.',
          '=' * 96]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--books', nargs='+', default=['B365', 'PS'], choices=list(BOOKS))
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = prepare(cache=not args.no_cache)
    reps = [analyse_book(df, b, args.splits) for b in args.books]
    report = render(reps)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'drift_{ts}.json'), 'w') as f:
        json.dump(reps, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, f'drift_{ts}.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote output/experiments/drift_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
