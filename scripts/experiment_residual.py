"""E1 — residual model: train on what the market gets wrong.

Production trains a model and hopes its disagreements with the price are
useful. E0 showed they are not: the deviation is noise (corr with the market
residual -0.0031) and the stacked blend weight on the model is negative. This
inverts the formulation — instead of predicting the outcome and comparing to
the price afterwards, put the price in as the starting point and let the model
learn ONLY the correction.

Mechanically that is XGBoost's `base_margin`: with `multi:softprob`, setting
base_margin = log(devigged market probability) makes softmax(log p + f) the
output, so f = 0 reproduces the market exactly and the trees only ever learn
the log-correction f. An efficient market drives f to zero.

This is the clean null. The stacked test in experiment_odds_free asks the same
question post-hoc with one linear parameter; this asks it natively, with 800
trees and the full feature set free to find any correction that exists.

ARMS

  control              production features, plain multi:softprob, no base_margin
  residual_no_odds     base_margin, features minus B365*/IP_* (they ARE the
                       baseline now, so the roadmap spec drops them)
  residual_with_odds   base_margin, full production features — kept because the
                       model's error is odds-dependent, so the price level may
                       tell it WHERE to correct even though the price level is
                       already in the margin
  residual_placebo     base_margin, feature block row-permuted — the noise
                       floor of the whole pipeline. Any "edge" the real arms
                       show must exceed this.

POSITIVE CONTROL. Before touching football, the script fits a synthetic problem
with a correction of known size injected on top of a synthetic market. If the
method cannot recover that, a null on real data means nothing and the run
aborts. Measured on the reference build: logloss 1.0270 -> 0.8238.

SUCCESS CRITERIA (FOOTBALL_NEXT_STEPS E1)
  nats added over market > +0.005   (100x the +0.00000 production manages)
  delta RPS              >= -0.0012
  CLV vs closing         > 0

Writes output/experiments/residual_<ts>.{json,txt}. Never writes into models/.

Usage:
    python3 scripts/experiment_residual.py
    python3 scripts/experiment_residual.py --arms control residual_no_odds
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import prepare, add_1x2_market, stack_test, logit  # noqa: E402
from experiment_h2h import arm_features, feats_required                      # noqa: E402
from model_registry import get_spec                                          # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')
ODDS_FEATURES = ['B365H', 'B365D', 'B365A', 'IP_H', 'IP_D', 'IP_A']
ARMS = ('control', 'residual_no_odds', 'residual_with_odds', 'residual_placebo')
PLACEBO_SEED = 23

# d_rps is arm_rps - market_rps, so beating the market is NEGATIVE. The gate
# demands a real improvement (<= -0.0012), not merely the absence of a regression.
GATE = {'nats': 0.005, 'd_rps': -0.0012, 'clv': 0.0}


def devig(mat):
    inv = 1.0 / mat
    return inv / inv.sum(axis=1, keepdims=True)


def rps(P, y):
    Y = np.zeros_like(P)
    Y[np.arange(len(P)), y] = 1.0
    cP, cY = np.cumsum(P, 1), np.cumsum(Y, 1)
    return float(((cP[:, :-1] - cY[:, :-1]) ** 2).sum(1).mean() / (P.shape[1] - 1))


def brier(P, y):
    return float(((P - np.eye(3)[y]) ** 2).sum(1).mean())


def mlogloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(P)), y], 1e-9, 1)).mean())


def boot_ci(x, iters=4000, seed=0):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    return tuple(float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def positive_control():
    """Can base_margin recover a correction that is definitely there?"""
    import xgboost as xgb
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    mkt = rng.dirichlet([4, 3, 3], size=n)
    lg = np.log(mkt) + np.column_stack([0.8 * X[:, 0], np.zeros(n), -0.8 * X[:, 0]])
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    bm = np.log(np.clip(mkt, 1e-6, 1))
    m = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=60,
                          learning_rate=0.1, max_depth=3, tree_method='hist')
    m.fit(X, y, base_margin=bm)
    pr = m.predict_proba(X, base_margin=bm)
    return {'logloss_market': mlogloss(mkt, y),
            'logloss_market_plus_correction': mlogloss(pr, y),
            'recovered': bool(mlogloss(pr, y) < mlogloss(mkt, y) - 0.01)}


def arm_feature_list(arm, spec):
    feats = arm_features('1x2', 'base', spec)
    if arm in ('residual_no_odds',):
        feats = [f for f in feats if f not in ODDS_FEATURES]
    return feats


def run_arm(d, arm, feats, bm, y, splits, spec):
    """Out-of-fold probabilities for one arm."""
    tscv = TimeSeriesSplit(n_splits=splits)
    P = np.full((len(d), 3), np.nan)
    X = d[feats]
    use_margin = arm != 'control'
    for tr, te in tscv.split(d):
        model = spec.build()
        if use_margin:
            model.fit(X.iloc[tr], y[tr], base_margin=bm[tr])
            P[te] = model.predict_proba(X.iloc[te], base_margin=bm[te])
        else:
            model.fit(X.iloc[tr], y[tr])
            P[te] = model.predict_proba(X.iloc[te])
    return P


def build(df, arms, splits):
    spec = get_spec('1x2', 'xgboost')
    base_feats = arm_features('1x2', 'base', spec)
    dense = feats_required(base_feats)
    need = dense + ['target_1x2', 'mkt_H', 'mkt_D', 'mkt_A', 'B365H', 'B365D', 'B365A']
    d = df.dropna(subset=need).copy().sort_values('date').reset_index(drop=True)
    if spec.uses_categorical and 'league_cat' in d.columns:
        d['league_cat'] = d['league_cat'].astype('category')

    y = d['target_1x2'].values.astype(int)
    mkt = d[['mkt_H', 'mkt_D', 'mkt_A']].values          # devigged taken price
    bm = np.log(np.clip(mkt, 1e-6, 1.0))

    has_close = d[['close_H', 'close_D', 'close_A']].notna().all(axis=1).values
    close_dv = np.full_like(mkt, np.nan)
    close_dv[has_close] = devig(d.loc[has_close, ['close_H', 'close_D', 'close_A']].values)
    # E0's rule: CLV is only meaningful where the taken price is an OPENING quote.
    clv_ok = has_close & (d['odds_is_closing'] == False).values   # noqa: E712

    rep = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'rows': int(len(d)),
           'splits': splits, 'clv_scoreable': int(clv_ok.sum()),
           'positive_control': positive_control(),
           'market': {'rps': rps(mkt, y), 'brier': brier(mkt, y),
                      'logloss': mlogloss(mkt, y)}}
    if not rep['positive_control']['recovered']:
        rep['error'] = 'positive control FAILED — base_margin did not recover a known correction'
        return rep
    if clv_ok.sum():
        rep['market']['closing_rps'] = rps(close_dv[clv_ok], y[clv_ok])

    rng = np.random.default_rng(PLACEBO_SEED)
    rep['arms'] = {}
    for arm in arms:
        feats = arm_feature_list(arm, spec)
        dd = d
        if arm == 'residual_placebo':
            # Permute the feature block as a unit — same rows, same marginals,
            # no relationship to the fixture. Assign COLUMN BY COLUMN: a 2-D
            # `df[feats].values` collapses to object dtype the moment the block
            # contains the `league_cat` category column, and XGBoost then
            # rejects every feature as object.
            dd = d.copy()
            order = rng.permutation(len(dd))
            for c in feats:
                dd[c] = d[c].iloc[order].values
        print(f'\n=== {arm} ({len(feats)} features, '
              f'base_margin={"yes" if arm != "control" else "no"}) ===')
        P = run_arm(dd, arm, feats, bm, y, splits, spec)
        ok = np.isfinite(P).all(1)
        pick = np.where(ok, P.argmax(1), 0)
        r = np.arange(len(P))

        entry = {
            'n_features': len(feats), 'n_scored': int(ok.sum()),
            'rps': rps(P[ok], y[ok]), 'brier': brier(P[ok], y[ok]),
            'logloss': mlogloss(P[ok], y[ok]),
            # How far the arm moves away from the price at all. ~0 means the
            # trees found nothing to correct, which is a clean null rather than
            # a broken run.
            'mean_abs_correction': float(np.abs(P[ok] - mkt[ok]).mean()),
            'max_abs_correction': float(np.abs(P[ok] - mkt[ok]).max()),
        }
        entry['stack_vs_market'] = stack_test(
            P[ok][np.arange(ok.sum()), pick[ok]], mkt[ok][np.arange(ok.sum()), pick[ok]],
            (pick[ok] == y[ok]).astype(int))
        entry['nats_over_market'] = entry['stack_vs_market']['nats_added_over_market']
        entry['d_rps_vs_market'] = entry['rps'] - rep['market']['rps']

        m = ok & clv_ok
        if m.sum() > 500:
            rr = np.arange(m.sum())
            pk = P[m].argmax(1)
            clv = (np.log(close_dv[m][rr, pk]) - np.log(mkt[m][rr, pk]))
            lo, hi = boot_ci(clv)
            entry['clv'] = {'n': int(m.sum()), 'mean': float(clv.mean()),
                            'ci': [lo, hi], 'beats_zero': bool(lo > 0)}
            entry['stack_vs_closing'] = stack_test(
                P[m][rr, pk], close_dv[m][rr, pk], (pk == y[m]).astype(int))

        entry['gates'] = {
            'nats': bool(entry['nats_over_market'] > GATE['nats']),
            'd_rps': bool(entry['d_rps_vs_market'] <= GATE['d_rps']),
            'clv': bool(entry.get('clv', {}).get('mean', -1) > GATE['clv']),
        }
        entry['passes_all_gates'] = all(entry['gates'].values())
        rep['arms'][arm] = entry
    return rep


def render(rep):
    L = ['=' * 100, f'E1 — RESIDUAL MODEL AGAINST THE MARKET — {rep["generated"]}', '=' * 100, '']
    pc = rep['positive_control']
    L += [f'Positive control: synthetic market logloss {pc["logloss_market"]:.4f} -> '
          f'{pc["logloss_market_plus_correction"]:.4f} with base_margin  '
          f'=> correction recovered: {pc["recovered"]}',
          '  (if this failed, a null below would be a bug rather than a finding)', '']
    if 'error' in rep:
        return '\n'.join(L + ['', 'ABORTED: ' + rep['error']])
    m = rep['market']
    L += [f'rows {rep["rows"]}   CLV-scoreable {rep["clv_scoreable"]}',
          f'market baseline: RPS {m["rps"]:.5f}  Brier {m["brier"]:.5f}  '
          f'logloss {m["logloss"]:.5f}'
          + (f'   closing RPS {m["closing_rps"]:.5f}' if 'closing_rps' in m else ''), '']
    L += ['', f'{"arm":22} {"feat":>5} {"RPS":>9} {"dRPS":>9} {"Brier":>9} '
              f'{"nats/mkt":>10} {"|corr|":>8} {"CLV":>9} {"gates":>7}']
    for arm, e in rep['arms'].items():
        clv = f'{100 * e["clv"]["mean"]:+8.3f}%' if 'clv' in e else f'{"-":>9}'
        g = e['gates']
        gs = ''.join('Y' if g[k] else '.' for k in ('nats', 'd_rps', 'clv'))
        L.append(f'{arm:22} {e["n_features"]:5d} {e["rps"]:9.5f} '
                 f'{e["d_rps_vs_market"]:+9.5f} {e["brier"]:9.5f} '
                 f'{e["nats_over_market"]:+10.5f} {e["mean_abs_correction"]:8.4f} '
                 f'{clv} {gs:>7}')
    L += ['', '  gates: nats>+0.005 / dRPS<=-0.0012 (beats market) / CLV>0   (Y=pass, .=fail)',
          '  |corr| = mean |arm probability - market probability|; ~0 means the trees',
          '  found nothing to correct, which is a clean null, not a broken run.', '']
    for arm, e in rep['arms'].items():
        s = e['stack_vs_market']
        L += ['', f'  --- {arm} ---',
              f'    blend weight on model vs market  {100 * s["blend_weight_model"]:+.0f}%',
              f'    nats added over market           {s["nats_added_over_market"]:+.5f}',
              f'    mean/max |correction|            {e["mean_abs_correction"]:.4f} / '
              f'{e["max_abs_correction"]:.4f}']
        if 'stack_vs_closing' in e:
            L.append(f'    nats added over CLOSING          '
                     f'{e["stack_vs_closing"]["nats_added_over_market"]:+.5f}')
        if 'clv' in e:
            c = e['clv']
            L.append(f'    CLV vs closing                   {100 * c["mean"]:+.3f}% '
                     f'[{100 * c["ci"][0]:+.3f},{100 * c["ci"][1]:+.3f}]  n={c["n"]}')
        L.append(f'    PASSES ALL GATES: {e["passes_all_gates"]}')
    L += ['', '=' * 100,
          'READ: the residual arms must beat BOTH the control and the placebo. A',
          'placebo that matches a real arm means the formulation is measuring its own',
          'noise. A |corr| near zero with nats near zero is the honest null: given the',
          'price, these features say nothing about what the price got wrong.',
          '=' * 100]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = add_1x2_market(prepare(cache=not args.no_cache))
    rep = build(df, args.arms, args.splits)
    report = render(rep)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'residual_{ts}.json'), 'w') as f:
        json.dump(rep, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, f'residual_{ts}.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote output/experiments/residual_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
