"""E3 — RPS-aligned custom objective for the 1X2 head.

The 1X2 target is ORDINAL (Home > Draw > Away) and every loss in this repo is
ordinality-blind: multi-logloss penalises "true Home, predicted Away" exactly as
hard as "true Home, predicted Draw". The Ranked Probability Score does not, and
RPS is the standard metric for this market.

    RPS = 1/(K-1) * sum_{i=1}^{K-1} (cumP_i - cumY_i)^2

It is squared error on CUMULATIVE probabilities, so the gradient and Hessian
through the softmax are analytic. Writing L = 0.5 * sum_i e_i^2 with
e_i = cumP_i - cumY_i:

    dL/dp_k = sum_{i >= k} e_i                       (p_k enters every cumP_i, i >= k)
    dL/dz_k = p_k * (dL/dp_k - sum_j p_j dL/dp_j)    (softmax Jacobian)

and the Gauss-Newton Hessian, dropping second derivatives of the residuals:

    dc_i/dz = sum_{j <= i} p_j * (onehot_j - p)
    h_k     = sum_i (dc_i/dz_k)^2

The gradient is verified against a numerical derivative in `self_check()` to
~1e-11, and the objective is verified to actually lower RPS on a synthetic
ordinal problem. Both run before the real experiment; a failure aborts, because
a null result from a silently-broken objective is worthless.

WHY TWO HYPERPARAMETER CONFIGURATIONS

`min_child_weight` thresholds the SUM OF HESSIANS in a node, and the two
objectives have very different Hessian scales — multi-logloss uses 2p(1-p)
(mean ~0.4) while the RPS Gauss-Newton Hessian is an order of magnitude
smaller. Running the RPS arm under `best_params_1x2.json` (min_child_weight=7,
tuned for logloss) would therefore impose far heavier regularisation on it and
the comparison would measure that, not the objective. So both arms run under:

    tuned    production best_params_1x2.json — the deployment-relevant number,
             but structurally biased toward logloss; read with that in mind
    neutral  one shared modest config, tuned for neither — the fair comparison

The measured mean Hessian per objective is reported so the bias is visible
rather than argued about.

SUCCESS CRITERIA (FOOTBALL_NEXT_STEPS E3)
  delta RPS <= -0.0025 (a ~1.2% improvement), AND
  no discrimination loss: accuracy and one-vs-rest AUC must not fall
  (the same guard `fit_league_calibrators` applies to a Platt fit, added after
  a calibrator that collapsed toward the base rate improved Brier/ECE/log-loss
  while destroying ordering).

PRIOR: this should improve RPS modestly and produce ZERO edge. The model
already matches the market's distributional spread (E1), so a better-shaped
loss refines shape, not information. Worth doing as metric hygiene, and
mandatory before any Asian-handicap pricing, where ordinality is the whole game.

Writes output/experiments/rps_<ts>.{json,txt}. Never writes into models/.

Usage:
    python3 scripts/experiment_rps.py
    python3 scripts/experiment_rps.py --configs neutral --splits 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import prepare, add_1x2_market, stack_test   # noqa: E402
from experiment_h2h import arm_features, feats_required                # noqa: E402
from model_registry import get_spec, _tuned                            # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')
K = 3

# Shared, deliberately un-tuned. Neither objective has an advantage here.
NEUTRAL_PARAMS = dict(
    objective='multi:softprob', num_class=K, n_estimators=400,
    learning_rate=0.05, max_depth=4, min_child_weight=1,
    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method='hist', enable_categorical=True, seed=42,
)

GATE_D_RPS = -0.0025


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _rps_grad_hess(Y, z):
    """Gradient and Gauss-Newton Hessian of RPS w.r.t. softmax margins."""
    p = _softmax(z)
    err = np.cumsum(p, 1)[:, :-1] - np.cumsum(Y, 1)[:, :-1]      # (n, K-1)

    gp = np.zeros_like(p)
    for k in range(p.shape[1]):
        gp[:, k] = err[:, k:].sum(1)              # p_k enters cumP_i for i >= k
    grad = p * (gp - (gp * p).sum(1, keepdims=True))

    eye = np.eye(p.shape[1])
    run = np.zeros_like(p)
    hess = np.zeros_like(p)
    for i in range(p.shape[1] - 1):
        run = run + (eye[i][None, :] * p[:, [i]] - p[:, [i]] * p)
        hess += run ** 2
    # XGBoost needs a strictly positive Hessian; reg_lambda keeps the leaf
    # weight finite where this floor binds.
    return grad, np.maximum(hess, 1e-6)


def rps_objective(y_true, y_pred):
    z = np.asarray(y_pred, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, K)
    Y = np.eye(K)[np.asarray(y_true).astype(int)]
    return _rps_grad_hess(Y, z)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rps(P, y):
    Y = np.eye(P.shape[1])[y]
    return float((((np.cumsum(P, 1)[:, :-1] - np.cumsum(Y, 1)[:, :-1]) ** 2).sum(1).mean())
                 / (P.shape[1] - 1))


def brier(P, y):
    return float(((P - np.eye(K)[y]) ** 2).sum(1).mean())


def mlogloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(P)), y], 1e-9, 1)).mean())


def ovr_auc(P, y):
    try:
        return float(roc_auc_score(y, P, multi_class='ovr', average='macro'))
    except ValueError:
        return float('nan')


def boot_ci(x, iters=4000, seed=0):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    return tuple(float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


# --------------------------------------------------------------------------- #
# Self-checks — a null from a broken objective is worthless
# --------------------------------------------------------------------------- #
def self_check():
    rng = np.random.default_rng(0)
    n = 8
    z = rng.normal(size=(n, K))
    Y = np.eye(K)[rng.integers(0, K, n)]

    def loss(zz):
        p = _softmax(zz)
        return 0.5 * ((np.cumsum(p, 1)[:, :-1] - np.cumsum(Y, 1)[:, :-1]) ** 2).sum(1)

    ana, _ = _rps_grad_hess(Y, z)
    num = np.zeros_like(z)
    eps = 1e-6
    for i in range(n):
        for k in range(K):
            zp, zm = z.copy(), z.copy()
            zp[i, k] += eps
            zm[i, k] -= eps
            num[i, k] = (loss(zp)[i] - loss(zm)[i]) / (2 * eps)
    grad_err = float(np.abs(ana - num).max())

    # Does the objective actually lower RPS on a problem with real ordinality?
    import xgboost as xgb
    m = 4000
    X = rng.normal(size=(m, 5))
    lin = X[:, 0] + 0.6 * rng.normal(size=m)
    yy = np.digitize(lin, np.quantile(lin, [1 / 3, 2 / 3]))
    common = dict(num_class=K, n_estimators=60, max_depth=3,
                  learning_rate=0.3, tree_method='hist')
    a = xgb.XGBClassifier(objective='multi:softprob', **common).fit(X, yy)
    b = xgb.XGBClassifier(objective=rps_objective, **common).fit(X, yy)
    r_ll, r_rps = rps(a.predict_proba(X), yy), rps(b.predict_proba(X), yy)
    return {
        'grad_max_abs_error': grad_err,
        'grad_ok': bool(grad_err < 1e-7),
        'synthetic_rps_logloss_trained': r_ll,
        'synthetic_rps_rps_trained': r_rps,
        'objective_lowers_rps': bool(r_rps < r_ll),
    }


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_arm(d, feats, y, splits, params, use_rps):
    import xgboost as xgb
    P = np.full((len(d), K), np.nan)
    hess_means = []
    tscv = TimeSeriesSplit(n_splits=splits)
    X = d[feats]
    for tr, te in tscv.split(d):
        p = dict(params)
        if use_rps:
            p['objective'] = rps_objective
        model = xgb.XGBClassifier(**p)
        model.fit(X.iloc[tr], y[tr])
        P[te] = model.predict_proba(X.iloc[te])
        # Hessian scale actually seen, so the min_child_weight bias is visible.
        z = model.predict(X.iloc[tr], output_margin=True)
        Y = np.eye(K)[y[tr]]
        if use_rps:
            _, h = _rps_grad_hess(Y, z)
        else:
            pp = _softmax(z)
            h = 2.0 * pp * (1.0 - pp)
        hess_means.append(float(h.mean()))
    return P, float(np.mean(hess_means))


def build(df, configs, splits):
    spec = get_spec('1x2', 'xgboost')
    feats = arm_features('1x2', 'base', spec)
    dense = feats_required(feats)
    need = dense + ['target_1x2', 'mkt_H', 'mkt_D', 'mkt_A']
    d = df.dropna(subset=need).copy().sort_values('date').reset_index(drop=True)
    if 'league_cat' in d.columns:
        d['league_cat'] = d['league_cat'].astype('category')
    y = d['target_1x2'].values.astype(int)
    mkt = d[['mkt_H', 'mkt_D', 'mkt_A']].values

    has_close = d[['close_H', 'close_D', 'close_A']].notna().all(axis=1).values
    clv_ok = has_close & (d['odds_is_closing'] == False).values     # noqa: E712
    close_dv = np.full_like(mkt, np.nan)
    if has_close.any():
        cm = d.loc[has_close, ['close_H', 'close_D', 'close_A']].values
        inv = 1.0 / cm
        close_dv[has_close] = inv / inv.sum(axis=1, keepdims=True)

    rep = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'rows': int(len(d)),
           'features': len(feats), 'splits': splits,
           'self_check': self_check(),
           'market': {'rps': rps(mkt, y), 'brier': brier(mkt, y),
                      'logloss': mlogloss(mkt, y), 'accuracy': float((mkt.argmax(1) == y).mean()),
                      'ovr_auc': ovr_auc(mkt, y)},
           'configs': {}}
    sc = rep['self_check']
    if not (sc['grad_ok'] and sc['objective_lowers_rps']):
        rep['error'] = 'self-check FAILED — objective is not correct; result would be meaningless'
        return rep

    tuned_params = dict(NEUTRAL_PARAMS)
    tuned_params.update(dict(objective='multi:softprob', num_class=K,
                             n_estimators=100, learning_rate=0.1, max_depth=5,
                             eval_metric='mlogloss', tree_method='hist',
                             enable_categorical=True))
    tuned_params.update(_tuned('1x2'))
    tuned_params.pop('eval_metric', None)

    param_sets = {'tuned': tuned_params, 'neutral': dict(NEUTRAL_PARAMS)}

    for cfg in configs:
        params = param_sets[cfg]
        rep['configs'][cfg] = {'params': {k: v for k, v in params.items()
                                          if k not in ('objective',)}, 'arms': {}}
        for arm, use_rps in (('logloss', False), ('rps', True)):
            print(f'\n=== config={cfg}  arm={arm} ===')
            P, hmean = run_arm(d, feats, y, splits, params, use_rps)
            ok = np.isfinite(P).all(1)
            Po, yo = P[ok], y[ok]
            e = {
                'rps': rps(Po, yo), 'brier': brier(Po, yo), 'logloss': mlogloss(Po, yo),
                'accuracy': float((Po.argmax(1) == yo).mean()), 'ovr_auc': ovr_auc(Po, yo),
                'mean_hessian': hmean, 'n_scored': int(ok.sum()),
                'mean_p_draw': float(Po[:, 1].mean()), 'sd_p_draw': float(Po[:, 1].std()),
                'sd_p_home': float(Po[:, 0].std()),
            }
            pick = Po.argmax(1)
            r = np.arange(len(Po))
            e['stack_vs_market'] = stack_test(Po[r, pick], mkt[ok][r, pick],
                                              (pick == yo).astype(int))
            m = ok & clv_ok
            if m.sum() > 500:
                rr = np.arange(m.sum())
                pk = P[m].argmax(1)
                clv = np.log(close_dv[m][rr, pk]) - np.log(mkt[m][rr, pk])
                lo, hi = boot_ci(clv)
                e['clv'] = {'n': int(m.sum()), 'mean': float(clv.mean()), 'ci': [lo, hi]}
            rep['configs'][cfg]['arms'][arm] = e

        a, b = rep['configs'][cfg]['arms']['logloss'], rep['configs'][cfg]['arms']['rps']
        d_rps = b['rps'] - a['rps']
        rep['configs'][cfg]['verdict'] = {
            'd_rps': d_rps,
            'd_rps_pct': 100 * d_rps / a['rps'],
            'd_accuracy': b['accuracy'] - a['accuracy'],
            'd_ovr_auc': b['ovr_auc'] - a['ovr_auc'],
            'gate_d_rps': bool(d_rps <= GATE_D_RPS),
            'gate_no_discrimination_loss': bool(b['accuracy'] >= a['accuracy']
                                                and b['ovr_auc'] >= a['ovr_auc']),
        }
        v = rep['configs'][cfg]['verdict']
        v['passes'] = bool(v['gate_d_rps'] and v['gate_no_discrimination_loss'])
    return rep


def render(rep):
    L = ['=' * 100, f'E3 — RPS-ALIGNED OBJECTIVE — {rep["generated"]}', '=' * 100, '']
    sc = rep['self_check']
    L += [f'Self-check: analytic vs numerical gradient max error '
          f'{sc["grad_max_abs_error"]:.2e} (ok={sc["grad_ok"]});  synthetic ordinal RPS '
          f'{sc["synthetic_rps_logloss_trained"]:.5f} -> '
          f'{sc["synthetic_rps_rps_trained"]:.5f} (lowers RPS='
          f'{sc["objective_lowers_rps"]})', '']
    if 'error' in rep:
        return '\n'.join(L + ['ABORTED: ' + rep['error']])
    m = rep['market']
    L += [f'rows {rep["rows"]}   features {rep["features"]}',
          f'market baseline: RPS {m["rps"]:.5f}  Brier {m["brier"]:.5f}  '
          f'acc {100 * m["accuracy"]:.2f}%  OvR-AUC {m["ovr_auc"]:.4f}', '']
    for cfg, c in rep['configs'].items():
        note = ('production params — tuned FOR logloss, so biased against the RPS arm'
                if cfg == 'tuned' else 'shared modest params — tuned for neither objective')
        L += ['', '=' * 100, f'CONFIG: {cfg}   ({note})', '=' * 100,
              f'{"arm":10} {"RPS":>9} {"Brier":>9} {"logloss":>9} {"acc":>8} '
              f'{"OvR-AUC":>9} {"meanHess":>9} {"nats/mkt":>10} {"CLV":>9}']
        for arm, e in c['arms'].items():
            clv = f'{100 * e["clv"]["mean"]:+8.3f}%' if 'clv' in e else f'{"-":>9}'
            L.append(f'{arm:10} {e["rps"]:9.5f} {e["brier"]:9.5f} {e["logloss"]:9.5f} '
                     f'{100 * e["accuracy"]:7.2f}% {e["ovr_auc"]:9.4f} '
                     f'{e["mean_hessian"]:9.4f} '
                     f'{e["stack_vs_market"]["nats_added_over_market"]:+10.5f} {clv}')
        v = c['verdict']
        L += ['',
              f'  delta RPS        {v["d_rps"]:+.5f}  ({v["d_rps_pct"]:+.2f}%)   '
              f'gate (<= {GATE_D_RPS}): {"PASS" if v["gate_d_rps"] else "FAIL"}',
              f'  delta accuracy   {100 * v["d_accuracy"]:+.2f}pp',
              f'  delta OvR-AUC    {v["d_ovr_auc"]:+.5f}',
              f'  discrimination guard: '
              f'{"PASS" if v["gate_no_discrimination_loss"] else "FAIL"}',
              f'  >> E3 PASSES: {v["passes"]}']
        a, b = c['arms']['logloss'], c['arms']['rps']
        L += ['',
              f'  draw handling:  P(draw) mean {a["mean_p_draw"]:.4f} -> {b["mean_p_draw"]:.4f}'
              f'   sd {a["sd_p_draw"]:.4f} -> {b["sd_p_draw"]:.4f}',
              f'  home spread:    sd P(home) {a["sd_p_home"]:.4f} -> {b["sd_p_home"]:.4f}',
              f'  Hessian scale:  {a["mean_hessian"]:.4f} (logloss) vs '
              f'{b["mean_hessian"]:.4f} (RPS)  — min_child_weight thresholds this sum, '
              f'which is why two configs are run']
    L += ['', '=' * 100,
          'READ: RPS is a metric-quality result, not an edge result. Even a clean pass',
          'changes nothing about beating the price — E0/E1 established the model adds',
          'nothing over the market regardless of loss shape. The reason to care is',
          'ordinal markets (Asian handicap), where mis-ranking Draw against Away costs',
          'real money and multi-logloss cannot see the difference.',
          '=' * 100]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=['tuned', 'neutral'],
                    choices=['tuned', 'neutral'])
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = add_1x2_market(prepare(cache=not args.no_cache))
    rep = build(df, args.configs, args.splits)
    report = render(rep)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'rps_{ts}.json'), 'w') as f:
        json.dump(rep, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, f'rps_{ts}.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote output/experiments/rps_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
