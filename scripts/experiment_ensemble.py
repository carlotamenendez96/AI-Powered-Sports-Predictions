"""E5 — multi-paradigm stacked ensemble, run as a cheap kill first.

The proposal: a Level-0 zoo spanning boosting, bagging, a generative model and a
linear anchor, combined by a constrained Level-1 meta-learner (NNLS on the
simplex), on the theory that decorrelated errors cancel and the blend calibrates
better than any single paradigm.

Its own stated diversity criterion is the cheapest thing to test and it is a
KILL GATE: pairwise residual correlation between Level-0 models must be < 0.70.
If five estimators trained on the same 59 features — of which the market price
is the dominant one — all make the same errors, no meta-learner can help, and
there is no reason to build CatBoost or a Dixon-Coles fitter to find that out.
So this runs the cheap members first and reports the correlation matrix before
anything expensive is contemplated.

LEVEL-0 ZOO (all cheap, all out-of-fold on the production feature set)
    xgboost            production, via the registry (tuned params)
    extratrees         bagging, 500 trees
    randomforest       bagging with bootstrap, via the registry
    adaboost           adaptive boosting on shallow trees (SAMME)
    elasticnet_lr      multinomial logistic, L1/L2 mix — the linear anchor
    xgboost_oddsfree   XGBoost minus B365*/IP_* — a CHEAP PROXY for the
                       generative member's role. Dixon-Coles is the one
                       candidate that structurally cannot see the price, and
                       fitting it per-league is weeks of work; an odds-free
                       booster is the same "blind to the market" idea for one
                       extra fit. It is a proxy and is labelled as one.

LEVEL-1
    soft_vote            equal weights (the naive baseline the proposal rejects)
    nnls_simplex         weights >= 0, sum to 1, fit to minimise multiclass
                         Brier — preserves the simplex by construction
    nnls_with_market     the same, with THE MARKET as an extra candidate

That last one is the decisive variant and it is not in the original proposal.
E0 and E1 both found the optimal weight on the model GIVEN the price is
negative; a simplex constraint cannot express a negative weight, so it would
instead put ~all mass on the market. Including the market as a candidate lets
the meta-learner say so directly rather than being forced to blend models it
does not want.

The meta-learner is CROSS-FIT (KFold over the OOF rows): a stacker scored on
the rows its weights were fitted on is grading its own homework, and with the
effect sizes in this repo that bias is the same order as the effect.

SUCCESS CRITERIA (user proposal / FOOTBALL_NEXT_STEPS E5)
    pairwise Level-0 residual correlation < 0.70   <- kill gate, checked first
    delta Brier and delta RPS <= -1.2% vs control
    CLV > 0

Writes output/experiments/ensemble_<ts>.{json,txt}. Never writes into models/.

Usage:
    python3 scripts/experiment_ensemble.py
    python3 scripts/experiment_ensemble.py --models xgboost extratrees elasticnet_lr
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import prepare, add_1x2_market, stack_test   # noqa: E402
from experiment_h2h import arm_features, feats_required                # noqa: E402
from model_registry import get_spec                                    # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')
ODDS_FEATURES = ['B365H', 'B365D', 'B365A', 'IP_H', 'IP_D', 'IP_A']
K = 3
DIVERSITY_GATE = 0.70
IMPROVEMENT_GATE = -0.012        # -1.2% relative on Brier and RPS

ZOO = ('xgboost', 'extratrees', 'randomforest', 'adaboost',
       'elasticnet_lr', 'xgboost_oddsfree')


def _tabular(est):
    """Impute + scale for estimators that cannot take NaNs or raw scales."""
    return Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler()),
                     ('clf', est)])


def build_model(name):
    """(estimator, uses_categorical, drop_odds)."""
    if name == 'xgboost':
        return get_spec('1x2', 'xgboost').build(), True, False
    if name == 'xgboost_oddsfree':
        return get_spec('1x2', 'xgboost').build(), True, True
    if name == 'randomforest':
        return get_spec('1x2', 'rf').build(), False, False
    if name == 'extratrees':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', ExtraTreesClassifier(n_estimators=500, max_features='sqrt',
                                         min_samples_leaf=5, n_jobs=-1,
                                         random_state=0)),
        ]), False, False
    if name == 'adaboost':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=300, learning_rate=0.05, random_state=0)),
        ]), False, False
    if name == 'elasticnet_lr':
        return _tabular(LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.5,
            max_iter=3000, random_state=0)), False, False
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rps(P, y):
    Y = np.eye(K)[y]
    return float((((np.cumsum(P, 1)[:, :-1] - np.cumsum(Y, 1)[:, :-1]) ** 2).sum(1).mean())
                 / (K - 1))


def brier(P, y):
    return float(((P - np.eye(K)[y]) ** 2).sum(1).mean())


def mlogloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(P)), y], 1e-9, 1)).mean())


def boot_ci(x, iters=4000, seed=0):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    return tuple(float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


# --------------------------------------------------------------------------- #
# level-0
# --------------------------------------------------------------------------- #
def oof_level0(d, name, feats, y, splits):
    est, uses_cat, drop_odds = build_model(name)
    f = [c for c in feats if c not in ODDS_FEATURES] if drop_odds else list(feats)
    if not uses_cat:
        f = [c for c in f if c != 'league_cat']
    X = d[f]
    if uses_cat and 'league_cat' in f:
        X = X.copy()
        X['league_cat'] = X['league_cat'].astype('category')
    P = np.full((len(d), K), np.nan)
    tscv = TimeSeriesSplit(n_splits=splits)
    for tr, te in tscv.split(d):
        est, _, _ = build_model(name)        # fresh estimator per fold
        est.fit(X.iloc[tr], y[tr])
        P[te] = est.predict_proba(X.iloc[te])
    return P, len(f)


# --------------------------------------------------------------------------- #
# level-1
# --------------------------------------------------------------------------- #
def fit_simplex_weights(stack, y):
    """argmin_w ||sum_i w_i P_i - Y||^2  s.t.  w >= 0, sum w = 1.

    Minimising squared error on the one-hot target is exactly the multiclass
    Brier score, and the simplex constraint means the blend is a convex
    combination of probability vectors — so it lands on the simplex without any
    renormalisation that would distort the calibration being blended.
    """
    m = len(stack)
    Y = np.eye(K)[y]

    def loss(w):
        P = np.tensordot(w, stack, axes=(0, 0))
        return ((P - Y) ** 2).sum(1).mean()

    w0 = np.full(m, 1.0 / m)
    res = minimize(loss, w0, method='SLSQP',
                   bounds=[(0.0, 1.0)] * m,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
                   options={'maxiter': 500, 'ftol': 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else w0


def crossfit_stack(stack, y, splits=5, seed=0):
    """Out-of-sample blend + the mean weight vector, KFold over OOF rows."""
    n = stack.shape[1]
    P = np.zeros((n, K))
    ws = []
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        w = fit_simplex_weights(stack[:, tr, :], y[tr])
        ws.append(w)
        P[te] = np.tensordot(w, stack[:, te, :], axes=(0, 0))
    return P, np.mean(ws, axis=0)


# --------------------------------------------------------------------------- #
def build(df, models, splits):
    spec = get_spec('1x2', 'xgboost')
    feats = arm_features('1x2', 'base', spec)
    dense = feats_required(feats)
    need = dense + ['target_1x2', 'mkt_H', 'mkt_D', 'mkt_A']
    d = df.dropna(subset=need).copy().sort_values('date').reset_index(drop=True)
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
           'splits': splits,
           'market': {'rps': rps(mkt, y), 'brier': brier(mkt, y),
                      'logloss': mlogloss(mkt, y)},
           'level0': {}}

    preds = {}
    for name in models:
        t0 = time.time()
        print(f'\n=== level-0: {name} ===')
        P, nfeat = oof_level0(d, name, feats, y, splits)
        ok = np.isfinite(P).all(1)
        preds[name] = P
        rep['level0'][name] = {
            'n_features': nfeat, 'rps': rps(P[ok], y[ok]), 'brier': brier(P[ok], y[ok]),
            'logloss': mlogloss(P[ok], y[ok]),
            'accuracy': float((P[ok].argmax(1) == y[ok]).mean()),
            'fit_seconds': round(time.time() - t0, 1),
        }
        print(f'    RPS {rep["level0"][name]["rps"]:.5f}  '
              f'Brier {rep["level0"][name]["brier"]:.5f}  '
              f'({rep["level0"][name]["fit_seconds"]}s)')

    ok = np.all([np.isfinite(P).all(1) for P in preds.values()], axis=0)
    Y = np.eye(K)[y]

    # ---- THE KILL GATE ----------------------------------------------------
    # Three views, because the first is the one the proposal specifies and the
    # third is the one that actually matters here.
    names = list(preds)
    corr = {'residual': {}, 'prediction': {}, 'deviation_from_market': {}}
    for a, b in itertools.combinations(names, 2):
        Pa, Pb = preds[a][ok], preds[b][ok]
        pairs = {
            'residual': ((Y[ok] - Pa).ravel(), (Y[ok] - Pb).ravel()),
            'prediction': (Pa.ravel(), Pb.ravel()),
            'deviation_from_market': ((Pa - mkt[ok]).ravel(), (Pb - mkt[ok]).ravel()),
        }
        for kind, (u, v) in pairs.items():
            corr[kind][f'{a} | {b}'] = float(np.corrcoef(u, v)[0, 1])
    rep['correlations'] = corr
    rep['diversity_gate'] = {
        'threshold': DIVERSITY_GATE,
        'max_residual_corr': max(corr['residual'].values()),
        'min_residual_corr': min(corr['residual'].values()),
        'pairs_over_threshold': sum(1 for v in corr['residual'].values()
                                    if v >= DIVERSITY_GATE),
        'pairs_total': len(corr['residual']),
    }
    rep['diversity_gate']['passes'] = bool(
        rep['diversity_gate']['max_residual_corr'] < DIVERSITY_GATE)

    # ---- level-1 ----------------------------------------------------------
    stack = np.stack([preds[n][ok] for n in names], axis=0)
    yo = y[ok]
    rep['level1'] = {}

    Psv = stack.mean(axis=0)
    rep['level1']['soft_vote'] = {'weights': {n: 1.0 / len(names) for n in names}}
    ens = {'soft_vote': Psv}

    Pn, wn = crossfit_stack(stack, yo, splits)
    rep['level1']['nnls_simplex'] = {'weights': dict(zip(names, map(float, wn)))}
    ens['nnls_simplex'] = Pn

    stack_m = np.concatenate([stack, mkt[ok][None, ...]], axis=0)
    Pm, wm = crossfit_stack(stack_m, yo, splits)
    rep['level1']['nnls_with_market'] = {
        'weights': dict(zip(names + ['MARKET'], map(float, wm)))}
    ens['nnls_with_market'] = Pm

    control = preds['xgboost'][ok]
    ctrl = {'rps': rps(control, yo), 'brier': brier(control, yo)}
    rep['control'] = ctrl

    for nm, P in list(ens.items()) + [('control_xgboost', control), ('market', mkt[ok])]:
        e = {'rps': rps(P, yo), 'brier': brier(P, yo), 'logloss': mlogloss(P, yo),
             'accuracy': float((P.argmax(1) == yo).mean())}
        e['d_rps_vs_control'] = (e['rps'] - ctrl['rps']) / ctrl['rps']
        e['d_brier_vs_control'] = (e['brier'] - ctrl['brier']) / ctrl['brier']
        pick = P.argmax(1)
        r = np.arange(len(P))
        e['stack_vs_market'] = stack_test(P[r, pick], mkt[ok][r, pick],
                                          (pick == yo).astype(int))
        sub = clv_ok[ok]
        if sub.sum() > 500:
            rr = np.arange(sub.sum())
            pk = P[sub].argmax(1)
            clv = np.log(close_dv[ok][sub][rr, pk]) - np.log(mkt[ok][sub][rr, pk])
            lo, hi = boot_ci(clv)
            e['clv'] = {'n': int(sub.sum()), 'mean': float(clv.mean()), 'ci': [lo, hi]}
        e['gate_improvement'] = bool(e['d_rps_vs_control'] <= IMPROVEMENT_GATE
                                     and e['d_brier_vs_control'] <= IMPROVEMENT_GATE)
        rep.setdefault('scored', {})[nm] = e
    return rep


def render(rep):
    L = ['=' * 100, f'E5 — MULTI-PARADIGM STACKED ENSEMBLE — {rep["generated"]}', '=' * 100,
         '', f'rows {rep["rows"]}   folds {rep["splits"]}', '']

    L += ['LEVEL-0 ZOO', f'{"model":20} {"feat":>5} {"RPS":>9} {"Brier":>9} '
                         f'{"logloss":>9} {"acc":>8} {"fit s":>8}']
    for n, e in rep['level0'].items():
        L.append(f'{n:20} {e["n_features"]:5d} {e["rps"]:9.5f} {e["brier"]:9.5f} '
                 f'{e["logloss"]:9.5f} {100 * e["accuracy"]:7.2f}% {e["fit_seconds"]:8.1f}')
    m = rep['market']
    L.append(f'{"MARKET":20} {"-":>5} {m["rps"]:9.5f} {m["brier"]:9.5f} {m["logloss"]:9.5f}')

    g = rep['diversity_gate']
    L += ['', '', '=' * 100,
          'KILL GATE — LEVEL-0 ERROR DIVERSITY (pairwise residual correlation)',
          '=' * 100,
          f'  threshold {g["threshold"]}   max {g["max_residual_corr"]:.4f}   '
          f'min {g["min_residual_corr"]:.4f}   '
          f'{g["pairs_over_threshold"]}/{g["pairs_total"]} pairs at or above it',
          f'  >> DIVERSITY GATE PASSES: {g["passes"]}', '']
    for kind, title in (('residual', 'residual correlation (as the proposal specifies)'),
                        ('prediction', 'raw prediction correlation'),
                        ('deviation_from_market', 'correlation of DEVIATION FROM THE MARKET')):
        L += [f'  --- {title} ---']
        for pair, v in sorted(rep['correlations'][kind].items(), key=lambda kv: -kv[1]):
            L.append(f'    {pair:44} {v:+.4f}')
        L.append('')
    L += ['  The third block is the decisive one: if every model deviates from the',
          '  price in the same direction, they are one model wearing six hats, and',
          '  averaging them cancels nothing.', '']

    L += ['', '=' * 100, 'LEVEL-1 WEIGHTS (cross-fit, simplex-constrained)', '=' * 100]
    for nm, e in rep['level1'].items():
        ws = '  '.join(f'{k}={v:.3f}' for k, v in e['weights'].items())
        L.append(f'  {nm:20} {ws}')

    L += ['', '', '=' * 100, 'SCORED', '=' * 100,
          f'{"":22} {"RPS":>9} {"Brier":>9} {"dRPS%":>8} {"dBrier%":>9} '
          f'{"acc":>8} {"nats/mkt":>10} {"CLV":>9} {"gate":>5}']
    for nm, e in rep['scored'].items():
        clv = f'{100 * e["clv"]["mean"]:+8.3f}%' if 'clv' in e else f'{"-":>9}'
        L.append(f'{nm:22} {e["rps"]:9.5f} {e["brier"]:9.5f} '
                 f'{100 * e["d_rps_vs_control"]:+7.2f}% {100 * e["d_brier_vs_control"]:+8.2f}% '
                 f'{100 * e["accuracy"]:7.2f}% '
                 f'{e["stack_vs_market"]["nats_added_over_market"]:+10.5f} {clv} '
                 f'{"Y" if e["gate_improvement"] else ".":>5}')
    L += ['', f'  improvement gate: dRPS and dBrier both <= {100 * IMPROVEMENT_GATE:.1f}% '
              f'vs control_xgboost', '']
    L += ['=' * 100,
          'READ: the diversity gate is the whole experiment. Six estimators fitted to',
          'the same 59 features, of which the price is dominant, are six estimates of',
          'one conditional expectation. Ensembling reduces VARIANCE; if the weight the',
          'meta-learner wants to give the models is near zero once the market is a',
          'candidate, the problem was never variance.',
          '=' * 100]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(ZOO), choices=list(ZOO))
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = add_1x2_market(prepare(cache=not args.no_cache))
    rep = build(df, args.models, args.splits)
    report = render(rep)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'ensemble_{ts}.json'), 'w') as f:
        json.dump(rep, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, f'ensemble_{ts}.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote output/experiments/ensemble_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
