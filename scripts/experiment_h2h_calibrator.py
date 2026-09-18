"""Can head-to-head work as a POST-PREDICTION calibrator instead of a feature?

scripts/experiment_h2h.py answered "does XGBoost gain anything from the h2h_*
columns" — no. That is not automatically the same question as this one. The
booster sees 65 columns through colsample_bytree=0.6, max_depth=3 and gamma=5;
a weak-but-real signal can be drowned there and still be recoverable by a
direct one-parameter fit applied to the finished probability. So test the
post-hoc form on its own terms.

Three nested fits per market, all on out-of-fold predictions:

    market          logit(y) ~ logit(p_market)
    market + h2h    logit(y) ~ logit(p_market) + <h2h term>
    model  + h2h    logit(y) ~ logit(p_model)  + <h2h term>

and, for the production-shaped version, the full stack with and without:

    stack           logit(y) ~ logit(p_model) + logit(p_market)
    stack + h2h     logit(y) ~ logit(p_model) + logit(p_market) + <h2h term>

Two separate questions, deliberately kept apart:

  1. `market + h2h` — is there anything in the head-to-head record that the
     PRICE has not already absorbed? If this coefficient is zero, no calibrator
     of any shape keyed on h2h can produce an edge, because this is the most
     direct extraction available. This is the decisive one.
  2. `model + h2h` — would a post-hoc h2h correction make the MODEL's own
     probabilities better, even if only by pushing them toward the price?
     A real yes here is worth something for pick quality even with no edge.

Coefficients are reported with a standard error from the observed Fisher
information, so "zero" is a claim with an interval rather than an eyeball.
Nats are 5-fold CROSS-FITTED (calibrator fit on 4 folds, scored on the held-out
one), because a calibrator scored on the rows it was fitted to always looks
good.

Usage:
    python3 scripts/experiment_h2h_calibrator.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import (          # noqa: E402
    prepare, add_1x2_market, add_ou_market, oof_predictions, logit,
)
from experiment_h2h import arm_features, feats_required   # noqa: E402
from feature_engineering import H2H_FEATURES, H2H_WINDOW  # noqa: E402
from model_registry import get_spec                       # noqa: E402

OOF_CACHE = os.path.join(PROJECT_ROOT, 'output', 'experiments', '_oof_base.pkl')

# (label, outcome, market column, model column, candidate h2h terms)
#
# Each row is one binary question posed in the orientation the h2h columns are
# already built in (home team's perspective), so no sign flipping is needed.
MARKETS = [
    ('O/U 2.5', 'over', 'mkt_over', 'p_over',
     ['h2h_ou_rate', 'h2h_goals', 'h2h_venue_goals']),
    ('1X2 home win', 'home', 'mkt_H', 'p_H',
     ['h2h_pts', 'h2h_gd', 'h2h_venue_pts']),
    ('1X2 draw', 'draw', 'mkt_D', 'p_D',
     ['h2h_draw_rate']),
]


def build_oof(splits=5):
    """Base-arm (no h2h features) OOF predictions for both heads, with the
    h2h columns carried alongside so a post-hoc fit can use them."""
    if os.path.exists(OOF_CACHE):
        print(f'Loading cached OOF from {OOF_CACHE}')
        return pd.read_pickle(OOF_CACHE)

    df = add_ou_market(add_1x2_market(prepare()))
    parts = {}
    for head in ('1x2', 'ou'):
        spec = get_spec(head, 'xgboost')
        feats = arm_features(head, 'base', spec)
        print(f'\n=== base OOF / {head} ({len(feats)} features) ===')
        parts[head] = oof_predictions(
            df, head, 'base', n_splits=splits, feats=feats,
            dropna_on=feats_required(feats), carry=list(H2H_FEATURES))

    # The two heads drop different rows (the O/U head additionally needs an
    # over/under price), so join on nothing — keep them separate and let each
    # market question use the frame that carries its own columns.
    out = {'1x2': parts['1x2'], 'ou': parts['ou']}
    pd.to_pickle(out, OOF_CACHE)
    print(f'\nCached OOF to {OOF_CACHE}')
    return out


def fit_logit(X, y):
    """Unregularised logistic fit + standard errors from observed information.

    sklearn gives no SEs, and the whole point here is to say how close to zero
    a coefficient is. se = sqrt(diag((X' W X)^-1)) with W = diag(p(1-p)).
    """
    lr = LogisticRegression(C=1e8, max_iter=5000).fit(X, y)
    Xd = np.column_stack([np.ones(len(X)), X])
    p = lr.predict_proba(X)[:, 1]
    W = p * (1 - p)
    try:
        cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
        se = np.sqrt(np.diag(cov))[1:]          # drop intercept
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return lr, lr.coef_[0], se


def crossfit_logloss(X, y, splits=5, seed=0):
    """Out-of-sample logloss for a calibrator of this shape.

    A post-hoc calibrator fitted and scored on the same rows is graded on its
    own homework; with a handful of parameters the bias is small but it is
    exactly the size of the effects being chased here.
    """
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        lr = LogisticRegression(C=1e8, max_iter=5000).fit(X[tr], y[tr])
        pred[te] = lr.predict_proba(X[te])[:, 1]
    return log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6)), pred


def analyse(oof, label, outcome, mkt_col, model_col, terms):
    d = oof.dropna(subset=[mkt_col, model_col]).copy()
    if outcome == 'over':
        y = d['y'].values.astype(int)
    elif outcome == 'home':
        y = (d['y'].values == 0).astype(int)
    else:
        y = (d['y'].values == 1).astype(int)

    lm = logit(d[mkt_col].values)
    lp = logit(d[model_col].values)

    res = {'label': label, 'n': int(len(d)), 'base_rate': float(y.mean()),
           'terms': {}}

    # Reference log-losses, cross-fitted so every number below is comparable.
    ll_mkt, _ = crossfit_logloss(lm[:, None], y)
    ll_stack, _ = crossfit_logloss(np.column_stack([lp, lm]), y)
    res['logloss_market'] = float(ll_mkt)
    res['logloss_stack'] = float(ll_stack)

    for t in terms:
        m = d[t].notna().values
        if m.sum() < 1000:
            continue
        x = d[t].values
        # Standardise so the coefficient reads as "per 1 SD of the h2h term"
        # and is comparable across terms on different scales.
        xs = (x - np.nanmean(x[m])) / np.nanstd(x[m])

        yy, lmm, lpp, xx = y[m], lm[m], lp[m], xs[m]

        _, c_mh, se_mh = fit_logit(np.column_stack([lmm, xx]), yy)
        _, c_ph, se_ph = fit_logit(np.column_stack([lpp, xx]), yy)
        _, c_sh, se_sh = fit_logit(np.column_stack([lpp, lmm, xx]), yy)

        ll_mkt_sub, _ = crossfit_logloss(lmm[:, None], yy)
        ll_mkt_h2h, _ = crossfit_logloss(np.column_stack([lmm, xx]), yy)
        ll_stk_sub, _ = crossfit_logloss(np.column_stack([lpp, lmm]), yy)
        ll_stk_h2h, _ = crossfit_logloss(np.column_stack([lpp, lmm, xx]), yy)
        ll_mdl_sub, _ = crossfit_logloss(lpp[:, None], yy)
        ll_mdl_h2h, _ = crossfit_logloss(np.column_stack([lpp, xx]), yy)

        res['terms'][t] = {
            'n': int(m.sum()),
            'coef_over_market': float(c_mh[1]), 'se_over_market': float(se_mh[1]),
            'z_over_market': float(c_mh[1] / se_mh[1]) if se_mh[1] else float('nan'),
            'coef_over_model': float(c_ph[1]), 'se_over_model': float(se_ph[1]),
            'z_over_model': float(c_ph[1] / se_ph[1]) if se_ph[1] else float('nan'),
            'coef_over_stack': float(c_sh[2]), 'se_over_stack': float(se_sh[2]),
            'z_over_stack': float(c_sh[2] / se_sh[2]) if se_sh[2] else float('nan'),
            'nats_over_market': float(ll_mkt_sub - ll_mkt_h2h),
            'nats_over_stack': float(ll_stk_sub - ll_stk_h2h),
            'nats_over_model': float(ll_mdl_sub - ll_mdl_h2h),
        }
    return res


def render(results):
    L = []
    L.append('=' * 78)
    L.append('H2H AS A POST-PREDICTION CALIBRATOR')
    L.append('=' * 78)
    L.append('')
    L.append('Coefficients are per 1 SD of the h2h term, from an unregularised')
    L.append('logistic fit; |z| > 2 is the usual "distinguishable from zero" line.')
    L.append('Nats are 5-fold cross-fitted, so a calibrator is never graded on the')
    L.append('rows it was fitted to. Vig is worth roughly 0.02-0.05 nats for scale.')
    L.append('')
    for r in results:
        L.append('')
        L.append(f"### {r['label']}   n={r['n']}  base rate={100 * r['base_rate']:.1f}%")
        L.append(f"    cross-fitted logloss: market {r['logloss_market']:.5f} | "
                 f"model+market {r['logloss_stack']:.5f}")
        L.append('')
        L.append('    h2h term            on top of PRICE           on top of MODEL')
        L.append('                        coef (se)     z   nats    coef (se)     z   nats')
        for t, v in r['terms'].items():
            L.append(
                f"      {t:18s} {v['coef_over_market']:+.4f} ({v['se_over_market']:.4f}) "
                f"{v['z_over_market']:+5.1f} {v['nats_over_market']:+.5f}   "
                f"{v['coef_over_model']:+.4f} ({v['se_over_model']:.4f}) "
                f"{v['z_over_model']:+5.1f} {v['nats_over_model']:+.5f}")
        L.append('')
        L.append('    added to the full model+market stack (the production shape):')
        for t, v in r['terms'].items():
            L.append(f"      {t:18s} coef {v['coef_over_stack']:+.4f} "
                     f"({v['se_over_stack']:.4f})  z {v['z_over_stack']:+5.1f}  "
                     f"nats {v['nats_over_stack']:+.5f}")
    L.append('')
    L.append('=' * 78)
    L.append('READ: the PRICE column is decisive. A coefficient indistinguishable')
    L.append('from zero there means the market has already absorbed the head-to-head')
    L.append('record, and NO post-prediction calibrator keyed on it can add an edge —')
    L.append('this fit is the most direct extraction available, so nothing more')
    L.append('elaborate will do better. A large coefficient on top of the MODEL with')
    L.append('none on top of the PRICE means a calibrator would only be pushing the')
    L.append('model back toward the price it already had.')
    L.append('=' * 78)
    return '\n'.join(L)


def main():
    oof = build_oof()
    results = []
    for label, outcome, mkt, mdl, terms in MARKETS:
        head = 'ou' if outcome == 'over' else '1x2'
        print(f'\n=== analysing {label} ===')
        results.append(analyse(oof[head], label, outcome, mkt, mdl, terms))

    out_dir = os.path.join(PROJECT_ROOT, 'output', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    report = render(results)
    with open(os.path.join(out_dir, f'h2h_calibrator_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    with open(os.path.join(out_dir, f'h2h_calibrator_{ts}.txt'), 'w') as f:
        f.write(report)
    print('\n' + report)
    print(f'\nWrote output/experiments/h2h_calibrator_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
