"""Odds-free model experiment.

Question: does the model carry information the market price does not already
have? Production trains on B365H/D/A + IP_H/D/A, so its output is largely an
echo of the price and `EV = p_model * odds - 1` scores the echo's noise. This
compares two arms head-to-head on out-of-fold predictions:

    with_odds   production feature list
    odds_free   production minus B365H/D/A and IP_H/D/A

For each arm the decisive metric is how many nats the model adds *on top of the
devigged market price*, via a stacked logistic regression
`logit(P) ~ logit(p_model) + logit(p_market)`. A flat-stake EV-bucket backtest
on real odds is reported alongside as the practical read.

Writes reports to output/experiments/. Never writes into models/ — production
artifacts are untouched. (prepare_data() does refresh data_sets/elo_ratings.json,
a deterministic recompute from the same history the normal pipeline performs.)

Usage:
    python3 scripts/experiment_odds_free.py
    python3 scripts/experiment_odds_free.py --arms odds_free --heads ou
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))

from train_model import ModelTrainer          # noqa: E402
from model_registry import get_spec           # noqa: E402

ODDS_FEATURES = ['B365H', 'B365D', 'B365A', 'IP_H', 'IP_D', 'IP_A']

# Preference order for the Over/Under 2.5 market price. football-data.co.uk
# changed column names over the years (Bb* in the older seasons), so coalesce.
OU_ODDS_COLS = [
    ('B365>2.5', 'B365<2.5'),
    ('Avg>2.5', 'Avg<2.5'),
    ('BbAv>2.5', 'BbAv<2.5'),
    ('Max>2.5', 'Max<2.5'),
    ('BbMx>2.5', 'BbMx<2.5'),
]

CACHE_PATH = os.path.join(PROJECT_ROOT, 'output', 'experiments', '_prepared.pkl')


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def prepare(cache=True):
    """Engineered training frame, cached because prepare_data() is slow."""
    if cache and os.path.exists(CACHE_PATH):
        print(f"Loading cached frame from {CACHE_PATH}")
        return pd.read_pickle(CACHE_PATH)
    trainer = ModelTrainer(os.path.join(PROJECT_ROOT, 'data_sets', 'MatchHistory'))
    df = trainer.prepare_data()
    if cache:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_pickle(CACHE_PATH)
        print(f"Cached prepared frame to {CACHE_PATH}")
    return df


def add_ou_market(df):
    """Attach devigged market P(Over 2.5) from the best available odds pair."""
    over = pd.Series(np.nan, index=df.index)
    under = pd.Series(np.nan, index=df.index)
    for oc, uc in OU_ODDS_COLS:
        if oc in df.columns and uc in df.columns:
            o, u = pd.to_numeric(df[oc], errors='coerce'), pd.to_numeric(df[uc], errors='coerce')
            fill = over.isna() & o.notna() & u.notna() & (o > 1.0) & (u > 1.0)
            over[fill], under[fill] = o[fill], u[fill]
    df['ou_over_odds'] = over
    df['ou_under_odds'] = under
    raw_o, raw_u = 1.0 / over, 1.0 / under
    df['mkt_over'] = raw_o / (raw_o + raw_u)      # devigged
    return df


def add_1x2_market(df):
    ip = np.column_stack([1.0 / df['B365H'], 1.0 / df['B365D'], 1.0 / df['B365A']])
    df[['mkt_H', 'mkt_D', 'mkt_A']] = ip / ip.sum(axis=1, keepdims=True)   # devigged
    return df


def feature_list(head, arm, spec):
    base = {
        '1x2': ['B365H', 'B365D', 'B365A'],
        'ou': ['B365H', 'B365D', 'B365A', 'H_form_ou', 'A_form_ou'],
    }[head]
    trainer = ModelTrainer(os.path.join(PROJECT_ROOT, 'data_sets', 'MatchHistory'))
    feats = list(dict.fromkeys(base + trainer.common_features))
    if arm == 'odds_free':
        feats = [f for f in feats if f not in ODDS_FEATURES]
    if not spec.uses_categorical and 'league_cat' in feats:
        feats = [f for f in feats if f != 'league_cat']
    return feats


def oof_predictions(df, head, arm, n_splits=5, feats=None,
                    dropna_on=None, carry=()):
    """Out-of-fold model probabilities, aligned with market + outcome.

    `feats` overrides the arm's feature list, so another experiment can reuse
    this loop to A/B a feature set instead of the odds/odds-free arms.

    `dropna_on` overrides which feature columns a row must have to be scored.
    It defaults to all of `feats`, which is right when every feature is dense,
    but a feature that is legitimately NaN for some rows (a first-ever meeting
    has no head-to-head record) would otherwise delete those rows from one arm
    and not the other, and the arms would no longer be scored on the same
    matches. Pass the dense subset to keep them aligned; XGBoost takes the
    NaNs natively.

    `carry` names extra columns to pass through onto the OOF frame, for
    slicing the result afterwards.
    """
    spec = get_spec(head, 'xgboost')
    feats = list(feats) if feats is not None else feature_list(head, arm, spec)

    target = 'target_1x2' if head == '1x2' else 'total_goals'
    market_cols = ['mkt_H', 'mkt_D', 'mkt_A'] if head == '1x2' else ['mkt_over']
    odds_cols = (['B365H', 'B365D', 'B365A'] if head == '1x2'
                 else ['ou_over_odds', 'ou_under_odds'])

    dense = list(feats) if dropna_on is None else list(dropna_on)
    need = dense + [target] + market_cols + odds_cols
    carry = [c for c in carry if c in df.columns]
    d = df.dropna(subset=need).copy().sort_values('date')
    if spec.uses_categorical and 'league_cat' in d.columns:
        d['league_cat'] = d['league_cat'].astype('category')

    print(f"  [{head}/{arm}] {len(d)} rows, {len(feats)} features")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    parts = []
    for fold, (tr, te) in enumerate(tscv.split(d)):
        cv_train, cv_test = d.iloc[tr], d.iloc[te]
        model = spec.build()
        model.fit(cv_train[feats], cv_train[target])

        out = cv_test[market_cols + odds_cols + ['date', 'league'] + carry].copy()
        if head == '1x2':
            p = model.predict_proba(cv_test[feats])
            out[['p_H', 'p_D', 'p_A']] = p
            out['y'] = cv_test[target].astype(int).values
        else:
            lam = model.predict(cv_test[feats])
            p_le2 = np.exp(-lam) * (1 + lam + lam ** 2 / 2)
            out['p_over'] = np.clip(1.0 - p_le2, 0.001, 0.999)
            out['y'] = (cv_test[target] > 2.5).astype(int).values
        out['fold'] = fold
        parts.append(out)
        print(f"    fold {fold + 1}/{n_splits} done ({len(cv_test)} test rows)")
    return pd.concat(parts, ignore_index=True)


def stack_test(p_model, p_market, y):
    """Nats the model adds on top of the market price."""
    X = np.column_stack([logit(p_model), logit(p_market)])
    both = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
    mkt = LogisticRegression(C=1e6, max_iter=2000).fit(X[:, [1]], y)
    mod = LogisticRegression(C=1e6, max_iter=2000).fit(X[:, [0]], y)
    ll_both = log_loss(y, both.predict_proba(X)[:, 1])
    ll_mkt = log_loss(y, mkt.predict_proba(X[:, [1]])[:, 1])
    ll_mod = log_loss(y, mod.predict_proba(X[:, [0]])[:, 1])
    b_model, b_market = both.coef_[0]
    denom = b_model + b_market
    return {
        'n': int(len(y)),
        'coef_model': float(b_model),
        'coef_market': float(b_market),
        'blend_weight_model': float(b_model / denom) if denom else float('nan'),
        'logloss_model_only': float(ll_mod),
        'logloss_market_only': float(ll_mkt),
        'logloss_both': float(ll_both),
        'nats_added_over_market': float(ll_mkt - ll_both),
        'raw_logloss_model': float(log_loss(y, np.clip(p_model, 1e-6, 1 - 1e-6))),
        'raw_logloss_market': float(log_loss(y, np.clip(p_market, 1e-6, 1 - 1e-6))),
    }


EV_BINS = [(-9, 0, 'neg'), (0, .03, '0-0.03'), (.03, .06, '0.03-0.06'),
           (.06, .10, '0.06-0.10'), (.10, .15, '0.10-0.15'), (.15, 9, '>=0.15')]


def ev_backtest(p_model, odds, y):
    """Flat-stake ROI by EV bucket — the practical 'is there a value lane' read."""
    ev = p_model * odds - 1.0
    rows = []
    for lo, hi, label in EV_BINS:
        m = (ev >= lo) & (ev < hi)
        n = int(m.sum())
        if n == 0:
            continue
        pnl = np.where(y[m] == 1, odds[m] - 1.0, -1.0)
        rows.append({
            'bucket': label, 'n': n,
            'model_prob': float(p_model[m].mean() * 100),
            'actual': float(y[m].mean() * 100),
            'flat_roi': float(pnl.mean() * 100),
        })
    return rows


def evaluate(oof, head):
    """Reduce a head's OOF frame to the picked-selection view + full-dist test."""
    if head == '1x2':
        P = oof[['p_H', 'p_D', 'p_A']].values
        M = oof[['mkt_H', 'mkt_D', 'mkt_A']].values
        O = oof[['B365H', 'B365D', 'B365A']].values
        y = oof['y'].values
        pick = P.argmax(axis=1)
        r = np.arange(len(P))
        res = {
            'multiclass_logloss_model': float(log_loss(y, P, labels=[0, 1, 2])),
            'multiclass_logloss_market': float(log_loss(y, M, labels=[0, 1, 2])),
        }
        # Binary view on the model's own pick: did it win, at what price.
        p_model, p_market = P[r, pick], M[r, pick]
        odds, won = O[r, pick], (pick == y).astype(int)
    else:
        p_model = oof['p_over'].values
        p_market = oof['mkt_over'].values
        odds = oof['ou_over_odds'].values
        won = oof['y'].values
        res = {
            'binary_logloss_model': float(log_loss(won, np.clip(p_model, 1e-6, 1 - 1e-6))),
            'binary_logloss_market': float(log_loss(won, np.clip(p_market, 1e-6, 1 - 1e-6))),
        }
    res['stack_on_pick'] = stack_test(p_model, p_market, won)
    if head == 'ou':
        # Production bets whichever side it prefers, so score the picked side:
        # Over when p_over >= .5, else Under (prob, odds and outcome all flip).
        take_over = p_model >= 0.5
        sel_p = np.where(take_over, p_model, 1 - p_model)
        sel_odds = np.where(take_over, odds, oof['ou_under_odds'].values)
        sel_won = np.where(take_over, won, 1 - won)
        res['ev_buckets'] = ev_backtest(sel_p, sel_odds, sel_won)
    else:
        res['ev_buckets'] = ev_backtest(p_model, odds, won)
    return res


def render(results):
    L = []
    L.append('=' * 78)
    L.append('ODDS-FREE MODEL EXPERIMENT')
    L.append('=' * 78)
    L.append('')
    L.append('Decisive metric: nats the model adds on top of the devigged market price.')
    L.append('A value lane needs this to be clearly positive AND large enough to beat vig.')
    L.append('')
    for head in results:
        L.append('')
        L.append(f'### {head.upper()} head')
        for arm in results[head]:
            r = results[head][arm]
            s = r['stack_on_pick']
            L.append('')
            L.append(f'  --- arm: {arm} ---')
            for k in ('multiclass_logloss_model', 'multiclass_logloss_market',
                      'binary_logloss_model', 'binary_logloss_market'):
                if k in r:
                    L.append(f'    {k:32s} {r[k]:.4f}')
            L.append(f'    n (OOF rows on picked side)      {s["n"]}')
            L.append(f'    coef logit(p_model)              {s["coef_model"]:+.3f}')
            L.append(f'    coef logit(p_market)             {s["coef_market"]:+.3f}')
            L.append(f'    implied blend weight on model    {100 * s["blend_weight_model"]:.0f}%')
            L.append(f'    logloss market-only              {s["logloss_market_only"]:.4f}')
            L.append(f'    logloss model+market             {s["logloss_both"]:.4f}')
            L.append(f'    >> NATS ADDED OVER MARKET        {s["nats_added_over_market"]:+.5f}')
            L.append('')
            L.append('    EV bucket        n      model%   actual%   flat-ROI')
            for b in r['ev_buckets']:
                L.append(f'      {b["bucket"]:12s} {b["n"]:6d}   {b["model_prob"]:6.1f}%   '
                         f'{b["actual"]:6.1f}%   {b["flat_roi"]:+7.1f}%')
    L.append('')
    L.append('=' * 78)
    L.append('READ: compare `NATS ADDED OVER MARKET` between with_odds and odds_free.')
    L.append('If odds_free is materially higher, the odds features were suppressing an')
    L.append('independent signal and a real value lane is buildable. If both are ~0,')
    L.append('this feature set has no edge over the price and the value lane should be')
    L.append('retired rather than retuned.')
    L.append('=' * 78)
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['with_odds', 'odds_free'],
                    choices=['with_odds', 'odds_free'])
    ap.add_argument('--heads', nargs='+', default=['1x2', 'ou'], choices=['1x2', 'ou'])
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = prepare(cache=not args.no_cache)
    df = add_1x2_market(df)
    df = add_ou_market(df)

    results = {}
    for head in args.heads:
        results[head] = {}
        for arm in args.arms:
            print(f'\n=== {head} / {arm} ===')
            oof = oof_predictions(df, head, arm, n_splits=args.splits)
            results[head][arm] = evaluate(oof, head)

    out_dir = os.path.join(PROJECT_ROOT, 'output', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    report = render(results)
    with open(os.path.join(out_dir, f'odds_free_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, f'odds_free_{ts}.txt'), 'w') as f:
        f.write(report)
    print('\n' + report)
    print(f'\nWrote output/experiments/odds_free_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
