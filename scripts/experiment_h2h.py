"""Head-to-head feature experiment (FEATURE_ENGINEERING_IDEAS 1.6).

Question: does the record between these two specific teams add anything the
rest of the feature set — and the market price — does not already have?

Every other feature describes the two teams independently. The h2h_* columns
are the only ones that describe the *pairing*: "these two always draw", "this
fixture is always 3+ goals". The claim is that rivalry / stylistic-matchup
effects are repeatable and the global model misses them.

Three arms, identical in every other respect, on the same engineered frame:

    base      production feature list with the h2h_* columns removed
    placebo   the h2h_* columns present but row-permuted as a block (control)
    h2h       production feature list

scored with the same decisive metric as experiment_odds_free: nats added on
top of the devigged market price via a stacked logistic regression. A feature
that only helps the model reproduce the price is worth nothing here; the test
is whether it helps the model *disagree with the price and be right*.

Also reports, per head:
  - XGBoost gain share of the h2h_* block AND of the shuffled block, because
    gain share alone turns out not to distinguish a real feature from a random
    one of the same shape;
  - the same metrics restricted to rows with a thick H2H sample
    (`h2h_n >= H2H_WINDOW`), since a signal diluted by first-ever meetings
    would show up there first.

Writes to output/experiments/. Never touches models/.

Usage:
    python3 scripts/experiment_h2h.py
    python3 scripts/experiment_h2h.py --heads 1x2 --splits 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import (          # noqa: E402
    prepare, add_1x2_market, add_ou_market, feature_list,
    oof_predictions, evaluate,
)
from feature_engineering import H2H_FEATURES, H2H_WINDOW   # noqa: E402
from model_registry import get_spec                        # noqa: E402

ARMS = ('base', 'placebo', 'h2h')

# Why a placebo arm: the tuned booster uses colsample_bytree=0.6, so widening
# the feature list by nine columns changes which columns every tree gets
# to choose from. At the effect sizes this project measures (the L10/L15
# experiment moved nats added over market by 1e-5) that reshuffling is not
# obviously smaller than the signal being tested. The placebo arm carries the
# same nine columns with their values permuted as a block across rows — same
# width, same marginal distributions, same NaN pattern, same sampling
# perturbation, no relationship to the fixture. `h2h` earns a verdict only by
# beating `placebo`, not by beating `base`.
PLACEBO_SEED = 11


def arm_features(head, arm, spec):
    """Feature list for one arm.

    `base` is the production list exactly as `train_model.common_features`
    defines it. The h2h arms ADD the nine columns on top — they were dropped
    from production on 2026-09-18 after measuring flat, so this script has to
    put them back rather than take them away. It is written as an add so it
    keeps working either way: if the block is ever restored to
    `common_features`, `dict.fromkeys` collapses the duplicates and `base`
    still strips them.

    `placebo` gets the same list as `h2h` — same width, same names; only the
    values differ, and those are swapped into the frame, not the list.
    """
    feats = feature_list(head, 'with_odds', spec)
    if arm == 'base':
        return [f for f in feats if f not in H2H_FEATURES]
    return list(dict.fromkeys(feats + list(H2H_FEATURES)))


def placebo_frame(df, seed=PLACEBO_SEED):
    """Copy of `df` with the h2h block row-permuted as a unit.

    Permuted as a block, not column by column, so the columns stay mutually
    coherent (an h2h_n of 0 still comes with NaN averages) — the only thing
    destroyed is the link between a fixture and its own head-to-head record.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(out))
    block = out[list(H2H_FEATURES)].to_numpy()[order]
    for i, c in enumerate(H2H_FEATURES):
        out[c] = block[:, i]
    return out


def gain_share(df, head, feats):
    """What fraction of the fitted booster's total gain the h2h block takes.

    Fit once on the whole frame (this is an attribution readout, not a score —
    the scoring is all out-of-fold), then sum gain over the h2h_* columns.
    """
    spec = get_spec(head, 'xgboost')
    target = 'target_1x2' if head == '1x2' else 'total_goals'
    d = df.dropna(subset=feats_required(feats) + [target]).copy()
    if spec.uses_categorical and 'league_cat' in d.columns:
        d['league_cat'] = d['league_cat'].astype('category')
    model = spec.build()
    model.fit(d[feats], d[target])
    gain = model.get_booster().get_score(importance_type='gain')
    total = sum(gain.values()) or 1.0
    h2h_gain = {k: v for k, v in gain.items() if k in H2H_FEATURES}
    ranked = sorted(gain.items(), key=lambda kv: -kv[1])
    rank_of = {k: i + 1 for i, (k, _) in enumerate(ranked)}
    return {
        'h2h_share_of_total_gain': sum(h2h_gain.values()) / total,
        'h2h_used': len(h2h_gain),
        'h2h_available': len([f for f in feats if f in H2H_FEATURES]),
        'per_feature': {
            k: {'gain_share': v / total, 'rank': rank_of[k]}
            for k, v in sorted(h2h_gain.items(), key=lambda kv: -kv[1])
        },
        'n_features': len(feats),
        'n_rows': int(len(d)),
    }


def feats_required(feats):
    """dropna subset: the h2h averages are legitimately NaN for a first-ever
    meeting, so dropping on them would silently delete 10% of the corpus and
    bias the comparison toward established pairs."""
    return [f for f in feats if f not in H2H_FEATURES]


def render(results, thick):
    L = []
    L.append('=' * 78)
    L.append('HEAD-TO-HEAD FEATURE EXPERIMENT (FEATURE_ENGINEERING_IDEAS 1.6)')
    L.append('=' * 78)
    L.append('')
    L.append(f'H2H_WINDOW = {H2H_WINDOW} prior meetings; {len(H2H_FEATURES)} columns:')
    L.append('  ' + ', '.join(H2H_FEATURES))
    L.append('')
    L.append('Decisive metric: nats added on top of the devigged market price.')
    L.append('base = production features minus the h2h block; h2h = with it.')
    L.append('')
    for head in results:
        L.append('')
        L.append(f'### {head.upper()} head')
        L.append('')
        rows = []
        for arm in ARMS:
            if arm not in results[head]:
                continue
            r = results[head][arm]
            s = r['stack_on_pick']
            rows.append((arm, s))
            L.append(f'  --- arm: {arm} ---')
            for k in ('multiclass_logloss_model', 'multiclass_logloss_market',
                      'binary_logloss_model', 'binary_logloss_market'):
                if k in r:
                    L.append(f'    {k:32s} {r[k]:.4f}')
            L.append(f'    n (OOF rows on picked side)      {s["n"]}')
            L.append(f'    coef logit(p_model)              {s["coef_model"]:+.3f}')
            L.append(f'    coef logit(p_market)             {s["coef_market"]:+.3f}')
            L.append(f'    implied blend weight on model    {100 * s["blend_weight_model"]:+.0f}%')
            L.append(f'    logloss market-only              {s["logloss_market_only"]:.4f}')
            L.append(f'    logloss model+market             {s["logloss_both"]:.4f}')
            L.append(f'    >> NATS ADDED OVER MARKET        {s["nats_added_over_market"]:+.5f}')
            if 'brier' in r:
                L.append(f'    OOF Brier (model / market)       '
                         f'{r["brier"]["model"]:.5f} / {r["brier"]["market"]:.5f}')
            L.append('')
            L.append('    EV bucket        n      model%   actual%   flat-ROI')
            for b in r['ev_buckets']:
                L.append(f'      {b["bucket"]:12s} {b["n"]:6d}   {b["model_prob"]:6.1f}%   '
                         f'{b["actual"]:6.1f}%   {b["flat_roi"]:+7.1f}%')
            L.append('')
        by_arm = dict(rows)
        if 'base' in by_arm:
            base_nats = by_arm['base']['nats_added_over_market']
            for arm in ('placebo', 'h2h'):
                if arm in by_arm:
                    d = by_arm[arm]['nats_added_over_market'] - base_nats
                    L.append(f'    DELTA ({arm:7s} - base) nats      {d:+.5f}')
            if 'placebo' in by_arm and 'h2h' in by_arm:
                d = (by_arm['h2h']['nats_added_over_market']
                     - by_arm['placebo']['nats_added_over_market'])
                L.append(f'    DELTA (h2h - placebo) nats       {d:+.5f}   <- the real test')
            briers = {a: results[head][a]['brier']['model']
                      for a in ARMS if a in results[head] and 'brier' in results[head][a]}
            if 'base' in briers:
                for arm in ('placebo', 'h2h'):
                    if arm in briers:
                        rel = 100 * (briers['base'] - briers[arm]) / briers['base']
                        L.append(f'    Brier improvement vs base ({arm:7s}) {rel:+.3f}%'
                                 f'   (bar: +1.000%)')
            L.append('')
        for arm in ('h2h', 'placebo'):
            key = f'_gain_{arm}'
            if head not in results or key not in results[head]:
                continue
            g = results[head][key]
            label = 'h2h block' if arm == 'h2h' else 'SHUFFLED block (control)'
            L.append(f'    gain share, {label:24s} '
                     f'{100 * g["h2h_share_of_total_gain"]:.1f}%  '
                     f'({g["h2h_used"]}/{g["h2h_available"]} columns used, '
                     f'{g["n_features"]} features total)')
            for k, v in g['per_feature'].items():
                L.append(f'      {k:20s} gain {100 * v["gain_share"]:5.2f}%   rank #{v["rank"]}')
            L.append('')
        if head in thick:
            t = thick[head]
            L.append(f'    thick-H2H subset (h2h_n >= {H2H_WINDOW}): n={t["n"]}')
            for arm in ARMS:
                if arm in t['arms']:
                    L.append(f'      {arm:6s} nats added over market   '
                             f'{t["arms"][arm]["nats_added_over_market"]:+.5f}')
            L.append('')
    L.append('=' * 78)
    L.append('READ: the h2h block earns its place only if h2h - placebo is clearly')
    L.append('positive and large enough to beat vig. h2h - base on its own is not')
    L.append('evidence: widening the feature list perturbs colsample_bytree draws, and')
    L.append('the placebo measures that perturbation. A high gain share with a ~0 delta')
    L.append('means the model is using the columns to reproduce the price, not to beat')
    L.append('it — the outcome the L10/L15 form experiment reached on 2026-09-13.')
    L.append('=' * 78)
    return '\n'.join(L)


def brier(oof, head):
    """OOF Brier — the acceptance bar FEATURE_ENGINEERING_IDEAS sets (>=1%).

    Multiclass (sum of squared errors over the three outcomes) for 1X2, the
    ordinary binary Brier for O/U, plus the market's own score for scale.
    """
    if head == '1x2':
        P = oof[['p_H', 'p_D', 'p_A']].values
        M = oof[['mkt_H', 'mkt_D', 'mkt_A']].values
        Y = np.zeros_like(P)
        Y[np.arange(len(P)), oof['y'].values.astype(int)] = 1.0
        return {'model': float(((P - Y) ** 2).sum(axis=1).mean()),
                'market': float(((M - Y) ** 2).sum(axis=1).mean())}
    y = oof['y'].values
    return {'model': float(((oof['p_over'].values - y) ** 2).mean()),
            'market': float(((oof['mkt_over'].values - y) ** 2).mean())}


def thick_subset(oof_by_arm, head, min_n):
    """Re-score each arm on the rows with a full H2H sample.

    The OOF frames carry `h2h_n` (attached in main), so the arms stay aligned.
    """
    from experiment_odds_free import stack_test
    out = {'arms': {}, 'n': 0}
    for arm, oof in oof_by_arm.items():
        m = oof['h2h_n'] >= min_n
        sub = oof[m]
        if len(sub) < 500:
            continue
        if head == '1x2':
            P = sub[['p_H', 'p_D', 'p_A']].values
            M = sub[['mkt_H', 'mkt_D', 'mkt_A']].values
            y = sub['y'].values
            pick = P.argmax(axis=1)
            r = np.arange(len(P))
            p_model, p_market, won = P[r, pick], M[r, pick], (pick == y).astype(int)
        else:
            p_model = sub['p_over'].values
            p_market = sub['mkt_over'].values
            won = sub['y'].values
        out['arms'][arm] = stack_test(p_model, p_market, won)
        out['n'] = int(len(sub))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='+', default=['1x2', 'ou'], choices=['1x2', 'ou'])
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--skip-gain', action='store_true')
    args = ap.parse_args()

    df = prepare(cache=not args.no_cache)
    missing = [c for c in H2H_FEATURES if c not in df.columns]
    if missing:
        sys.exit(f'Prepared frame has no h2h columns ({missing}). '
                 f'Delete output/experiments/_prepared.pkl and re-run.')
    df = add_1x2_market(df)
    df = add_ou_market(df)

    print(f'H2H coverage: {(df["h2h_n"] > 0).mean():.1%} of rows have a prior meeting, '
          f'{(df["h2h_n"] >= H2H_WINDOW).mean():.1%} have {H2H_WINDOW}+')

    # One placebo frame reused across heads, so the control is identical
    # wherever it appears.
    frames = {arm: df for arm in ARMS}
    if 'placebo' in args.arms:
        frames['placebo'] = placebo_frame(df)

    results, thick = {}, {}
    for head in args.heads:
        results[head] = {}
        oof_by_arm = {}
        for arm in args.arms:
            spec = get_spec(head, 'xgboost')
            feats = arm_features(head, arm, spec)
            print(f'\n=== {head} / {arm} ({len(feats)} features) ===')
            # dropna_on excludes the h2h columns: they are legitimately NaN
            # for a first-ever meeting, and dropping on them would score the
            # two arms on different sets of matches. carry brings h2h_n onto
            # the OOF frame for the thick-sample slice below.
            frame = frames[arm]
            oof = oof_predictions(frame, head, arm, n_splits=args.splits, feats=feats,
                                  dropna_on=feats_required(feats), carry=['h2h_n'])
            oof_by_arm[arm] = oof
            results[head][arm] = evaluate(oof, head)
            results[head][arm]['brier'] = brier(oof, head)
        if {'base', 'h2h'} <= set(oof_by_arm):
            thick[head] = thick_subset(
                {a: oof_by_arm[a] for a in ('base', 'h2h')}, head, H2H_WINDOW)
        if not args.skip_gain and 'h2h' in args.arms:
            spec = get_spec(head, 'xgboost')
            feats = arm_features(head, 'h2h', spec)
            for arm in ('h2h', 'placebo'):
                if arm == 'placebo' and 'placebo' not in args.arms:
                    continue
                print(f'\n=== {head} / gain attribution ({arm}) ===')
                # Same call on the permuted frame: if shuffled columns take a
                # comparable share of gain, the share is a measure of how
                # splittable the columns are, not of how much they know.
                results[head][f'_gain_{arm}'] = gain_share(frames[arm], head, feats)

    out_dir = os.path.join(PROJECT_ROOT, 'output', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    report = render(results, thick)
    with open(os.path.join(out_dir, f'h2h_{ts}.json'), 'w') as f:
        json.dump({'results': results, 'thick': thick}, f, indent=2, default=float)
    with open(os.path.join(out_dir, f'h2h_{ts}.txt'), 'w') as f:
        f.write(report)
    print('\n' + report)
    print(f'\nWrote output/experiments/h2h_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
