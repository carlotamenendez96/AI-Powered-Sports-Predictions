"""Fit per-league Platt calibrators (Phase C2).

Runs OOF predictions in both full-features and minimal-features modes,
fits Platt scaling per (league, market), merges the two results
(preferring full-mode where available), and writes
`data_sets/league_calibration.json` for inference (C4).

End-to-end runtime: ~6 min (two OOF passes on prepared data).

Usage:
    PYTHONPATH=$(pwd):$(pwd)/ml_project python3 scripts/run_fit_calibration.py
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))

from train_model import ModelTrainer
from ml_project.calibration.diagnose import out_of_fold_predictions
from ml_project.calibration.fit import (
    MIN_PLATT_SLOPE,
    fit_league_calibrators,
    merge_calibrators,
)

CALIBRATION_PATH = os.path.join(PROJECT_ROOT, 'data_sets', 'league_calibration.json')
N_SPLITS = 5
MIN_N_PER_LEAGUE = 100

# Match the diagnostic's minimal-mode feature subset exactly.
SHOT_CORNER_FEATURES = {
    'H_form_sf', 'H_form_sa', 'H_form_cf', 'H_form_ca',
    'A_form_sf', 'A_form_sa', 'A_form_cf', 'A_form_ca',
    'H_home_sf', 'H_home_sa',
    'A_away_sf', 'A_away_sa',
}


def _load_params(name: str, default: dict) -> dict:
    p = os.path.join(PROJECT_ROOT, 'models', f'best_params_{name}.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default


def _run_oof_for_mode(trainer, df_train, mode: str):
    """Run 1X2 + O/U OOF for the given mode ('full' or 'minimal').

    Returns (df_1x2, oof_1x2, df_ou, oof_ou) — the dropna-filtered DataFrames
    aligned with their OOF prediction arrays. Filtering matches the diagnostic.
    """
    common = trainer.common_features
    if mode == 'minimal':
        common = [f for f in common if f not in SHOT_CORNER_FEATURES]

    print(f'\n--- {mode} mode: 1X2 OOF (features={len(common) + 3}) ---')
    features_1x2 = list(dict.fromkeys(['B365H', 'B365D', 'B365A'] + common))
    params_1x2 = _load_params('1x2', {
        'objective': 'multi:softprob', 'num_class': 3,
        'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
        'eval_metric': 'mlogloss', 'early_stopping_rounds': 10,
        'tree_method': 'hist', 'enable_categorical': True,
    })
    df_1x2 = df_train.dropna(subset=features_1x2 + ['target_1x2']).copy()
    df_1x2 = df_1x2.sort_values('date').reset_index(drop=True)
    oof_1x2, _ = out_of_fold_predictions(df_1x2, features_1x2, 'target_1x2',
                                          params_1x2, n_splits=N_SPLITS)

    print(f'\n--- {mode} mode: O/U OOF (features={len(common)}) ---')
    features_ou = list(dict.fromkeys(common))
    params_ou = _load_params('ou', {
        'objective': 'binary:logistic',
        'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
        'eval_metric': 'logloss', 'early_stopping_rounds': 10,
        'tree_method': 'hist', 'enable_categorical': True,
    })
    df_ou = df_train.dropna(subset=features_ou + ['target_ou']).copy()
    df_ou = df_ou.sort_values('date').reset_index(drop=True)
    oof_ou, _ = out_of_fold_predictions(df_ou, features_ou, 'target_ou',
                                         params_ou, n_splits=N_SPLITS)

    return df_1x2, oof_1x2, df_ou, oof_ou


def _fit_for_mode(df_1x2, oof_1x2, df_ou, oof_ou, mode: str, rejections=None):
    """Fit Platt per league for both markets. Returns {league: {market: result}}.

    Leagues whose calibrator flattens or inverts the model's ordering are
    dropped by `fit_league_calibrators` and collected into `rejections`.
    """
    cal_1x2 = fit_league_calibrators(df_1x2, oof_1x2, 'target_1x2', 'oneXtwo',
                                      min_n=MIN_N_PER_LEAGUE, source_mode=mode,
                                      rejections=rejections)
    cal_ou  = fit_league_calibrators(df_ou,  oof_ou,  'target_ou',  'ou',
                                      min_n=MIN_N_PER_LEAGUE, source_mode=mode,
                                      rejections=rejections)
    out = {}
    for league in set(cal_1x2) | set(cal_ou):
        out[league] = {}
        if league in cal_1x2:
            out[league]['oneXtwo'] = cal_1x2[league]
        if league in cal_ou:
            out[league]['ou'] = cal_ou[league]
    return out


def _rejection_table(rejections: list):
    """Print the leagues dropped by the degenerate-calibrator guards.

    A rejected league falls back to raw probabilities at inference, which is
    the intended outcome — but it must be visible, not silent, or the file
    just looks like it covers fewer leagues than expected.
    """
    print('\n=== Rejected by fit guards (fall back to raw probs) ===')
    if not rejections:
        print('None — every fitted calibrator kept the model\'s ordering.')
        return
    print(f'{"league":<35} {"market":<8} {"mode":<8} {"n":>5} '
          f'{"worst_a":>8} {"acc_delta":>10}  reasons')
    print('-' * 110)
    for r in sorted(rejections, key=lambda x: (x['market'], x['league'])):
        worst = min(r['slopes'].values())
        print(f"{r['league'][:35]:<35} {r['market']:<8} {r['source_mode']:<8} "
              f"{r['n']:>5} {worst:>8.4f} {r['acc_delta']:>+10.4f}  "
              f"{'; '.join(r['reasons'])}")
    print(f'\nRejected: {len(rejections)} league/market calibrators.')


def _summary_table(label: str, results: dict):
    print(f'\n=== {label}: improvement summary ===')
    print(f'{"league":<35} {"market":<8} {"n":>5} {"brier_before":>12} {"brier_after":>12} '
          f'{"delta":>8} {"improved":>8}')
    print('-' * 100)
    rows = []
    for league, mkts in sorted(results.items()):
        for market, r in sorted(mkts.items()):
            rows.append((league, market, r['n'], r['before']['brier'], r['after']['brier'],
                         r['brier_delta'], r['improved']))
    # Sort by absolute Brier improvement (most negative first = biggest gain)
    rows.sort(key=lambda r: r[5])
    for r in rows:
        league, market, n, bb, ba, delta, imp = r
        flag = '✅' if imp else '❌'
        print(f"{league[:35]:<35} {market:<8} {n:>5} {bb:>12.4f} {ba:>12.4f} {delta:>+8.4f} {flag:>8}")
    n_improved = sum(1 for r in rows if r[6])
    print(f'\nImproved: {n_improved}/{len(rows)} ({100*n_improved/max(1,len(rows)):.0f}%)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=CALIBRATION_PATH,
                    help=f'Output path (default: {CALIBRATION_PATH}).')
    args = ap.parse_args()

    print('Preparing training data (one-time, ~30s)...')
    trainer = ModelTrainer(data_dir=os.path.join(PROJECT_ROOT, 'data_sets', 'MatchHistory'))
    df_train = trainer.prepare_data()
    print(f'Prepared {len(df_train)} rows across {df_train["league"].nunique()} leagues.')

    rejections = []

    # --- full-features OOF ---
    df_1x2_f, oof_1x2_f, df_ou_f, oof_ou_f = _run_oof_for_mode(trainer, df_train, 'full')
    full_results = _fit_for_mode(df_1x2_f, oof_1x2_f, df_ou_f, oof_ou_f, 'full',
                                 rejections=rejections)
    print(f'\nFull-mode fitted {len(full_results)} leagues.')

    # --- minimal-features OOF ---
    df_1x2_m, oof_1x2_m, df_ou_m, oof_ou_m = _run_oof_for_mode(trainer, df_train, 'minimal')
    minimal_results = _fit_for_mode(df_1x2_m, oof_1x2_m, df_ou_m, oof_ou_m, 'minimal',
                                    rejections=rejections)
    print(f'\nMinimal-mode fitted {len(minimal_results)} leagues.')

    # --- merge, prefer full ---
    merged = merge_calibrators(full_results, minimal_results)
    print(f'\nMerged: {len(merged)} leagues with calibrators.')

    # --- persist ---
    payload = {
        'version': '1',
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'min_n_per_league': MIN_N_PER_LEAGUE,
        'n_splits': N_SPLITS,
        'min_platt_slope': MIN_PLATT_SLOPE,
        'leagues': merged,
        'rejected_at_fit': rejections,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved → {args.out}')

    _rejection_table(rejections)
    _summary_table('Full-mode', full_results)
    _summary_table('Minimal-mode (covers leagues full mode misses)', minimal_results)


if __name__ == '__main__':
    main()
