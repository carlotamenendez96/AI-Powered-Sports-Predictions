"""Validate per-league Platt calibrators on a chronological holdout (Phase C3).

For each league:
  - Sort its matches by date.
  - Fit Platt on the first 80% (training slice).
  - Apply Platt to the last 20% (held-out, never seen during fit).
  - Compare Brier / log loss / ECE on the holdout, with and without calibration.

Pass criteria (from FOOTBALL_NEXT_STEPS.md):
  - ≥60% of leagues show Brier improvement on the holdout.
  - No league regresses by > 5%.

Outputs JSON + Markdown to output/calibration/validate_<ts>.{json,md}.

Usage:
    PYTHONPATH=$(pwd):$(pwd)/ml_project python3 scripts/run_validate_calibration.py
"""

import argparse
import datetime
import glob
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
from ml_project.calibration.validate import (
    aggregate_acceptance,
    chronological_holdout_validate,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'calibration')
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, 'data_sets', 'league_calibration.json')
N_SPLITS = 5
HOLDOUT_FRAC = 0.20
MIN_N_TOTAL = 120
MIN_IMPROVEMENT_RATE = 0.60
MAX_REGRESSION_PCT = 5.0


def filter_calibration_file(cal_path: str,
                            full_results: dict,
                            minimal_results: dict,
                            max_regression_pct: float) -> list:
    """Remove entries from `league_calibration.json` that fail the holdout test.

    Returns the list of removed entries (for logging).
    """
    if not os.path.exists(cal_path):
        print(f'Calibration file not found at {cal_path}; skipping filter.')
        return []

    # Back up the original.
    backup = cal_path + '.prefilter.bak'
    with open(cal_path) as f:
        cal = json.load(f)
    with open(backup, 'w') as f:
        json.dump(cal, f, indent=2)

    removed = []
    leagues = cal.get('leagues', {})
    for league, markets in list(leagues.items()):
        for market in list(markets.keys()):
            entry = markets[market]
            source = entry.get('source_mode', 'full')
            mode_results = full_results if source == 'full' else minimal_results
            ho = mode_results.get(market, {}).get(league)
            if ho is None:
                # No holdout data (league too small for C3). Conservative:
                # keep the entry — it passed C2 in-sample. Note it.
                continue
            reasons = []
            if ho['regression_pct'] > max_regression_pct:
                reasons.append(f"brier regression {ho['regression_pct']:.2f}% "
                               f"> {max_regression_pct}%")
            if not ho.get('discrimination_ok', True):
                reasons.append(
                    f"discrimination lost (acc {ho['acc_delta']:+.4f}, "
                    f"auc {ho['auc_delta']}, worst slope {ho['worst_slope']})")
            if reasons:
                removed.append({
                    'league': league,
                    'market': market,
                    'source_mode': source,
                    'regression_pct': ho['regression_pct'],
                    'brier_before': ho['before']['brier'],
                    'brier_after':  ho['after']['brier'],
                    'acc_delta': ho.get('acc_delta'),
                    'auc_delta': ho.get('auc_delta'),
                    'worst_slope': ho.get('worst_slope'),
                    'n_test': ho['n_test'],
                    'reasons': reasons,
                })
                del markets[market]
        if not markets:
            del leagues[league]

    cal['holdout_filter_applied'] = True
    cal['holdout_filter_max_regression_pct'] = max_regression_pct
    cal['holdout_filter_removed'] = removed
    cal['holdout_filtered_at'] = datetime.datetime.now().isoformat(timespec='seconds')

    with open(cal_path, 'w') as f:
        json.dump(cal, f, indent=2)
    return removed

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
    common = trainer.common_features
    if mode == 'minimal':
        common = [f for f in common if f not in SHOT_CORNER_FEATURES]
    print(f'\n--- {mode} mode: 1X2 OOF ---')
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

    print(f'\n--- {mode} mode: O/U OOF ---')
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


def _validate_mode(df_1x2, oof_1x2, df_ou, oof_ou, mode: str):
    r1 = chronological_holdout_validate(df_1x2, oof_1x2, 'target_1x2', 'oneXtwo',
                                         holdout_frac=HOLDOUT_FRAC,
                                         min_n_total=MIN_N_TOTAL,
                                         source_mode=mode)
    r2 = chronological_holdout_validate(df_ou,  oof_ou,  'target_ou',  'ou',
                                         holdout_frac=HOLDOUT_FRAC,
                                         min_n_total=MIN_N_TOTAL,
                                         source_mode=mode)
    return {'oneXtwo': r1, 'ou': r2}


def _print_section(title: str, results: dict):
    print(f'\n=== {title} ===')
    print(f'{"league":<35} {"market":<8} {"n_test":>6} {"brier_before":>12} '
          f'{"brier_after":>12} {"delta":>8} {"reg%":>7}')
    print('-' * 100)
    rows = []
    for market, mres in results.items():
        for league, r in mres.items():
            rows.append((league, market, r['n_test'], r['before']['brier'],
                         r['after']['brier'], r['brier_delta'],
                         r['regression_pct'], r['improved']))
    rows.sort(key=lambda x: x[5])  # by delta, most-improved first
    for r in rows:
        league, market, n_test, bb, ba, delta, reg, imp = r
        flag = '✅' if imp else '❌'
        print(f"{league[:35]:<35} {market:<8} {n_test:>6} {bb:>12.4f} "
              f"{ba:>12.4f} {delta:>+8.4f} {reg:>+6.2f} {flag}")
    n_total = len(rows)
    n_imp = sum(1 for r in rows if r[7])
    if n_total:
        print(f'\nImproved: {n_imp}/{n_total} ({100 * n_imp / n_total:.1f}%)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=OUTPUT_DIR)
    ap.add_argument('--no-filter', action='store_true',
                    help='Skip rewriting league_calibration.json to drop holdout-failing entries.')
    ap.add_argument('--calibration', default=CALIBRATION_PATH,
                    help='Path to the calibration JSON file to filter in-place.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Preparing training data...')
    trainer = ModelTrainer(data_dir=os.path.join(PROJECT_ROOT, 'data_sets', 'MatchHistory'))
    df_train = trainer.prepare_data()
    print(f'Prepared {len(df_train)} rows across {df_train["league"].nunique()} leagues.')

    df_1x2_f, oof_1x2_f, df_ou_f, oof_ou_f = _run_oof_for_mode(trainer, df_train, 'full')
    full_results = _validate_mode(df_1x2_f, oof_1x2_f, df_ou_f, oof_ou_f, 'full')

    df_1x2_m, oof_1x2_m, df_ou_m, oof_ou_m = _run_oof_for_mode(trainer, df_train, 'minimal')
    minimal_results = _validate_mode(df_1x2_m, oof_1x2_m, df_ou_m, oof_ou_m, 'minimal')

    # Acceptance per mode + combined
    print('\n')
    full_pass, full_summary = aggregate_acceptance(
        {f'{lg}|{mkt}': r for mkt, mres in full_results.items() for lg, r in mres.items()},
        MIN_IMPROVEMENT_RATE, MAX_REGRESSION_PCT)
    minimal_pass, minimal_summary = aggregate_acceptance(
        {f'{lg}|{mkt}': r for mkt, mres in minimal_results.items() for lg, r in mres.items()},
        MIN_IMPROVEMENT_RATE, MAX_REGRESSION_PCT)

    _print_section('Full-features mode', full_results)
    _print_section('Minimal-features mode', minimal_results)

    print(f'\n=== Acceptance gate ===')
    print(f'  full mode:    {"PASS" if full_pass else "FAIL"}    {full_summary}')
    print(f'  minimal mode: {"PASS" if minimal_pass else "FAIL"} {minimal_summary}')

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    json_path = os.path.join(args.out_dir, f'validate_{ts}.json')
    md_path = os.path.join(args.out_dir, f'validate_{ts}.md')

    payload = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'holdout_frac': HOLDOUT_FRAC,
        'min_n_total': MIN_N_TOTAL,
        'thresholds': {
            'min_improvement_rate': MIN_IMPROVEMENT_RATE,
            'max_regression_pct':   MAX_REGRESSION_PCT,
        },
        'full':    {'results': full_results,    'summary': full_summary,    'passed': full_pass},
        'minimal': {'results': minimal_results, 'summary': minimal_summary, 'passed': minimal_pass},
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)

    # Markdown
    lines = [
        '# Calibration holdout validation (C3)',
        '',
        f'Generated: {payload["generated_at"]}',
        f'Holdout: last {int(HOLDOUT_FRAC * 100)}% per league chronologically.',
        f'Thresholds: ≥{int(MIN_IMPROVEMENT_RATE * 100)}% improvement rate, '
        f'≤{MAX_REGRESSION_PCT:.0f}% max regression.',
        '',
        '## Acceptance gate',
        '',
        f'- **Full-features mode**: {"✅ PASS" if full_pass else "❌ FAIL"} — '
        f'{full_summary["n_improved"]}/{full_summary["n_leagues"]} improved '
        f'({100 * full_summary["improvement_rate"]:.0f}%), worst regression '
        f'{full_summary["worst_regression_pct"]:+.1f}% in '
        f'`{full_summary["worst_regression_league"]}`.',
        f'- **Minimal-features mode**: {"✅ PASS" if minimal_pass else "❌ FAIL"} — '
        f'{minimal_summary["n_improved"]}/{minimal_summary["n_leagues"]} improved '
        f'({100 * minimal_summary["improvement_rate"]:.0f}%), worst regression '
        f'{minimal_summary["worst_regression_pct"]:+.1f}% in '
        f'`{minimal_summary["worst_regression_league"]}`.',
        '',
    ]
    for label, results in (('Full-features', full_results), ('Minimal-features', minimal_results)):
        lines.append(f'## {label} mode — per-league holdout results')
        lines.append('')
        lines.append('| league | market | n_test | brier_before | brier_after | delta | reg% | improved |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
        rows = []
        for market, mres in results.items():
            for league, r in mres.items():
                rows.append((league, market, r['n_test'], r['before']['brier'],
                             r['after']['brier'], r['brier_delta'],
                             r['regression_pct'], r['improved']))
        rows.sort(key=lambda x: x[5])
        for r in rows:
            lines.append('| ' + ' | '.join([
                r[0], r[1], str(r[2]),
                f'{r[3]:.4f}', f'{r[4]:.4f}', f'{r[5]:+.4f}',
                f'{r[6]:+.2f}', '✅' if r[7] else '❌',
            ]) + ' |')
        lines.append('')

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'\nSaved:\n  {json_path}\n  {md_path}')

    # Prune older audit outputs — keep only the most recent pair. Each
    # retrain produces a fresh pair via this script; older ones are
    # superseded the moment a new retrain finishes.
    for _patt in ('validate_*.json', 'validate_*.md'):
        _old = sorted(glob.glob(os.path.join(args.out_dir, _patt)))[:-1]
        for _f in _old:
            try: os.remove(_f)
            except OSError: pass
    print(f'Pruned older validate_* outputs (kept latest pair).')

    # --- filter step: rewrite league_calibration.json without failing entries ---
    if args.no_filter:
        print('\n--no-filter set; leaving league_calibration.json unchanged.')
    else:
        print(f'\n=== Filtering {args.calibration} ===')
        removed = filter_calibration_file(args.calibration, full_results,
                                          minimal_results, MAX_REGRESSION_PCT)
        if removed:
            print(f'Removed {len(removed)} entry(ies) that failed the holdout gates '
                  f'(brier regression >{MAX_REGRESSION_PCT}%, or discrimination lost):')
            for r in removed:
                print(f"  - {r['league']:<25} {r['market']:<8} source={r['source_mode']:<8} "
                      f"brier {r['brier_before']:.4f} → {r['brier_after']:.4f} "
                      f"({r['regression_pct']:+.2f}%), n_test={r['n_test']}")
                for reason in r.get('reasons', []):
                    print(f"      · {reason}")
            print('Backup saved as <path>.prefilter.bak; production C4 will fall back '
                  'to raw probs for these (league, market) pairs.')
        else:
            print('No entries failed the holdout gates — nothing filtered.')


if __name__ == '__main__':
    main()
