"""Chronological-holdout validation for Platt calibrators (Phase C3).

For each league, sort matches by date, fit Platt on the first (1 − holdout_frac),
evaluate on the last holdout_frac. No time-leakage by construction.

Returns per-league before/after Brier on the *test* slice, plus aggregate
acceptance metrics:
  - fraction of leagues where Brier improves on the holdout
  - max regression (worst Brier deterioration as a % of baseline)
"""

from typing import Dict, Tuple

import numpy as np

from .fit import (
    MIN_PLATT_SLOPE,
    _delta,
    apply_platt_binary,
    apply_platt_multiclass,
    fit_platt_binary,
    fit_platt_multiclass,
    metrics,
)


def chronological_holdout_validate(df, oof_probs, target_col, market: str,
                                   holdout_frac: float = 0.2,
                                   min_n_total: int = 120,
                                   source_mode: str = 'full') -> Dict[str, dict]:
    """Per-league chronological holdout. Returns {league: {...}}.

    `df` must have a 'date' column and a 'league' column. `oof_probs` is the
    OOF-prediction array aligned with `df`'s row order. We re-derive
    chronological order per-league before splitting.

    min_n_total of 120 = at least 96 train + 24 test. Smaller leagues skipped.
    """
    if market not in ('oneXtwo', 'ou'):
        raise ValueError(f"Unknown market: {market!r}")

    valid = ~np.isnan(oof_probs).any(axis=1) & df[target_col].notna()
    df_v = df.loc[valid].copy()
    probs_v = oof_probs[valid]

    # Reset to fresh integer index aligned with probs_v.
    df_v = df_v.reset_index(drop=True)

    out: Dict[str, dict] = {}
    for league, sub in df_v.groupby('league'):
        n = len(sub)
        if n < min_n_total:
            continue
        # Sort this league's matches by date; carry positional index for probs.
        sub_sorted = sub.sort_values('date')
        idx = sub_sorted.index.values  # positions in df_v / probs_v
        split = int(n * (1 - holdout_frac))
        if split < 50 or (n - split) < 20:
            continue

        train_idx = idx[:split]
        test_idx  = idx[split:]
        train_probs = probs_v[train_idx]
        test_probs  = probs_v[test_idx]
        train_t = df_v.loc[train_idx, target_col].astype(int).values
        test_t  = df_v.loc[test_idx,  target_col].astype(int).values

        if market == 'oneXtwo':
            params = fit_platt_multiclass(train_probs, train_t, n_classes=3)
            cal_test = apply_platt_multiclass(test_probs, params)
            slopes = [a for a, _ in params]
        else:
            a, b = fit_platt_binary(train_probs[:, 1], train_t)
            cal_test = np.zeros_like(test_probs)
            cal_test[:, 1] = apply_platt_binary(test_probs[:, 1], a, b)
            cal_test[:, 0] = 1.0 - cal_test[:, 1]
            slopes = [a]

        before = metrics(test_probs, test_t)
        after  = metrics(cal_test,    test_t)

        brier_delta = round(after['brier'] - before['brier'], 4)
        # Regression % relative to baseline. Positive = got worse.
        regression_pct = round(
            (brier_delta / before['brier']) * 100.0 if before['brier'] > 0 else 0.0,
            2,
        )

        # Discrimination on the holdout. Brier alone cannot see resolution
        # loss — it improves when a flattened calibrator trades ordering for
        # calibration — so accuracy/AUC and the slope are gated separately.
        acc_delta = round(after['accuracy'] - before['accuracy'], 4)
        auc_delta = _delta(after['auc'], before['auc'])
        worst_slope = round(min(slopes), 4)
        discrimination_ok = (
            acc_delta >= 0
            and (auc_delta is None or auc_delta >= 0)
            and worst_slope >= MIN_PLATT_SLOPE
        )

        out[league] = {
            'n_train':       int(split),
            'n_test':        int(n - split),
            'source_mode':   source_mode,
            'before':        before,
            'after':         after,
            'brier_delta':   brier_delta,
            'regression_pct': regression_pct,
            'acc_delta':      acc_delta,
            'auc_delta':      auc_delta,
            'worst_slope':    worst_slope,
            'discrimination_ok': bool(discrimination_ok),
            # A calibrator counts as an improvement only if it improves Brier
            # *without* giving up discrimination.
            'improved':      bool(brier_delta <= 0 and discrimination_ok),
            'train_date_max': str(sub_sorted['date'].iloc[split - 1].date()) if 'date' in sub_sorted.columns else None,
            'test_date_min':  str(sub_sorted['date'].iloc[split].date())     if 'date' in sub_sorted.columns else None,
            'test_date_max':  str(sub_sorted['date'].iloc[-1].date())        if 'date' in sub_sorted.columns else None,
        }
    return out


def aggregate_acceptance(results: Dict[str, dict],
                         min_improvement_rate: float = 0.6,
                         max_regression_pct: float = 5.0
                         ) -> Tuple[bool, dict]:
    """Apply the C3 acceptance gate.

    Pass iff:
      - at least `min_improvement_rate` of leagues show improvement, AND
      - no league regresses by more than `max_regression_pct` percent.

    Returns (passed, summary_dict).
    """
    if not results:
        return False, {'reason': 'no leagues evaluated', 'n_leagues': 0}

    n = len(results)
    n_improved = sum(1 for r in results.values() if r['improved'])
    worst_regression = max((r['regression_pct'] for r in results.values()), default=0.0)
    worst_league = max(results.items(), key=lambda x: x[1]['regression_pct'])

    n_discrimination_ok = sum(1 for r in results.values()
                              if r.get('discrimination_ok', True))
    degraded = sorted(league for league, r in results.items()
                      if not r.get('discrimination_ok', True))

    improvement_rate = n_improved / n
    rate_ok = improvement_rate >= min_improvement_rate
    regression_ok = worst_regression <= max_regression_pct
    # Any league that loses discrimination fails the batch outright. This is
    # deliberately stricter than the Brier gates: a flattened calibrator scores
    # *well* on Brier/ECE, so a rate-based threshold would let it through.
    discrimination_ok = not degraded

    return rate_ok and regression_ok and discrimination_ok, {
        'n_leagues': n,
        'n_improved': n_improved,
        'improvement_rate': round(improvement_rate, 3),
        'worst_regression_pct': worst_regression,
        'worst_regression_league': worst_league[0],
        'n_discrimination_ok': n_discrimination_ok,
        'discrimination_degraded': degraded,
        'rate_ok': rate_ok,
        'regression_ok': regression_ok,
        'discrimination_ok': discrimination_ok,
        'thresholds': {
            'min_improvement_rate': min_improvement_rate,
            'max_regression_pct': max_regression_pct,
            'min_platt_slope': MIN_PLATT_SLOPE,
        },
    }
