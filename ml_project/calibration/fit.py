"""Platt-scaling fit + persistence for per-league probability recalibration (C2).

For each (league, market), fits a small logistic regression on the OOF
predictions produced by C1's diagnostic engine. The output is a tiny set
of `(a, b)` parameters per league per market — together they form
`data_sets/league_calibration.json`, the lookup table that
`predict_matches.py` will use at inference (C4).

Markets:
  1X2  — per-class one-vs-rest Platt (renormalised after scaling).
  O/U  — single binary Platt on P(over).

Calibration is on OOF predictions, which are unbiased measurements of
the model's probability. Fitting on them and then evaluating in-sample
(below) is mildly optimistic — that's what C3's chronological holdout
will properly correct. C2's job is to produce the calibration table and
flag any leagues where the in-sample fit doesn't even improve Brier.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .diagnose import brier_multiclass, expected_calibration_error, log_loss_safe


_EPS = 1e-6

# Minimum acceptable Platt slope. A well-behaved calibrator rescales the
# model's logit, so `a` sits somewhere near 1.0. A slope at or below this
# means the calibrator has flattened the model's ordering into the league
# base rate — and a negative slope actively inverts it. Either way the
# argmax pick stops tracking the model, which is how the 2026-08-25 fit
# shipped 27/37 leagues with at least one negative slope and turned away
# favourites into home picks. Brier/ECE/log-loss are all blind to this
# (Brier trades resolution for calibration and still improves), so the
# slope bound is the gate that has to catch it.
MIN_PLATT_SLOPE = 0.3


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt_binary(p_pos: np.ndarray,
                     targets: np.ndarray) -> Tuple[float, float]:
    """Fit P(pos) Platt scaling: calibrated = sigmoid(a * logit(p) + b).

    Returns (a, b). Uses sklearn LogisticRegression — single-feature problem
    with mild default L2 regularisation.
    """
    X = _safe_logit(p_pos).reshape(-1, 1)
    y = targets.astype(int)
    if len(np.unique(y)) < 2:
        # Degenerate: all-same target. Fall back to identity (a=1, b=0).
        return 1.0, 0.0
    clf = LogisticRegression(solver='lbfgs', max_iter=200)
    clf.fit(X, y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def apply_platt_binary(p_pos: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * _safe_logit(p_pos) + b)


def fit_platt_multiclass(probs: np.ndarray,
                         targets: np.ndarray,
                         n_classes: int = 3) -> List[Tuple[float, float]]:
    """One-vs-rest Platt per class. Returns list of (a, b) per class."""
    out = []
    for k in range(n_classes):
        a, b = fit_platt_binary(probs[:, k], (targets == k).astype(int))
        out.append((a, b))
    return out


def apply_platt_multiclass(probs: np.ndarray,
                           params: List[Tuple[float, float]]) -> np.ndarray:
    """Apply per-class Platt, then renormalise so each row sums to 1."""
    cal = np.zeros_like(probs)
    for k, (a, b) in enumerate(params):
        cal[:, k] = apply_platt_binary(probs[:, k], a, b)
    row_sums = cal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return cal / row_sums


def _nan_to_none(x: float, ndigits: int = 4) -> Optional[float]:
    """Round, mapping NaN to None so the result is valid JSON."""
    return None if x is None or np.isnan(x) else round(float(x), ndigits)


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    """after − before, or None when either side is undefined."""
    if after is None or before is None:
        return None
    return round(after - before, 4)


def discrimination(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Argmax accuracy + macro one-vs-rest AUC.

    These are the *resolution* half of the Brier decomposition — the part a
    base-rate-collapsing calibrator destroys while Brier still improves.
    AUC is `None` when a class is absent from `targets` (undefined there);
    callers must treat that as "no signal", not as a pass.
    """
    n_classes = probs.shape[1]
    acc = float((np.argmax(probs, axis=1) == targets).mean())
    try:
        with warnings.catch_warnings():
            # A league slice can be missing a class; we return None for AUC in
            # that case, so sklearn's warning about it is noise.
            warnings.simplefilter('ignore', UndefinedMetricWarning)
            if n_classes == 2:
                auc = float(roc_auc_score(targets, probs[:, 1]))
            else:
                auc = float(roc_auc_score(targets, probs, multi_class='ovr',
                                          average='macro',
                                          labels=list(range(n_classes))))
    except ValueError:
        auc = float('nan')
    return {'accuracy': round(acc, 4), 'auc': _nan_to_none(auc)}


def metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Brier + log loss + ECE + discrimination (accuracy / AUC).

    Brier, log-loss and ECE are the same definitions C1 uses. Accuracy and
    AUC are additive — they exist so the acceptance gates can see resolution
    loss, which the other three cannot distinguish from a calibration win.
    """
    ece, _ = expected_calibration_error(probs, targets)
    out = {
        'brier':    round(brier_multiclass(probs, targets), 4),
        'log_loss': round(log_loss_safe(probs, targets), 4),
        'ece':      round(ece, 4),
    }
    out.update(discrimination(probs, targets))
    return out


def fit_league_calibrators(df, oof_probs, target_col, market: str,
                           min_n: int = 100,
                           source_mode: str = 'full',
                           min_slope: float = MIN_PLATT_SLOPE,
                           rejections: Optional[List[dict]] = None
                           ) -> Dict[str, dict]:
    """Per league, fit Platt + record before/after metrics.

    `market` is 'oneXtwo' or 'ou' (used as key in the output dict).
    Returns: {league_name: {market: {platt, n, source_mode, before, after, improved}}}

    A league/market is **rejected** (omitted from the result entirely, so the
    league falls back to raw probabilities at inference) when either guard
    trips:
      - any fitted slope < `min_slope` — the calibrator has flattened or
        inverted the model's ordering;
      - argmax accuracy drops after calibration — resolution was traded away.
    Pass a list as `rejections` to collect the reasons for reporting.
    """
    if market not in ('oneXtwo', 'ou'):
        raise ValueError(f"Unknown market: {market!r}")

    out: Dict[str, dict] = {}
    valid = ~np.isnan(oof_probs).any(axis=1) & df[target_col].notna()
    df_v = df.loc[valid].reset_index(drop=True)
    probs_v = oof_probs[valid]

    for league, sub in df_v.groupby('league'):
        n = len(sub)
        if n < min_n:
            continue
        idx = sub.index.values
        p = probs_v[idx]
        t = sub[target_col].astype(int).values

        before = metrics(p, t)

        if market == 'oneXtwo':
            params = fit_platt_multiclass(p, t, n_classes=3)
            cal = apply_platt_multiclass(p, params)
            platt = {
                'home': {'a': round(params[0][0], 4), 'b': round(params[0][1], 4)},
                'draw': {'a': round(params[1][0], 4), 'b': round(params[1][1], 4)},
                'away': {'a': round(params[2][0], 4), 'b': round(params[2][1], 4)},
            }
        else:  # ou
            # OOF probs from O/U binary model are 2-column [P(under), P(over)].
            # We fit on P(over) since target_ou == 1 means over.
            a, b = fit_platt_binary(p[:, 1], t)
            cal = np.zeros_like(p)
            cal[:, 1] = apply_platt_binary(p[:, 1], a, b)
            cal[:, 0] = 1.0 - cal[:, 1]
            platt = {'over': {'a': round(a, 4), 'b': round(b, 4)}}

        after = metrics(cal, t)
        improved = after['brier'] <= before['brier']

        slopes = {c: platt[c]['a'] for c in platt}
        worst_slope = min(slopes.values())
        acc_delta = round(after['accuracy'] - before['accuracy'], 4)
        reasons = []
        if worst_slope < min_slope:
            degenerate = [c for c, a in slopes.items() if a < min_slope]
            reasons.append(
                f"slope {worst_slope:.4f} < {min_slope} on {'/'.join(sorted(degenerate))}")
        if acc_delta < 0:
            reasons.append(
                f"accuracy {before['accuracy']:.4f} -> {after['accuracy']:.4f} "
                f"({acc_delta:+.4f})")
        if reasons:
            if rejections is not None:
                rejections.append({
                    'league': league,
                    'market': market,
                    'source_mode': source_mode,
                    'n': int(n),
                    'slopes': slopes,
                    'acc_before': before['accuracy'],
                    'acc_after': after['accuracy'],
                    'acc_delta': acc_delta,
                    'brier_delta': round(after['brier'] - before['brier'], 4),
                    'reasons': reasons,
                })
            continue

        out[league] = {
            'n': int(n),
            'source_mode': source_mode,
            'platt': platt,
            'before': before,
            'after': after,
            'improved': bool(improved),
            'brier_delta': round(after['brier'] - before['brier'], 4),
            'ece_delta':   round(after['ece']   - before['ece'],   4),
            'acc_delta':   acc_delta,
            # None rather than NaN — json.dump would emit a bare `NaN`, which
            # is not valid JSON and trips strict parsers downstream.
            'auc_delta':   _delta(after['auc'], before['auc']),
            'worst_slope': round(worst_slope, 4),
        }
    return out


def merge_calibrators(full_results: Dict[str, Dict[str, dict]],
                      minimal_results: Dict[str, Dict[str, dict]]
                      ) -> Dict[str, Dict[str, dict]]:
    """Prefer full-mode entries; backfill leagues only present in minimal mode.

    Inputs are {league: {market: result}}; output has the same shape.
    """
    out: Dict[str, Dict[str, dict]] = {}
    all_leagues = set(full_results) | set(minimal_results)
    for league in sorted(all_leagues):
        for market in ('oneXtwo', 'ou'):
            chosen = None
            if league in full_results and market in full_results[league]:
                chosen = full_results[league][market]
            elif league in minimal_results and market in minimal_results[league]:
                chosen = minimal_results[league][market]
            if chosen is not None:
                out.setdefault(league, {})[market] = chosen
    return out
