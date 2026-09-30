"""Euroleague/EuroCup winner-probability Platt calibration (per competition).

Mirrors ``ml_project/nba/nba_calibration.py`` but fits **two** Platt calibrators
— one for Euroleague (E), one for EuroCup (U) — the Phase-0 decision (the
combined model shares features, but the two competitions can miscalibrate
differently). Same OOF-via-TimeSeriesSplit method; same ``(prob, applied,
source)`` 3-tuple apply contract so the UI/betting layer stays sport-agnostic.

Output ``data_sets/Euroleague/euroleague_calibration.json``::

    {"home_win_platt": {"E": {"a":..,"b":..}, "U": {"a":..,"b":..}},
     "per_competition": {"E": {"n_oof":..,"brier_before":..,"brier_after":..}, "U": {...}},
     "total_diagnostic": {...}, "generated_at": "<iso>"}
"""

import datetime
import json
import os
import sys
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(_REPO, "data_sets", "Euroleague", "training_data.csv")
OUT_PATH = os.path.join(_REPO, "data_sets", "Euroleague", "euroleague_calibration.json")

_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_platt(p, y):
    """Return (a, b) for calibrated = σ(a·logit(p)+b); None if too few samples."""
    if len(p) < 50 or len(np.unique(y)) < 2:
        return None
    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# --- Totals (Over/Under) normal-approximation P(Over) ---------------------

def _norm_cdf(x):
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_over(predicted_total, line, sigma):
    """P(total > line) under total ~ Normal(predicted_total, sigma) = Φ((mu−line)/sigma)."""
    try:
        mu, L, s = float(predicted_total), float(line), float(sigma)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return 1.0 if mu > L else (0.0 if mu < L else 0.5)
    return _norm_cdf((mu - L) / s)


def total_sigma(calibration_data, competition=None, default=None):
    """Residual sigma for the totals normal-approx — per competition (E/U),
    falling back to the pooled 'all' sigma, then `default`."""
    tou = (calibration_data or {}).get("total_over_under") or {}
    for key in (competition, "all"):
        node = tou.get(key) if key else None
        if node and node.get("sigma"):
            return float(node["sigma"])
    return default


def _total_overunder_stats(actual, oof_pred):
    """Validate the normal-approx for totals from OOF residuals."""
    actual = np.asarray(actual, float)
    pred = np.asarray(oof_pred, float)
    resid = actual - pred
    if len(resid) < 2:
        return None
    sigma = float(np.std(resid, ddof=1))
    z = resid / sigma if sigma > 0 else resid * 0.0
    return {
        "sigma": round(sigma, 3),
        "n_oof": int(len(resid)),
        "mae": round(float(np.mean(np.abs(resid))), 3),
        "resid_mean": round(float(np.mean(resid)), 3),
        "z_mean": round(float(np.mean(z)), 4),
        "z_std": round(float(np.std(z, ddof=1)), 4),
        "over_rate_at_mu": round(float(np.mean(actual > pred)), 4),
        "within_1sigma": round(float(np.mean(np.abs(z) < 1)), 4),
        "within_2sigma": round(float(np.mean(np.abs(z) < 2)), 4),
    }


def fit_calibration(data_path: str = DATA_PATH, out_path: str = OUT_PATH) -> dict:
    from train_euroleague_models import feature_list, load_params, WINNER_PARAMS, TOTAL_PARAMS
    # Same builders as training, so Platt is fit on OOF output of the SERVED family.
    from euroleague_models import build_total, build_winner

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    print(f"[calibrate] reading {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["is_eurocup"] = (df["competition"] == "U").astype(int)

    features = feature_list()
    df = df.dropna(subset=features).reset_index(drop=True)
    X, y = df[features], df["home_win"].values
    y_total = df["total_points"].values
    comp = df["competition"].values
    print(f"  {len(df):,} games × {len(features)} features  "
          f"({(comp=='E').sum()} E / {(comp=='U').sum()} U)")

    params = load_params(WINNER_PARAMS, {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "eval_metric": "logloss", "random_state": 42})
    params_t = load_params(TOTAL_PARAMS, {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "random_state": 42})

    print("  collecting OOF P(home_win) + total via TimeSeriesSplit ...")
    oof_p = np.full(len(df), np.nan)
    oof_total = np.full(len(df), np.nan)
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5).split(X), 1):
        clf = build_winner(params)
        clf.fit(X.iloc[tr], y[tr])
        oof_p[te] = clf.predict_proba(X.iloc[te])[:, 1]
        reg = build_total(params_t)
        reg.fit(X.iloc[tr], y_total[tr])
        oof_total[te] = reg.predict(X.iloc[te])
        print(f"    fold {fold}: n_test={len(te)}")
    mask = ~np.isnan(oof_p)

    platt, per_comp = {}, {}
    for c in ("E", "U"):
        m = mask & (comp == c)
        if m.sum() == 0:
            continue
        p_c, y_c = oof_p[m], y[m]
        fit = _fit_platt(p_c, y_c)
        before = brier_score_loss(y_c, p_c)
        if fit is None:
            print(f"  [{c}] n={m.sum()} too few / single-class → no calibrator (raw passthrough)")
            per_comp[c] = {"n_oof": int(m.sum()), "brier_before": float(before),
                           "brier_after": float(before), "calibrator": False}
            continue
        a, b = fit
        after = brier_score_loss(y_c, _sigmoid(a * _logit(p_c) + b))
        platt[c] = {"a": a, "b": b}
        per_comp[c] = {"n_oof": int(m.sum()), "brier_before": float(before),
                       "brier_after": float(after), "calibrator": True}
        flag = "" if after < before else "  ⚠ no Brier gain (already well-calibrated)"
        print(f"  [{c}] a={a:+.4f} b={b:+.4f}  Brier {before:.4f}→{after:.4f} "
              f"(Δ{after - before:+.4f}){flag}")

    # Total-points diagnostic (no fit).
    import pickle
    total_diag = {}
    total_path = os.path.join(_REPO, "models", "euroleague", "total_model.pkl")
    if os.path.exists(total_path):
        with open(total_path, "rb") as f:
            tm = pickle.load(f)
        pred, actual = tm.predict(X), df["total_points"].values
        total_diag = {
            "mean_pred": round(float(np.mean(pred)), 3),
            "mean_actual": round(float(np.mean(actual)), 3),
            "mean_abs_err": round(float(np.mean(np.abs(pred - actual))), 3),
            "bias_pred_minus_actual": round(float(np.mean(pred - actual)), 3)}

    # Totals normal-approx: per-competition sigma + a pooled 'all' fallback.
    tmask = ~np.isnan(oof_total)
    tou = {}
    for c in ("E", "U"):
        m = tmask & (comp == c)
        s = _total_overunder_stats(y_total[m], oof_total[m]) if m.sum() else None
        if s:
            tou[c] = s
    pooled = _total_overunder_stats(y_total[tmask], oof_total[tmask])
    if pooled:
        tou["all"] = pooled
        print(f"  totals O/U: all sigma={pooled['sigma']} over@mu={pooled['over_rate_at_mu']} "
              f"±1σ={pooled['within_1sigma']} ±2σ={pooled['within_2sigma']}"
              + "".join(f" | {c} sigma={tou[c]['sigma']}" for c in ('E', 'U') if c in tou))

    out = {
        "home_win_platt": platt,
        "per_competition": per_comp,
        "total_diagnostic": total_diag,
        "total_over_under": tou,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[calibrate] wrote {out_path}")
    return out


@lru_cache(maxsize=4)
def load_calibration_data(path: str = OUT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def apply_home_win_platt(prob: float,
                         calibration_data: dict,
                         competition: Optional[str] = None,
                         enabled: bool = True) -> Tuple[float, bool, Optional[str]]:
    """Calibrate ``P(home_win)`` for the given competition.

    Returns ``(calibrated, applied, source)`` (sport-agnostic 3-tuple). Falls
    back to raw if disabled, no data, or no calibrator for this competition.
    ``source`` is ``'platt-E'`` / ``'platt-U'`` on success.
    """
    if not enabled or not calibration_data or competition is None:
        return float(prob), False, None
    platt = (calibration_data.get("home_win_platt") or {}).get(competition) or {}
    a, b = platt.get("a"), platt.get("b")
    if a is None or b is None:
        return float(prob), False, None
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return prob, False, None
    return float(_sigmoid(a * _logit(p) + b)), True, f"platt-{competition}"


if __name__ == "__main__":
    try:
        fit_calibration()
    except Exception as e:
        print(f"[calibrate] FAIL: {e}", file=sys.stderr)
        sys.exit(1)
