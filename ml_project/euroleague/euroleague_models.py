"""Model builders for the Euroleague/EuroCup winner + total heads.

The single place that decides WHAT estimator each head is. Training
(``train_euroleague_models.py``) and calibration (``euroleague_calibration.py``)
both build through here, so the calibrator is always fit on out-of-fold output
of the same model family that is served.

Why linear (2026-09-30)
-----------------------
A regularisation comparison on 3,630 out-of-fold games (same 5 TimeSeriesSplit
folds as training) showed the previous untuned XGBoost heads (depth 5, 200
trees, ~4k rows) were overfitting:

* **winner** — XGBoost calibrated Brier 0.2258 vs **0.2184 for a one-feature
  ELO logistic** (+0.0074, 95% CI [+0.0044, +0.0103]). Regularised trees only
  drew level with ELO. Best arm: logistic on ELO + five home-minus-away
  differences, 0.2165 (−0.0019 vs ELO, CI [−0.0038, +0.0001]) — and nearly
  calibrated raw (0.2172), where XGBoost needed Platt to pull 0.2392 → 0.2258.
* **total** — Ridge MAE 13.52 vs XGBoost 14.03 (−0.51, CI [−0.66, −0.35]).

Those were model-vs-result numbers; whether either beats the bookmaker price is
a separate question that needs this season's settled odds.

The models are scikit-learn Pipelines, so the pickle carries its own feature
derivation: the predictor keeps passing the full ``features_winner.json``
columns and needs no change. Each pipeline starts with a median imputer because
logistic/Ridge reject NaN, where XGBoost handled it natively — without it a
thin-history fixture would crash prediction instead of degrading.

``EUROLEAGUE_MODEL_FAMILY=xgb`` restores the previous XGBoost heads (rollback).
Pickles reference this module by its bare name ``euroleague_models``, so it must
stay importable from ``ml_project/euroleague`` (the scripts' own directory).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMILY = os.environ.get("EUROLEAGUE_MODEL_FAMILY", "linear").strip().lower()

# Rest days beyond a week carry no extra signal (season start, breaks) and
# would otherwise dominate the scaled rest difference.
REST_CAP_DAYS = 7


class WinnerDiffs(BaseEstimator, TransformerMixin):
    """Full feature frame → the six columns the winner logistic uses."""

    columns = ["elo_diff", "pm10_diff", "venue_win_diff", "rest_diff", "b2b_diff", "is_eurocup"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return pd.DataFrame({
            "elo_diff": X["home_elo_pre"] - X["away_elo_pre"],
            "pm10_diff": X["home_l10_plus_minus"] - X["away_l10_plus_minus"],
            "venue_win_diff": X["home_venue_l5_win"] - X["away_venue_l5_win"],
            "rest_diff": (X["home_rest_days"].clip(upper=REST_CAP_DAYS)
                          - X["away_rest_days"].clip(upper=REST_CAP_DAYS)),
            "b2b_diff": X["home_b2b"] - X["away_b2b"],
            "is_eurocup": X["is_eurocup"],
        }, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)


def build_winner(xgb_params: dict | None = None):
    if FAMILY == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**(xgb_params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "eval_metric": "logloss", "random_state": 42}))
    return make_pipeline(WinnerDiffs(), SimpleImputer(strategy="median"),
                         StandardScaler(), LogisticRegression(max_iter=2000))


def build_total(xgb_params: dict | None = None):
    if FAMILY == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**(xgb_params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def describe(model, feature_names) -> list[tuple[str, float]]:
    """(name, weight) pairs, largest first — coefficients for the linear
    pipelines, importances for XGBoost. For the training log only."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_"):
        names = (list(model[0].get_feature_names_out())
                 if isinstance(model[0], WinnerDiffs) else list(feature_names))
        return sorted(zip(names, np.ravel(est.coef_)), key=lambda t: -abs(t[1]))
    return sorted(zip(feature_names, est.feature_importances_), key=lambda t: -t[1])
