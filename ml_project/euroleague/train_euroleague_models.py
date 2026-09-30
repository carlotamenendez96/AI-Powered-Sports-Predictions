"""Train the Euroleague/EuroCup winner + total models (combined model).

Mirrors ``ml_project/nba/train_nba_models.py``. One combined model across both
competitions with a single game-level categorical feature ``is_eurocup`` (E=0,
U=1) — the Phase-0 decision (cf. football's one-model-across-leagues with
``league_cat``). Reports a per-competition CV breakdown so we can see whether
the combined model under-performs on either competition (the trigger to split).

* **Winner** — target ``home_win``.
* **Total**  — target ``total_points``.

The estimator for each head comes from ``euroleague_models`` (logistic / Ridge
since 2026-09-30 — see that module for the measurement; ``EUROLEAGUE_MODEL_FAMILY
=xgb`` restores XGBoost).

5-fold TimeSeriesSplit CV, then a final fit on (95% train / 5% tail). Models
pickled to ``models/euroleague/{winner,total}_model.pkl``; feature manifests to
``models/euroleague/features_{winner,total}.json`` so the predictor reads
exactly the trained columns.
"""

import json
import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from euroleague_models import FAMILY, build_total, build_winner, describe

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(_REPO, "data_sets", "Euroleague", "training_data.csv")
MODEL_DIR = os.path.join(_REPO, "models", "euroleague")
WINNER_MODEL = os.path.join(MODEL_DIR, "winner_model.pkl")
TOTAL_MODEL  = os.path.join(MODEL_DIR, "total_model.pkl")
WINNER_PARAMS = os.path.join(MODEL_DIR, "best_params_winner.json")
TOTAL_PARAMS  = os.path.join(MODEL_DIR, "best_params_total.json")
WINNER_FEATURES = os.path.join(MODEL_DIR, "features_winner.json")
TOTAL_FEATURES  = os.path.join(MODEL_DIR, "features_total.json")

SCHEDULE_FEATS = ("rest_days", "b2b", "elo_pre")
L5_METRICS  = ("pts", "pts_allowed", "win", "fg_pct", "fg3_pct", "ft_pct",
               "reb", "ast", "tov", "plus_minus")
L10_METRICS = ("pts", "pts_allowed", "win", "plus_minus")
VENUE_METRICS = ("pts", "pts_allowed", "win")
# Game-level categorical: the combined-model competition flag.
GAME_FEATS = ("is_eurocup",)


def feature_list() -> List[str]:
    feats: list[str] = []
    for side in ("home", "away"):
        for m in SCHEDULE_FEATS:
            feats.append(f"{side}_{m}")
        for m in L5_METRICS:
            feats.append(f"{side}_l5_{m}")
        for m in L10_METRICS:
            feats.append(f"{side}_l10_{m}")
        for m in VENUE_METRICS:
            feats.append(f"{side}_venue_l5_{m}")
    feats.extend(GAME_FEATS)
    return feats


def load_params(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            print(f"  loaded tuned params from {os.path.basename(path)}")
            return json.load(f)
    print(f"  no tuned params at {os.path.basename(path)} → using defaults")
    return default


def _per_competition(df, X, y, model_factory, metric_fn, label):
    """Final-model 5-fold OOF metric, broken down by competition."""
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.full(len(df), np.nan)
    for tr, te in tscv.split(X):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        if hasattr(m, "predict_proba"):
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        else:
            oof[te] = m.predict(X.iloc[te])
    mask = ~np.isnan(oof)
    for comp in ("E", "U"):
        cm = mask & (df["competition"].values == comp)
        if cm.sum():
            print(f"    {label} [{comp}] n={cm.sum():>4}: {metric_fn(y.values[cm], oof[cm])}")


def train() -> int:
    if not os.path.exists(DATA_PATH):
        print(f"missing training data: {DATA_PATH}", file=sys.stderr)
        return 1
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[train] reading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["is_eurocup"] = (df["competition"] == "U").astype(int)
    print(f"  {len(df):,} games, {df['date'].min().date()} → {df['date'].max().date()}, "
          f"by competition: {df.groupby('competition').size().to_dict()}")

    features = feature_list()
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[train] FAIL — training_data missing cols: {missing}", file=sys.stderr)
        return 1
    df = df.dropna(subset=features)
    print(f"  after dropna(features): {len(df):,} rows × {len(features)} features")

    X = df[features]
    y_win, y_total = df["home_win"], df["total_points"]
    tscv = TimeSeriesSplit(n_splits=5)

    # ---- Winner -------------------------------------------------------
    print(f"\n🏆 winner ({FAMILY})")
    params_w = load_params(WINNER_PARAMS, {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "eval_metric": "logloss", "random_state": 42})
    accs, briers = [], []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        clf = build_winner(params_w)
        clf.fit(X.iloc[tr], y_win.iloc[tr])
        proba = clf.predict_proba(X.iloc[te])[:, 1]
        accs.append(accuracy_score(y_win.iloc[te], (proba >= 0.5).astype(int)))
        briers.append(brier_score_loss(y_win.iloc[te], proba))
        print(f"  fold {fold}: acc={accs[-1]:.4f}  brier={briers[-1]:.4f}  (n_test={len(te)})")
    print(f"  TS-CV mean: acc={np.mean(accs):.4f}  brier={np.mean(briers):.4f}")
    _per_competition(df, X, y_win, lambda: build_winner(params_w),
                     lambda yt, yp: f"acc={accuracy_score(yt,(yp>=0.5).astype(int)):.4f} brier={brier_score_loss(yt,yp):.4f}",
                     "winner")

    # Final fit on ALL rows. The old 95/5 split existed only to feed XGBoost an
    # eval_set it never early-stopped on; it just discarded the newest 5%.
    final_w = build_winner(params_w)
    final_w.fit(X, y_win)
    top = describe(final_w, features)[:8]
    print("  top predictors:")
    for name, imp in top:
        print(f"    {imp:.4f}  {name}")
    with open(WINNER_MODEL, "wb") as f:
        pickle.dump(final_w, f)
    with open(WINNER_FEATURES, "w") as f:
        json.dump(features, f, indent=2)
    print(f"  → {WINNER_MODEL}")

    # ---- Total --------------------------------------------------------
    print(f"\n🔢 total ({FAMILY})")
    params_t = load_params(TOTAL_PARAMS, {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "random_state": 42})
    maes, r2s = [], []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        reg = build_total(params_t)
        reg.fit(X.iloc[tr], y_total.iloc[tr])
        pred = reg.predict(X.iloc[te])
        maes.append(mean_absolute_error(y_total.iloc[te], pred))
        r2s.append(r2_score(y_total.iloc[te], pred))
        print(f"  fold {fold}: MAE={maes[-1]:.2f}  R²={r2s[-1]:.3f}  (n_test={len(te)})")
    print(f"  TS-CV mean: MAE={np.mean(maes):.2f}  R²={np.mean(r2s):.3f}")
    _per_competition(df, X, y_total, lambda: build_total(params_t),
                     lambda yt, yp: f"MAE={mean_absolute_error(yt,yp):.2f}", "total")

    final_t = build_total(params_t)
    final_t.fit(X, y_total)
    with open(TOTAL_MODEL, "wb") as f:
        pickle.dump(final_t, f)
    with open(TOTAL_FEATURES, "w") as f:
        json.dump(features, f, indent=2)
    print(f"  → {TOTAL_MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(train())
