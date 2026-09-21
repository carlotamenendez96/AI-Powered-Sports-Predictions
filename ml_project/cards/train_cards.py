"""Train the cards-market binary head (Paso 6).

Isolated from production 1X2 / O/U. Does NOT touch models/xgb_model_{1x2,ou}*
or league_calibration.json.

Usage:
    source venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ml_project
    python3 -m ml_project.cards.train_cards
    python3 -m ml_project.cards.train_cards --with-poisson-diagnostic
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "ml_project") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ml_project"))

from model_registry import get_spec, save_meta  # noqa: E402

from .constants import CARD_LINE, CARDS_FEATURES  # noqa: E402
from .data_loader import load_cards_history  # noqa: E402
from .feature_engineering import (  # noqa: E402
    CardsFeatureEngineer,
    load_referee_matches,
    verify_train_serve_parity,
)


def _attach_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-match ELO on the cards corpus (goals only)."""
    from elo_engine import EloTracker
    cols = ["date", "home_team", "away_team", "FTHG", "FTAG"]
    slim = df[cols].copy().sort_values("date")
    tracker = EloTracker()
    slim = tracker.process_history(slim)
    out = df.merge(
        slim[["date", "home_team", "H_elo"]],
        on=["date", "home_team"], how="left")
    out = out.merge(
        slim[["date", "away_team", "A_elo"]],
        on=["date", "away_team"], how="left")
    return out


def prepare_cards_frame(history_dir: str = "data_sets/MatchHistory",
                        ref_path: str = "data_sets/referees/referee_matches.csv"
                        ) -> pd.DataFrame:
    print("[cards] Loading MatchHistory rows with HY/AY…")
    df = load_cards_history(history_dir)
    print(f"[cards] {len(df)} rows across {df['league'].nunique()} leagues "
          f"(mean yellows={df['total_yellows'].mean():.2f}, "
          f"P(>{CARD_LINE})={(df['total_yellows'] > CARD_LINE).mean():.3f})")

    print("[cards] Attaching pre-match ELO…")
    df = _attach_elo(df)

    print("[cards] Engineering card features (leakage-free)…")
    fe = CardsFeatureEngineer(ref_matches=load_referee_matches(ref_path))
    df = fe.build_training_frame(df)

    parity = verify_train_serve_parity(df, n_samples=40)
    print(f"[cards] Train/serve parity check: {parity['checked']} samples, "
          f"{parity['mismatches']} mismatches")
    if parity["mismatches"]:
        print("[cards] WARN: train/serve parity mismatches — investigate before serving")

    # Drop rows that still lack the minimum serving features (league prior
    # needs at least one prior match in that league).
    needed = ["league_mean_yellows", "H_card_for", "A_card_for", "y_over"]
    before = len(df)
    df = df.dropna(subset=needed).copy()
    print(f"[cards] Dropped {before - len(df)} rows lacking league/team priors "
          f"→ {len(df)} train rows")
    return df


def oof_binary_predictions(df: pd.DataFrame, features: list, family: str = "xgboost",
                           n_splits: int = 5) -> np.ndarray:
    """TimeSeriesSplit OOF P(over). Early rows without a train fold stay NaN."""
    spec = get_spec("cards", family)
    feats = list(features)
    if not spec.uses_categorical and "league_cat" in feats:
        feats = [f for f in feats if f != "league_cat"]

    d = df.sort_values("date").reset_index(drop=True).copy()
    if spec.uses_categorical and "league_cat" in d.columns:
        d["league_cat"] = d["league_cat"].astype("category")

    oof = np.full(len(d), np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, te) in enumerate(tscv.split(d)):
        model = spec.build()
        model.fit(d.iloc[tr][feats], d.iloc[tr]["y_over"])
        probs = model.predict_proba(d.iloc[te][feats])[:, 1]
        oof[te] = probs
        y = d.iloc[te]["y_over"].values
        brier = brier_score_loss(y, probs)
        ll = log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))
        print(f"  fold {fold + 1}: train={len(tr)} test={len(te)} "
              f"Brier={brier:.4f} logloss={ll:.4f}")
    return oof, d, feats


def train_cards(df: pd.DataFrame, family: str = "xgboost",
                models_dir: str = "models", n_splits: int = 5
                ) -> dict:
    print(f"\n--- Training cards binary head (family={family}) ---")
    spec = get_spec("cards", family)
    features = list(CARDS_FEATURES)
    if not spec.uses_categorical and "league_cat" in features:
        features = [f for f in features if f != "league_cat"]

    oof, d, features = oof_binary_predictions(df, features, family=family,
                                              n_splits=n_splits)
    valid = ~np.isnan(oof)
    if valid.any():
        brier = brier_score_loss(d.loc[valid, "y_over"], oof[valid])
        ll = log_loss(d.loc[valid, "y_over"],
                      np.clip(oof[valid], 1e-6, 1 - 1e-6))
        print(f"OOF Brier={brier:.4f}  logloss={ll:.4f}  n={valid.sum()}")

    # Final fit on 95% head (same convention as train_model.py).
    if spec.uses_categorical and "league_cat" in d.columns:
        d["league_cat"] = d["league_cat"].astype("category")
    split = int(len(d) * 0.95)
    final = spec.build()
    final.fit(d.iloc[:split][features], d.iloc[:split]["y_over"])

    os.makedirs(models_dir, exist_ok=True)
    if family == "xgboost":
        artifact = "xgb_model_cards.json"
        final.save_model(os.path.join(models_dir, artifact))
    else:
        import joblib
        artifact = f"sk_model_cards_{family}.joblib"
        joblib.dump(final, os.path.join(models_dir, artifact))
    save_meta("cards", family, artifact, models_dir=models_dir)
    with open(os.path.join(models_dir, "features_cards.json"), "w") as f:
        json.dump(features, f, indent=2)

    # Persist OOF for calibration / experiment scripts.
    oof_path = os.path.join(models_dir, "oof_cards.npz")
    np.savez_compressed(
        oof_path,
        oof=oof,
        y=d["y_over"].values,
        dates=d["date"].astype(str).values,
        leagues=d["league"].astype(str).values,
        missing_ref=d["missing_ref"].values if "missing_ref" in d.columns
        else np.ones(len(d)),
        ref_n=d["ref_n_with_cards"].values if "ref_n_with_cards" in d.columns
        else np.zeros(len(d)),
        league_p_over=d["league_p_over"].values,
        ref_yellows_pg=d["ref_yellows_pg"].values if "ref_yellows_pg" in d.columns
        else np.full(len(d), np.nan),
    )
    print(f"[cards] Saved model → {models_dir}/{artifact}")
    print(f"[cards] Saved features → {models_dir}/features_cards.json")
    print(f"[cards] Saved OOF → {oof_path}")
    return {
        "family": family, "artifact": artifact, "n": int(len(d)),
        "oof_brier": float(brier) if valid.any() else None,
        "oof_logloss": float(ll) if valid.any() else None,
        "features": features,
        "frame": d,
        "oof": oof,
    }


def train_cards_total_diagnostic(df: pd.DataFrame, family: str = "xgboost",
                                 models_dir: str = "models") -> None:
    """Optional Poisson head on total_yellows — diagnostic only."""
    print(f"\n--- Training cards_total Poisson diagnostic (family={family}) ---")
    spec = get_spec("cards_total", family)
    features = list(CARDS_FEATURES)
    if not spec.uses_categorical and "league_cat" in features:
        features = [f for f in features if f != "league_cat"]
    d = df.dropna(subset=["total_yellows"]).sort_values("date").copy()
    if spec.uses_categorical and "league_cat" in d.columns:
        d["league_cat"] = d["league_cat"].astype("category")
    split = int(len(d) * 0.95)
    model = spec.build()
    model.fit(d.iloc[:split][features], d.iloc[:split]["total_yellows"])
    os.makedirs(models_dir, exist_ok=True)
    if family == "xgboost":
        artifact = "xgb_model_cards_total.json"
        model.save_model(os.path.join(models_dir, artifact))
    else:
        import joblib
        artifact = f"sk_model_cards_total_{family}.joblib"
        joblib.dump(model, os.path.join(models_dir, artifact))
    save_meta("cards_total", family, artifact, models_dir=models_dir)
    with open(os.path.join(models_dir, "features_cards_total.json"), "w") as f:
        json.dump(features, f, indent=2)
    print(f"[cards] Saved Poisson diagnostic → {models_dir}/{artifact}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=os.environ.get("MODEL_FAMILY_CARDS", "xgboost"))
    parser.add_argument("--with-poisson-diagnostic", action="store_true")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--history-dir", default="data_sets/MatchHistory")
    args = parser.parse_args()

    df = prepare_cards_frame(args.history_dir)
    train_cards(df, family=args.family, models_dir=args.models_dir)
    if args.with_poisson_diagnostic:
        train_cards_total_diagnostic(df, family="xgboost",
                                     models_dir=args.models_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
