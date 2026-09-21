#!/usr/bin/env python3
"""Serve-time cards predictions (Paso 6).

Reads matches_<date>.json + referees_<date>.json + MatchHistory rates,
writes output/predictions_cards_<date>.csv.

Skips matches that cannot build the same feature vector as training
(missing team history / league prior). Never silently zero-fills.

Usage:
    python3 -m ml_project.cards.predict_cards 2026-09-21
    python3 ml_project/cards/predict_cards.py 2026-09-21
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "ml_project") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ml_project"))

from model_registry import load_cards_model  # noqa: E402

from .calibration import apply_cards_calibrator, load_cards_calibration  # noqa: E402
from .config import get_cards_config  # noqa: E402
from .constants import CARD_LINE, CARDS_FEATURES  # noqa: E402
from .data_loader import load_cards_history  # noqa: E402
from .feature_engineering import CardsFeatureEngineer, load_referee_matches  # noqa: E402


# Flashscore "COUNTRY: League" → MatchHistory league label used in training.
# Only leagues that appear in the cards training universe matter; others are
# skipped (no silent feature skew).
_FLASHSCORE_TO_CARDS_LEAGUE = {
    "ENGLAND: Premier League": "ENG-Premier League",
    "ENGLAND: Championship": "ENG-Championship",
    "ENGLAND: League One": "ENG-League 1",
    "ENGLAND: League 1": "ENG-League 1",
    "ENGLAND: League Two": "ENG-League 2",
    "ENGLAND: League 2": "ENG-League 2",
    "ENGLAND: National League": "ENG-Conference",
    "SCOTLAND: Premiership": "SCO-Premier League",
    "SCOTLAND: Championship": "SCO-Division 1",
    "SCOTLAND: League One": "SCO-Division 2",
    "SCOTLAND: League Two": "SCO-Division 3",
    "SPAIN: LaLiga": "ESP-La Liga",
    "SPAIN: La Liga": "ESP-La Liga",
    "SPAIN: LaLiga2": "ESP-Segunda",
    "SPAIN: LaLiga 2": "ESP-Segunda",
    "FRANCE: Ligue 1": "FRA-Ligue 1",
    "FRANCE: Ligue 2": "FRA-Ligue 2",
    "GERMANY: Bundesliga": "GER-Bundesliga",
    "GERMANY: 2. Bundesliga": "GER-Bundesliga 2",
    "ITALY: Serie A": "ITA-Serie A",
    "ITALY: Serie B": "ITA-Serie B",
    "NETHERLANDS: Eredivisie": "NED-Eredivisie",
    "PORTUGAL: Liga Portugal": "POR-Liga 1",
    "PORTUGAL: Liga NOS": "POR-Liga 1",
    "BELGIUM: Jupiler Pro League": "BEL-Jupiler League",
    "BELGIUM: Pro League": "BEL-Jupiler League",
    "TURKEY: Super Lig": "TUR-Ligi 1",
    "TURKEY: Süper Lig": "TUR-Ligi 1",
    "GREECE: Super League": "GR-Super League",
}


def _canonical_league(flashscore_name: str) -> Optional[str]:
    if not flashscore_name:
        return None
    name = flashscore_name
    if ":" in name:
        country, rest = name.split(":", 1)
        base = rest.split(" - ", 1)[0].strip()
        name = f"{country.strip()}: {base}"
    return _FLASHSCORE_TO_CARDS_LEAGUE.get(name)


class CardsPredictor:
    def __init__(self, history_dir: str = "data_sets/MatchHistory",
                 models_dir: str = "models"):
        self.cfg = get_cards_config()
        self.model, self.family = load_cards_model(models_dir)
        feat_path = os.path.join(models_dir, "features_cards.json")
        with open(feat_path) as f:
            self.features = json.load(f)
        self.history = load_cards_history(history_dir)
        self.fe = CardsFeatureEngineer(ref_matches=load_referee_matches())
        self.cal = load_cards_calibration()
        self.use_cal = bool(self.cfg.get("use_calibration")) and bool(
            self.cal.get("accepted"))

        # Known league categories from the model (XGBoost) or from history.
        self.known_leagues = sorted(self.history["league"].astype(str).unique())

        # ELO lookup (final ratings snapshot if present).
        self.elo = {}
        elo_path = "data_sets/elo_ratings.json"
        if os.path.exists(elo_path):
            with open(elo_path) as f:
                self.elo = json.load(f)

        try:
            from entity_resolver import EntityResolver
            self.resolver = EntityResolver()
        except Exception:
            self.resolver = None

    def _resolve(self, name: str) -> str:
        if self.resolver is None:
            return name
        try:
            return self.resolver.get_canonical_name(name) or name
        except Exception:
            return name

    def _elo(self, name: str) -> float:
        if self.resolver is not None:
            try:
                v = self.resolver.get_elo(name)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return float(v)
            except Exception:
                pass
        canon = self._resolve(name)
        return float(self.elo.get(canon, self.elo.get(name, 1500)))

    def predict_date(self, date: str,
                     matches_path: Optional[str] = None,
                     referees_path: Optional[str] = None,
                     out_path: Optional[str] = None) -> pd.DataFrame:
        matches_path = matches_path or f"output/matches_{date}.json"
        referees_path = referees_path or f"output/referees_{date}.json"
        out_path = out_path or f"output/predictions_cards_{date}.csv"

        if not os.path.exists(matches_path):
            print(f"[cards] Missing {matches_path}")
            return pd.DataFrame()

        with open(matches_path) as f:
            matches = json.load(f)
        referees = {}
        if os.path.exists(referees_path):
            with open(referees_path) as f:
                referees = json.load(f)

        rows = []
        skipped = []
        for m in matches:
            mid = m.get("match_id", "")
            fs_league = m.get("league", "")
            cards_league = _canonical_league(fs_league)
            if not cards_league:
                skipped.append((mid, "league_not_in_cards_universe", fs_league))
                continue

            home_fs = m.get("home_team", "")
            away_fs = m.get("away_team", "")
            home = self._resolve(home_fs)
            away = self._resolve(away_fs)

            kickoff = m.get("start_time", date)
            try:
                date_obj = pd.to_datetime(kickoff, dayfirst=True)
            except Exception:
                date_obj = pd.Timestamp(date)

            ref_rec = referees.get(mid) or {}
            ref_name = (ref_rec.get("referee_name") if ref_rec else None)

            feats = self.fe.build_serve_row(
                self.history, home, away, cards_league, date_obj,
                referee_name=ref_name,
                h_elo=self._elo(home_fs),
                a_elo=self._elo(away_fs),
            )

            # Completeness: require league prior + both sides' L5 card_for.
            missing = []
            for key in ("league_mean_yellows", "H_card_for", "A_card_for"):
                v = feats.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    missing.append(key)
            if missing:
                skipped.append((mid, "incomplete_features", ",".join(missing)))
                continue

            input_df = pd.DataFrame([feats])
            input_df["league_cat"] = pd.Categorical(
                input_df["league_cat"], categories=self.known_leagues)
            for c in self.features:
                if c not in input_df.columns:
                    input_df[c] = np.nan

            p_raw = float(self.model.predict_proba(input_df[self.features])[0, 1])
            p_cal, applied, cal_src = apply_cards_calibrator(
                p_raw, self.cal, enabled=self.use_cal)
            pick = "Over 3.5" if p_cal >= 0.5 else "Under 3.5"
            conf = p_cal if p_cal >= 0.5 else 1.0 - p_cal

            rows.append({
                "Date": date_obj.strftime("%Y-%m-%d %H:%M")
                if hasattr(date_obj, "strftime") else str(date_obj),
                "League": fs_league,
                "Cards League": cards_league,
                "Home Team": home_fs,
                "Away Team": away_fs,
                "Prediction Cards": pick,
                "Conf Cards": f"{conf:.2f}",
                "Over %": f"{p_cal:.2f}",
                "Under %": f"{1.0 - p_cal:.2f}",
                "Over % (raw)": f"{p_raw:.2f}",
                "Under % (raw)": f"{1.0 - p_raw:.2f}",
                "Cal Source": cal_src or "",
                "Referee": ref_name or "",
                "n_ref": int(feats.get("ref_n_with_cards") or 0),
                "missing_ref": int(feats.get("missing_ref") or 1),
                "league_mean_yellows": round(float(feats["league_mean_yellows"]), 3),
                "league_p_over": round(float(feats["league_p_over"]), 3),
                "expected_yellows_proxy": (
                    round(float(feats["expected_yellows_proxy"]), 3)
                    if feats.get("expected_yellows_proxy") == feats.get("expected_yellows_proxy")
                    else ""),
                "Card Line": CARD_LINE,
                "match_id": mid,
            })

        res = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if not res.empty:
            res.to_csv(out_path, index=False)
            print(f"[cards] Wrote {len(res)} predictions → {out_path}")
        else:
            print(f"[cards] No serveable matches for {date}")
        if skipped:
            print(f"[cards] Skipped {len(skipped)} matches "
                  f"(not in universe / incomplete features)")
            for mid, reason, detail in skipped[:8]:
                print(f"         {mid}: {reason} ({detail})")
            if len(skipped) > 8:
                print(f"         … +{len(skipped) - 8} more")
        return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="YYYY-MM-DD slate date")
    parser.add_argument("--force", action="store_true",
                        help="Run even if cards.enabled is false in config")
    args = parser.parse_args()

    cfg = get_cards_config()
    if not cfg.get("enabled") and not args.force:
        print("[cards] sports.football.cards.enabled is false "
              "(and --force not set). Skipping. "
              "Enable after experiment_cards.py gate passes, or pass --force.")
        return 0

    model_path = "models/xgb_model_cards.json"
    meta_path = "models/model_meta_cards.json"
    if not os.path.exists(model_path) and not os.path.exists(meta_path):
        print("[cards] No trained cards model found. Run: "
              "python3 -m ml_project.cards.train_cards")
        return 1

    pred = CardsPredictor()
    pred.predict_date(args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
