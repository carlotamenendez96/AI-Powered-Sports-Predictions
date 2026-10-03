"""Build the Euroleague/EuroCup training matrix from the canonical corpus.

Mirrors ``ml_project/nba/nba_feature_engineering.py`` (the corpus shares the NBA
column contract) with one Euroleague-specific addition: a game-level
``competition`` column (E / U) carried through to the wide training matrix as a
categorical feature for the combined model.

Why no separate per-competition ELO caches: ``build_corpus`` assigns ``teamId``
per ``(competition, club_code)`` (namespaced), so a club has a different id in E
vs U. Keying ELO / rolling form by ``teamId`` therefore separates the
competition ladders automatically — one cache, two independent ladders. (Clubs
play only one European competition per season, so per-competition rest_days is
correct too; a club moving E↔U across seasons resets its id — a known v1
simplification, fine for a lower-tier EuroCup baseline.)

Reads  ``data_sets/Euroleague/team_game_stats.csv``
Writes ``data_sets/Euroleague/training_data.csv`` + ``euroleague_elo.json``.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_REPO, "data_sets", "Euroleague")
INPUT = os.path.join(DATA, "team_game_stats.csv")
OUTPUT = os.path.join(DATA, "training_data.csv")
ELO_CACHE = os.path.join(DATA, "euroleague_elo.json")

# ELO (basketball-style; home advantage tuned smaller than NBA's 100 since the
# Euroleague home edge is ~61.5% — close to NBA's, keep 100 as a sane default).
ELO_INIT = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 100.0

ROLL_METRICS = {
    "pts":         "teamScore",
    "pts_allowed": "opponentScore",
    "win":         "win",
    "fg_pct":      "fieldGoalsPercentage",
    "fg3_pct":     "threePointersPercentage",
    "ft_pct":      "freeThrowsPercentage",
    "reb":         "reboundsTotal",
    "ast":         "assists",
    "tov":         "turnovers",
    "plus_minus":  "plusMinusPoints",
}
L10_METRICS = ("pts", "pts_allowed", "win", "plus_minus")
VENUE_METRICS = ("pts", "pts_allowed", "win")


# ---------------------------------------------------------------------------
# Pure helpers (identical contract to NBA's; teamId carries the per-competition
# namespacing, so no competition-specific branching is needed here).
# ---------------------------------------------------------------------------

def add_schedule_features(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.sort_values(["teamId", "date"]).copy()
    ts["_d"] = pd.to_datetime(ts["date"])
    diff = ts.groupby("teamId")["_d"].diff().dt.days
    ts["rest_days"] = diff.fillna(7).clip(lower=0, upper=14).astype(int)
    ts["b2b"] = (diff == 1).fillna(False).astype(int)
    return ts.drop(columns=["_d"])


def add_rolling(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.sort_values(["teamId", "date"]).copy()
    for short, col in ROLL_METRICS.items():
        ts[f"l5_{short}"] = (
            ts.groupby("teamId")[col]
              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    for short in L10_METRICS:
        col = ROLL_METRICS[short]
        ts[f"l10_{short}"] = (
            ts.groupby("teamId")[col]
              .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))
    return ts


def add_venue_form(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.sort_values(["teamId", "home", "date"]).copy()
    for short in VENUE_METRICS:
        col = ROLL_METRICS[short]
        ts[f"venue_l5_{short}"] = (
            ts.groupby(["teamId", "home"])[col]
              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    return ts.sort_values(["date", "gameId"])


def compute_elo(games_wide: pd.DataFrame) -> tuple[dict, dict]:
    """Chronological pass → ((gameId, teamId)→pre_elo, final_ratings)."""
    pre: dict[tuple, float] = {}
    ratings: dict[int, float] = {}
    g = games_wide.sort_values(["date", "gameId"]).itertuples(index=False)
    for row in g:
        h, a = int(row.home_teamId), int(row.away_teamId)
        rh = ratings.get(h, ELO_INIT)
        ra = ratings.get(a, ELO_INIT)
        pre[(row.gameId, h)] = rh
        pre[(row.gameId, a)] = ra
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - ELO_HOME_ADV) / 400.0))
        actual_h = 1.0 if row.home_score > row.away_score else 0.0
        delta = ELO_K * (actual_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return pre, ratings


def _pivot_wide(ts: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One row per game with home_<f>/away_<f> + game-level metadata + targets."""
    keep = ["gameId", "date", "season", "postseason", "competition",
            "teamId", "home", "teamScore", "opponentScore"] + feature_cols
    ts = ts[keep].copy()
    home = ts[ts["home"] == 1].drop(columns=["home"]).copy()
    away = ts[ts["home"] == 0].drop(columns=["home"]).copy()
    # competition is shared by both sides → treat as a game key (not prefixed).
    GAME_KEYS = ("gameId", "date", "season", "postseason", "competition")
    home = home.rename(columns={c: f"home_{c}" for c in home.columns if c not in GAME_KEYS})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns if c not in GAME_KEYS})
    wide = home.merge(away, on=list(GAME_KEYS), how="inner")
    wide["home_win"] = (wide["home_teamScore"] > wide["away_teamScore"]).astype(int)
    wide["total_points"] = wide["home_teamScore"] + wide["away_teamScore"]
    return wide


def build(input_path: str = INPUT, output_path: str = OUTPUT,
          elo_cache_path: str = ELO_CACHE) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    print(f"[features] reading {input_path}")
    ts = pd.read_csv(input_path, low_memory=False)
    print(f"  loaded {len(ts):,} team-rows ({ts['gameId'].nunique():,} games, "
          f"seasons {int(ts['season'].min())}→{int(ts['season'].max())}, "
          f"competitions {sorted(ts['competition'].unique())})")

    ts = add_schedule_features(ts)
    ts = add_rolling(ts)
    ts = add_venue_form(ts)

    feature_cols = (
        ["rest_days", "b2b"] +
        [f"l5_{m}" for m in ROLL_METRICS] +
        [f"l10_{m}" for m in L10_METRICS] +
        [f"venue_l5_{m}" for m in VENUE_METRICS])

    wide = _pivot_wide(ts, feature_cols)

    elo_input = wide[["gameId", "date", "home_teamId", "away_teamId",
                      "home_teamScore", "away_teamScore"]].rename(
        columns={"home_teamScore": "home_score", "away_teamScore": "away_score"})
    pre_elo, final_ratings = compute_elo(elo_input)
    wide["home_elo_pre"] = wide.apply(lambda r: pre_elo.get((r["gameId"], int(r["home_teamId"])), ELO_INIT), axis=1)
    wide["away_elo_pre"] = wide.apply(lambda r: pre_elo.get((r["gameId"], int(r["away_teamId"])), ELO_INIT), axis=1)

    required = [f"home_l10_{m}" for m in L10_METRICS] + [f"away_l10_{m}" for m in L10_METRICS] \
             + [f"home_venue_l5_{m}" for m in VENUE_METRICS] + [f"away_venue_l5_{m}" for m in VENUE_METRICS]
    before = len(wide)
    wide = wide.dropna(subset=required)
    print(f"  warmup-drop: {before - len(wide):,} early-season rows; {len(wide):,} games remain")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wide = wide.sort_values(["date", "gameId"]).reset_index(drop=True)
    wide.to_csv(output_path, index=False)
    print(f"\n[features] wrote {len(wide):,} games × {wide.shape[1]} cols → {output_path}")
    print(f"  by competition: {wide.groupby('competition').size().to_dict()}")
    print(f"  home_win rate: {wide['home_win'].mean():.3f}   total μ={wide['total_points'].mean():.1f}")

    # Atomic: verification now rewrites this daily, possibly while a predict run reads it.
    tmp = elo_cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({str(k): round(v, 2) for k, v in final_ratings.items()}, f, indent=2)
    os.replace(tmp, elo_cache_path)
    print(f"  ELO cache: {len(final_ratings)} (team×competition) ladders → {elo_cache_path}")
    return wide


if __name__ == "__main__":
    try:
        build()
    except Exception as e:
        print(f"[features] FAIL: {e}", file=sys.stderr)
        sys.exit(1)
