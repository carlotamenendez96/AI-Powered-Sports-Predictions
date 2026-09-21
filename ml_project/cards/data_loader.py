"""Cards-specific MatchHistory loader.

Unlike `DataLoader` (which requires 1X2 odds and skips files without them),
this loader keeps every file that has HY+AY columns — the cards target —
and preserves Referee / red-card columns when present. No odds are required
(there are no card odds in the corpus).
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

_SEASON_SUFFIX_RE = re.compile(r"_(\d{2})-(\d{2})\.csv$")


def league_label_from_path(path: str) -> str:
    """Stable league label from filename (ENG-Premier_League_25-26 → ENG-Premier_League)."""
    basename = os.path.basename(path)
    m = _SEASON_SUFFIX_RE.search(basename)
    stem = basename[: m.start()] if m else basename[:-4]
    return stem.replace("_", " ")


def load_cards_history(history_dir: str = "data_sets/MatchHistory") -> pd.DataFrame:
    """Concatenate MatchHistory CSVs that carry HY+AY; drop rows missing either.

    Columns normalised to: date, home_team, away_team, FTHG, FTAG, FTR,
    league, HY, AY, HR, AR, Referee (optional → NaN when absent).
    Sorted by date ascending.
    """
    files = sorted(glob.glob(os.path.join(history_dir, "*.csv")))
    frames = []
    for path in files:
        try:
            raw = pd.read_csv(path)
        except Exception as e:
            print(f"[cards] skip {os.path.basename(path)}: {e}")
            continue
        if "HY" not in raw.columns or "AY" not in raw.columns:
            continue

        col_map = {
            "Date": "date",
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "Home": "home_team", "Away": "away_team",
            "FTHG": "FTHG", "FTAG": "FTAG", "FTR": "FTR",
            "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
            "Div": "league", "League": "league",
        }
        df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        # Prefer filename-derived league (stable across seasons) over Div codes.
        df["league"] = league_label_from_path(path)

        try:
            df["date"] = pd.to_datetime(df["date"], format="mixed",
                                        dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        for c in ("HY", "AY", "HR", "AR", "FTHG", "FTAG"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = pd.NA

        if "Referee" not in df.columns:
            df["Referee"] = pd.NA
        else:
            df["Referee"] = df["Referee"].astype("string")

        keep = ["date", "home_team", "away_team", "FTHG", "FTAG", "FTR",
                "league", "HY", "AY", "HR", "AR", "Referee"]
        df = df[keep].dropna(subset=["date", "home_team", "away_team", "HY", "AY"])
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError(f"No MatchHistory files with HY/AY under {history_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    out["total_yellows"] = out["HY"] + out["AY"]
    return out
