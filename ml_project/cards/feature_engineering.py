"""Leakage-free feature builder for the cards market (Paso 6).

Train path: `CardsFeatureEngineer.build_training_frame` walks MatchHistory in
date order and emits one feature row per match.

Serve path: `team_card_rates` / `league_card_prior` / `referee_card_rates`
mirror the same arithmetic with an explicit date cutoff.

Referee rates use ONLY matches STRICTLY before the fixture date with hy/ay
non-null. The catalog `--stats` aggregate is NOT reused (no date cutoff).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import (
    CARD_LINE,
    CARDS_FEATURES,
    CARDS_FORM_WINDOWS,
    REFEREE_MIN_N,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
try:
    from referees.build_referee_history import normalize_referee  # noqa: E402
except ImportError:  # pragma: no cover
    def normalize_referee(raw_name):  # type: ignore[misc]
        raw = str(raw_name or "").strip().lower()
        return (raw.replace(" ", "-") or None), raw_name


def _mean_or_nan(vals) -> float:
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _rates_from_records(records: list, n: int) -> Tuple[float, float, float]:
    """From a list of (y_for, y_against, red) oldest→newest, take last n."""
    window = records[-n:] if records else []
    if not window:
        return float("nan"), float("nan"), float("nan")
    yf = [r[0] for r in window if r[0] == r[0]]
    ya = [r[1] for r in window if r[1] == r[1]]
    reds = [r[2] for r in window if r[2] == r[2]]
    return _mean_or_nan(yf), _mean_or_nan(ya), _mean_or_nan(reds)


def team_card_rates(history: pd.DataFrame, team: str, date_before,
                    windows=CARDS_FORM_WINDOWS,
                    venue: Optional[str] = None) -> Dict[str, float]:
    """Serve-time rolling yellow/red rates. Strictly date < date_before."""
    out: Dict[str, float] = {}
    for _, sfx in windows:
        out[f"card_for{sfx}"] = float("nan")
        out[f"card_against{sfx}"] = float("nan")
    out["red_rate"] = float("nan")
    if history is None or history.empty or not team:
        return out

    date_before = pd.Timestamp(date_before)
    mask = history["date"] < date_before
    if venue == "home":
        mask = mask & (history["home_team"] == team)
    elif venue == "away":
        mask = mask & (history["away_team"] == team)
    else:
        mask = mask & ((history["home_team"] == team) | (history["away_team"] == team))

    games = history.loc[mask].sort_values("date")
    records = []
    for row in games.itertuples():
        is_home = row.home_team == team
        yf = float(row.HY) if pd.notna(row.HY) else float("nan")
        ya = float(row.AY) if pd.notna(row.AY) else float("nan")
        if not is_home:
            yf, ya = ya, yf
        r_col = row.HR if is_home else row.AR
        red = float(r_col > 0) if pd.notna(r_col) else float("nan")
        records.append((yf, ya, red))

    for n, sfx in windows:
        yf, ya, _ = _rates_from_records(records, n)
        out[f"card_for{sfx}"] = yf
        out[f"card_against{sfx}"] = ya
    _, _, red = _rates_from_records(records, 5)
    out["red_rate"] = red
    return out


def league_card_prior(history: pd.DataFrame, league: str, date_before
                      ) -> Tuple[float, float]:
    """Expanding league mean(total_yellows) and P(over), date < cutoff."""
    if history is None or history.empty or not league:
        return float("nan"), float("nan")
    date_before = pd.Timestamp(date_before)
    past = history[(history["league"] == league) & (history["date"] < date_before)]
    if past.empty:
        return float("nan"), float("nan")
    tot = past["HY"].astype(float) + past["AY"].astype(float)
    return float(tot.mean()), float((tot > CARD_LINE).mean())


def referee_card_rates(ref_matches: pd.DataFrame, referee_name_or_id,
                       date_before, min_n: int = REFEREE_MIN_N
                       ) -> Dict[str, float]:
    """Prior yellows/game + red rate for a referee, date-cut and gated."""
    empty = {
        "ref_yellows_pg": float("nan"),
        "ref_red_rate": float("nan"),
        "ref_n_with_cards": 0.0,
        "missing_ref": 1.0,
    }
    if ref_matches is None or ref_matches.empty or not referee_name_or_id:
        return empty

    rid, _ = normalize_referee(referee_name_or_id)
    date_before = pd.Timestamp(date_before)
    df = ref_matches
    dates = pd.to_datetime(df["date"], errors="coerce")
    hy = pd.to_numeric(df["hy"], errors="coerce") if "hy" in df.columns else pd.Series(np.nan, index=df.index)
    ay = pd.to_numeric(df["ay"], errors="coerce") if "ay" in df.columns else pd.Series(np.nan, index=df.index)
    with_cards = hy.notna() & ay.notna()
    past = (dates < date_before) & with_cards

    if "referee_id" in df.columns and rid:
        mask = past & (df["referee_id"].astype(str) == str(rid))
    else:
        mask = past & (
            df["referee_name"].astype(str).str.lower()
            == str(referee_name_or_id).strip().lower()
        )
    sub = df.loc[mask]
    n = len(sub)
    if n < min_n:
        out = dict(empty)
        out["ref_n_with_cards"] = float(n)
        return out

    yellows = (pd.to_numeric(sub["hy"], errors="coerce")
               + pd.to_numeric(sub["ay"], errors="coerce"))
    reds = (pd.to_numeric(sub.get("hr"), errors="coerce").fillna(0)
            + pd.to_numeric(sub.get("ar"), errors="coerce").fillna(0))
    return {
        "ref_yellows_pg": float(yellows.mean()),
        "ref_red_rate": float((reds > 0).mean()),
        "ref_n_with_cards": float(n),
        "missing_ref": 0.0,
    }


def load_referee_matches(path: str = "data_sets/referees/referee_matches.csv"
                         ) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"referee_id": str, "league": str})


class CardsFeatureEngineer:
    """Builds the leakage-free cards feature frame from MatchHistory rows."""

    def __init__(self, ref_matches: Optional[pd.DataFrame] = None):
        self.ref_matches = (ref_matches if ref_matches is not None
                            else load_referee_matches())
        # Pre-index referee history for O(1) prior lookups during the train walk.
        self._ref_index = self._build_ref_index(self.ref_matches)

    @staticmethod
    def _build_ref_index(ref_matches: pd.DataFrame) -> Dict[str, list]:
        """referee_id → list of (date, yellows, had_red) sorted by date."""
        idx: Dict[str, list] = defaultdict(list)
        if ref_matches is None or ref_matches.empty:
            return idx
        df = ref_matches.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        hy = pd.to_numeric(df.get("hy"), errors="coerce")
        ay = pd.to_numeric(df.get("ay"), errors="coerce")
        hr = pd.to_numeric(df.get("hr"), errors="coerce").fillna(0)
        ar = pd.to_numeric(df.get("ar"), errors="coerce").fillna(0)
        ok = hy.notna() & ay.notna() & df["date"].notna()
        for i in df.index[ok]:
            rid = str(df.at[i, "referee_id"]) if "referee_id" in df.columns else ""
            if not rid or rid == "nan":
                name = df.at[i, "referee_name"] if "referee_name" in df.columns else ""
                rid, _ = normalize_referee(name)
                if not rid:
                    continue
            yellows = float(hy.at[i] + ay.at[i])
            had_red = 1.0 if (hr.at[i] + ar.at[i]) > 0 else 0.0
            idx[rid].append((df.at[i, "date"], yellows, had_red))
        for rid in idx:
            idx[rid].sort(key=lambda t: t[0])
        return idx

    def _ref_rates_at(self, referee_raw, date_before) -> Dict[str, float]:
        empty = {
            "ref_yellows_pg": float("nan"),
            "ref_red_rate": float("nan"),
            "ref_n_with_cards": 0.0,
            "missing_ref": 1.0,
        }
        try:
            if referee_raw is None or pd.isna(referee_raw):
                return empty
        except (TypeError, ValueError):
            return empty
        raw = str(referee_raw).strip()
        if not raw or raw.lower() == "nan" or raw.lower() == "<na>":
            return empty
        rid, _ = normalize_referee(raw)
        if not rid or rid not in self._ref_index:
            return empty
        date_before = pd.Timestamp(date_before)
        prior = [t for t in self._ref_index[rid] if t[0] < date_before]
        n = len(prior)
        if n < REFEREE_MIN_N:
            out = dict(empty)
            out["ref_n_with_cards"] = float(n)
            return out
        return {
            "ref_yellows_pg": float(np.mean([t[1] for t in prior])),
            "ref_red_rate": float(np.mean([t[2] for t in prior])),
            "ref_n_with_cards": float(n),
            "missing_ref": 0.0,
        }

    def build_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-pass chronological feature build. Same-day fixtures scored as a
        block before any of them is folded into history (no same-day leakage).
        """
        df = df.sort_values("date").reset_index(drop=True).copy()
        df["total_yellows"] = df["HY"].astype(float) + df["AY"].astype(float)
        df["y_over"] = (df["total_yellows"] > CARD_LINE).astype(int)

        n = len(df)
        arrays = {c: np.full(n, np.nan) for c in CARDS_FEATURES
                  if c not in ("league_cat", "missing_ref")}
        arrays["missing_ref"] = np.ones(n)
        arrays["league_cat"] = np.empty(n, dtype=object)

        has_elo = "H_elo" in df.columns and "A_elo" in df.columns

        # Per-team overall / home / away histories: list of (y_for, y_against, red)
        team_all: Dict[str, list] = defaultdict(list)
        team_home: Dict[str, list] = defaultdict(list)
        team_away: Dict[str, list] = defaultdict(list)
        # Per-league expanding: list of total_yellows
        league_hist: Dict[str, list] = defaultdict(list)

        dates = df["date"].values
        homes = df["home_team"].values
        aways = df["away_team"].values
        leagues = df["league"].values
        hy = df["HY"].astype(float).values
        ay = df["AY"].astype(float).values
        hr = df["HR"].astype(float).values if "HR" in df.columns else np.full(n, np.nan)
        ar = df["AR"].astype(float).values if "AR" in df.columns else np.full(n, np.nan)
        refs = df["Referee"].values if "Referee" in df.columns else [None] * n
        h_elo = df["H_elo"].values if has_elo else None
        a_elo = df["A_elo"].values if has_elo else None

        order = np.argsort(dates, kind="stable")
        i = 0
        while i < n:
            j = i
            while j < n and dates[order[j]] == dates[order[i]]:
                j += 1

            # Score the whole day off history that predates it.
            for k in range(i, j):
                pos = order[k]
                home, away, league = homes[pos], aways[pos], leagues[pos]
                date = dates[pos]

                for n_w, sfx in CARDS_FORM_WINDOWS:
                    yf, ya, _ = _rates_from_records(team_all[home], n_w)
                    arrays[f"H_card_for{sfx}"][pos] = yf
                    arrays[f"H_card_against{sfx}"][pos] = ya
                    yf, ya, _ = _rates_from_records(team_all[away], n_w)
                    arrays[f"A_card_for{sfx}"][pos] = yf
                    arrays[f"A_card_against{sfx}"][pos] = ya

                yf, ya, _ = _rates_from_records(team_home[home], 5)
                arrays["H_home_card_for"][pos] = yf
                arrays["H_home_card_against"][pos] = ya
                yf, ya, _ = _rates_from_records(team_away[away], 5)
                arrays["A_away_card_for"][pos] = yf
                arrays["A_away_card_against"][pos] = ya

                _, _, red = _rates_from_records(team_all[home], 5)
                arrays["H_red_rate"][pos] = red
                _, _, red = _rates_from_records(team_all[away], 5)
                arrays["A_red_rate"][pos] = red

                hf = arrays["H_home_card_for"][pos]
                af = arrays["A_away_card_for"][pos]
                if hf == hf and af == af:
                    arrays["expected_yellows_proxy"][pos] = hf + af
                else:
                    hfo = arrays["H_card_for"][pos]
                    afo = arrays["A_card_for"][pos]
                    if hfo == hfo and afo == afo:
                        arrays["expected_yellows_proxy"][pos] = hfo + afo

                past_league = league_hist[league]
                if past_league:
                    arrays["league_mean_yellows"][pos] = float(np.mean(past_league))
                    arrays["league_p_over"][pos] = float(
                        np.mean([1.0 if t > CARD_LINE else 0.0 for t in past_league]))

                rates = self._ref_rates_at(refs[pos], date)
                for key in ("ref_yellows_pg", "ref_red_rate",
                            "ref_n_with_cards", "missing_ref"):
                    arrays[key][pos] = rates[key]

                arrays["league_cat"][pos] = league
                if has_elo and h_elo is not None:
                    he, ae = h_elo[pos], a_elo[pos]
                    if pd.notna(he) and pd.notna(ae):
                        arrays["elo_diff"][pos] = float(he) - float(ae)
                        arrays["abs_elo_diff"][pos] = abs(float(he) - float(ae))

            # Fold the day into history for later dates.
            for k in range(i, j):
                pos = order[k]
                home, away, league = homes[pos], aways[pos], leagues[pos]
                hy_v, ay_v = float(hy[pos]), float(ay[pos])
                hr_v = float(hr[pos]) if hr[pos] == hr[pos] else float("nan")
                ar_v = float(ar[pos]) if ar[pos] == ar[pos] else float("nan")
                h_red = float(hr_v > 0) if hr_v == hr_v else float("nan")
                a_red = float(ar_v > 0) if ar_v == ar_v else float("nan")

                team_all[home].append((hy_v, ay_v, h_red))
                team_all[away].append((ay_v, hy_v, a_red))
                team_home[home].append((hy_v, ay_v, h_red))
                team_away[away].append((ay_v, hy_v, a_red))
                league_hist[league].append(hy_v + ay_v)

            i = j
            if i % 3000 < (j - i) or i == n:
                print(f"  [cards FE] {i}/{n} rows…", flush=True)

        for c, arr in arrays.items():
            df[c] = arr
        df["league_cat"] = df["league_cat"].astype("category")
        return df

    def build_serve_row(self, history: pd.DataFrame, home: str, away: str,
                        league: str, date_before, referee_name: Optional[str],
                        h_elo: float = 1500.0, a_elo: float = 1500.0
                        ) -> Dict[str, float]:
        """One serve-time feature dict, same columns as training."""
        h_all = team_card_rates(history, home, date_before, venue=None)
        a_all = team_card_rates(history, away, date_before, venue=None)
        h_home = team_card_rates(history, home, date_before, venue="home",
                                 windows=((5, ""),))
        a_away = team_card_rates(history, away, date_before, venue="away",
                                 windows=((5, ""),))
        lm, lp = league_card_prior(history, league, date_before)
        rates = self._ref_rates_at(referee_name, date_before)

        row: Dict[str, float] = {}
        for _, sfx in CARDS_FORM_WINDOWS:
            row[f"H_card_for{sfx}"] = h_all[f"card_for{sfx}"]
            row[f"H_card_against{sfx}"] = h_all[f"card_against{sfx}"]
            row[f"A_card_for{sfx}"] = a_all[f"card_for{sfx}"]
            row[f"A_card_against{sfx}"] = a_all[f"card_against{sfx}"]
        row["H_home_card_for"] = h_home["card_for"]
        row["H_home_card_against"] = h_home["card_against"]
        row["A_away_card_for"] = a_away["card_for"]
        row["A_away_card_against"] = a_away["card_against"]
        row["H_red_rate"] = h_all["red_rate"]
        row["A_red_rate"] = a_all["red_rate"]

        hf, af = row["H_home_card_for"], row["A_away_card_for"]
        if hf == hf and af == af:
            row["expected_yellows_proxy"] = hf + af
        else:
            hfo, afo = row["H_card_for"], row["A_card_for"]
            row["expected_yellows_proxy"] = (
                hfo + afo if hfo == hfo and afo == afo else float("nan"))

        row["league_mean_yellows"] = lm
        row["league_p_over"] = lp
        row.update(rates)
        row["league_cat"] = league
        row["elo_diff"] = float(h_elo) - float(a_elo)
        row["abs_elo_diff"] = abs(float(h_elo) - float(a_elo))
        return row


def verify_train_serve_parity(df: pd.DataFrame, n_samples: int = 30,
                              seed: int = 0) -> dict:
    """Sample (team, date) pairs: train H_card_for vs serve team_card_rates."""
    rng = np.random.default_rng(seed)
    candidates = df.iloc[200:].dropna(subset=["H_card_for"]).index.tolist()
    if not candidates:
        return {"checked": 0, "mismatches": 0}
    n_samples = min(n_samples, len(candidates))
    picks = rng.choice(candidates, size=n_samples, replace=False)
    mismatches = 0
    for idx in picks:
        row = df.loc[idx]
        serve = team_card_rates(df, row["home_team"], row["date"], venue=None)
        if abs(float(row["H_card_for"]) - float(serve["card_for"])) > 1e-9:
            mismatches += 1
    return {"checked": n_samples, "mismatches": mismatches}
