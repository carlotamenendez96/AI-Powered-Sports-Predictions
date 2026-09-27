"""euroleague-api daily refresh + fixtures for the Euroleague/EuroCup pipeline.

Mirrors ``ml_project/nba/fetch_nba_daily.py``. Two modes:

* ``append-results [--date YYYY-MM-DD]`` (default yesterday)
    Pulls the current season's schedule+box-scores for BOTH competitions (E, U),
    keeps games PLAYED on the target date, transforms them to the canonical
    long-format via ``build_corpus`` helpers, and appends to
    ``data_sets/Euroleague/team_game_stats.csv`` — idempotent (dedups on
    (gameId, teamId), keeping the freshest row).

* ``fixtures [--date YYYY-MM-DD]`` (default tomorrow)
    Writes ``data_sets/Euroleague/fixtures_<date>.json`` — the upcoming (not-yet
    -played) games on the target date, in the same shape NBA's predictor reads:
    {gameId, competition, date, home_team_id, away_team_id, home_team,
     away_team, tipoff}.

The euroleague-api ``get_game_report_single_season`` returns the FULL season
schedule (played + unplayed, with dates) so both modes derive from it; results
also need ``get_game_stats_single_season`` for the box scores. ``time.sleep(1)``
before each API call. Season code = the ending year (2026 = the 2025-26 season).

Usage::
    python3 ml_project/euroleague/fetch_euroleague_daily.py append-results [--date YYYY-MM-DD]
    python3 ml_project/euroleague/fetch_euroleague_daily.py fixtures [--date YYYY-MM-DD]
"""

import argparse
import datetime
import json
import os
import sys
import time

import pandas as pd

from euroleague_utils import DATA_DIR, CORPUS, COMPETITIONS, COMPETITION_NAMES, TeamIdRegistry
from build_corpus import rows_for_merged, finalize_rows, CANONICAL_COLS

SLEEP_S = 1.0


def _season_code(d: datetime.date) -> int:
    """Euroleague season runs ~Oct→May; euroleague-api codes it by STARTING year.

    A game in Aug–Dec belongs to season ``year``; Jan–Jul to ``year-1``.
    e.g. 2024-10-03 → 2024 (the 2024-25 season); 2026-05-24 → 2025 (2025-26).
    (Verified against the raw data: E_2024_game_report has games dated Oct 2024.)
    """
    return d.year if d.month >= 8 else d.year - 1


def _on_date(rep: pd.DataFrame, date_str: str) -> pd.Series:
    """Boolean mask: report rows whose local/utc date matches date_str."""
    col = "localDate" if "localDate" in rep.columns else "date"
    d = pd.to_datetime(rep[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return d == date_str


def append_results(date_str: str) -> int:
    season = _season_code(datetime.date.fromisoformat(date_str))
    print(f"[append-results] {date_str} (season {season}) — both competitions")
    try:
        from euroleague_api.game_stats import GameStats
    except ImportError as e:
        print(f"[append-results] euroleague-api is not importable ({e}).")
        print("                 Install it: source venv/bin/activate && pip install -r requirements.txt")
        return 1
    new_parts = []
    failed = []
    for comp in COMPETITIONS:
        gs = GameStats(competition=comp)
        try:
            time.sleep(SLEEP_S)
            rep = gs.get_game_report_single_season(season)  # cheap schedule (bulk)
        except Exception as e:
            # Tolerated per-competition, same reasoning as fixtures() below.
            print(f"  ! {COMPETITION_NAMES[comp]}: report fetch failed ({type(e).__name__}: {e})")
            failed.append(comp)
            continue
        played = rep[(rep["played"] == True) & _on_date(rep, date_str)].copy()  # noqa: E712
        played = played.dropna(subset=["local.score", "road.score"])
        if played.empty:
            print(f"  {COMPETITION_NAMES[comp]}: no finished games on {date_str}")
            continue
        # Fetch box scores for ONLY the day's gamecodes (not the whole season).
        stat_rows = []
        for gc in played["Gamecode"].astype(int):
            try:
                time.sleep(SLEEP_S)
                stat_rows.append(gs.get_game_stats(season, int(gc)))
            except Exception as e:
                print(f"  ! {COMPETITION_NAMES[comp]} game {gc}: stats fetch failed ({e})")
        if not stat_rows:
            print(f"  {COMPETITION_NAMES[comp]}: box scores not available yet")
            continue
        sts = pd.concat(stat_rows, ignore_index=True)
        merged = played.merge(sts, on=["Season", "Gamecode"], how="inner", suffixes=("", "_stats"))
        if merged.empty:
            print(f"  {COMPETITION_NAMES[comp]}: results not in box-score feed yet")
            continue
        new_parts.append(rows_for_merged(merged, comp))
        print(f"  {COMPETITION_NAMES[comp]}: {len(merged)} finished game(s)")

    if len(failed) == len(COMPETITIONS):
        print(f"[append-results] EVERY competition report fetch failed ({', '.join(failed)}) — "
              f"broken feed, not an empty day. Corpus left untouched.")
        return 1

    if not new_parts:
        print("[append-results] nothing to append.")
        return 0
    new = finalize_rows(pd.concat(new_parts, ignore_index=True))

    if os.path.exists(CORPUS):
        existing = pd.read_csv(CORPUS, low_memory=False)
        before = len(existing)
        merged = pd.concat([existing, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["gameId", "teamId"], keep="last")
        added = len(merged) - before
    else:
        merged, added = new, len(new)
    merged = (merged.sort_values(["date", "gameId", "home"], ascending=[True, True, False])
              .reset_index(drop=True))
    merged.to_csv(CORPUS, index=False)
    print(f"[append-results] +{added} new team-rows → {CORPUS} ({len(merged):,} total)")
    return 0


def fixtures(date_str: str) -> int:
    season = _season_code(datetime.date.fromisoformat(date_str))
    print(f"[fixtures] {date_str} (season {season}) — both competitions")

    # Imported OUTSIDE the per-competition try since 2026-09-27. It used to sit
    # inside it, so a missing euroleague-api raised ModuleNotFoundError once per
    # competition, both were swallowed, an EMPTY fixtures file was written and
    # the process exited 0 — indistinguishable from a day with no games. The
    # package was in fact absent from requirements.txt for four months and the
    # whole pipeline reported empty slates rather than a broken install.
    try:
        from euroleague_api.schedule import Schedule
    except ImportError as e:
        print(f"[fixtures] euroleague-api is not importable ({e}).")
        print("           Install it: source venv/bin/activate && pip install -r requirements.txt")
        return 1

    reg = TeamIdRegistry()
    out = []
    failed = []
    for comp in COMPETITIONS:
        try:
            time.sleep(SLEEP_S)
            # Schedule.get_schedule, NOT GameStats.get_game_report_single_season
            # (2026-09-27). The game report is RESULTS-only: on 2026-09-27 it
            # returned 10 rows for season 2026, every one played=True, so the
            # `played != True` filter below could never match and this mode
            # returned 0 fixtures for every date — no fixtures, no predictions.
            # The schedule endpoint carries the full 380-game season including
            # unplayed games, and its `gamecode` ("E2026_7") is already exactly
            # the corpus gameId format that build_corpus produces.
            rep = Schedule(competition=comp).get_schedule(season)
        except Exception as e:
            # One competition being unavailable is normal, not fatal: EuroCup
            # raises KeyError('game') on a season with no games yet, and its
            # calendar starts later than EuroLeague's. Hard-failing here would
            # block EuroLeague predictions for the whole early season.
            print(f"  ! {COMPETITION_NAMES[comp]}: fetch failed ({type(e).__name__}: {e})")
            failed.append(comp)
            continue
        # `played` is the STRING "true"/"false" on this endpoint (it is a real
        # bool on the game-report one), so `!= True` would match every row.
        not_played = ~rep["played"].astype(str).str.strip().str.lower().eq("true")
        upcoming = rep[not_played & _on_date(rep, date_str)]
        for _, g in upcoming.iterrows():
            out.append({
                "gameId": str(g["gamecode"]).strip(),
                "competition": comp,
                "date": date_str,
                "home_team_id": reg.get(comp, str(g["homecode"]).strip()),
                "away_team_id": reg.get(comp, str(g["awaycode"]).strip()),
                "home_team": g.get("hometeam"),
                "away_team": g.get("awayteam"),
                "home_code": str(g["homecode"]).strip(),
                "away_code": str(g["awaycode"]).strip(),
                # `startime` holds the tip-off ("20:15"). NOT `confirmedtime`,
                # which despite the name is a boolean flag ("true") saying the
                # time is confirmed — using it yields a tipoff of "<date> true".
                "tipoff": f"{date_str} {str(g.get('startime') or '').strip()}".strip(),
            })
        print(f"  {COMPETITION_NAMES[comp]}: {len(upcoming)} fixture(s)")

    if len(failed) == len(COMPETITIONS):
        # Deliberately does NOT write the file: an empty fixtures_<date>.json is
        # read downstream as "no games today", so writing one here would convert
        # a total outage into a silently empty slate — and would clobber a good
        # file from an earlier run of the same date.
        print(f"[fixtures] EVERY competition fetch failed ({', '.join(failed)}) — "
              f"this is a broken feed, not an empty slate. No file written.")
        return 1

    reg.save()
    out_path = os.path.join(DATA_DIR, f"fixtures_{date_str}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)
    print(f"[fixtures] wrote {len(out)} fixtures → {out_path}")
    if failed:
        print(f"[fixtures] NOTE: {', '.join(COMPETITION_NAMES[c] for c in failed)} "
              f"unavailable this run — fixtures above cover the rest only.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("append-results", help="Append finished games for a date (default yesterday).")
    r.add_argument("--date", default=None, help="YYYY-MM-DD")
    f = sub.add_parser("fixtures", help="Write the day's scheduled fixtures (default tomorrow).")
    f.add_argument("--date", default=None, help="YYYY-MM-DD")
    args = ap.parse_args()

    today = datetime.date.today()
    if args.mode == "append-results":
        d = args.date or (today - datetime.timedelta(days=1)).isoformat()
        return append_results(d)
    else:
        d = args.date or (today + datetime.timedelta(days=1)).isoformat()
        return fixtures(d)


if __name__ == "__main__":
    sys.exit(main())
