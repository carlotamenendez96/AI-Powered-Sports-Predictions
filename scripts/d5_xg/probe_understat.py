"""D5 / gate-2(ii) — Understat historical-xG probe (READ-ONLY, reproducible).

Scopes the "historical xG as a TRAINING feature" test — the one signal the D2
arc flagged as *maybe* not market-priced-dead (we use xG live but never in
training). Understat is chosen over FBref for this: plain HTTP (no Selenium /
Cloudflare — discreet, fast), xG-native, and the reference xG dataset.

FINDINGS (2026-05-27) — see FOOTBALL_NEXT_STEPS "D5 → gate-2(ii)". (The probe
was written against the old single NEXT_STEPS.md, which has since been split
per-sport; the findings moved with it.)
  * DATA SHAPE: `Understat.read_schedule()` already carries match-level
    `home_xg` / `away_xg` next to `date, home_team, away_team, home_goals,
    away_goals, is_result` — the whole signal in ONE plain-HTTP call per
    league-season. No need for the heavier `read_team_match_stats`.
  * COVERAGE: Big-5 only (ENG/ESP/GER/ITA/FRA Premier divisions) + RFPL on the
    site — i.e. the MOST efficient markets, the opposite of where D2 says the
    edge lives. xG window ~2014/15+, so the feature is NaN for the 2010-2014
    slice of our MatchHistory corpus → the clean OOF test is the Big-5, 2014+
    subset (XGBoost tolerates NaN, but isolate to measure xG's contribution).
  * JOIN CRUX: Understat full official names ("Manchester City", "Nottingham
    Forest", "Wolverhampton Wanderers") vs our football-data corpus short names
    ("Man City", "Nott'm Forest", "Wolves"). Within-league WRatio + a small
    override map (the D6 lesson: keep the join within-league to avoid the
    cross-league false matches that sank a naive global fuzzy). This probe
    reports the live match rate.
  * PRIOR: low. D2 (cheap features dead) + D6 (a 2nd ELO source added nothing)
    say more tabular signal rarely moves Brier; the honest expectation is this
    closes like D6. But it's the one untested "maybe" and Understat makes the
    test cheap — so run it, don't assume.

Usage::
    python3 scripts/d5_xg/probe_understat.py shape [ENG-Premier League] [2023]
    python3 scripts/d5_xg/probe_understat.py join  [ENG-Premier League] [2023]
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Understat league key → football-data MatchHistory filename stem, for the join.
LEAGUE_TO_CORPUS = {
    "ENG-Premier League": "ENG-Premier_League",
    "ESP-La Liga":        "ESP-La_Liga",
    "GER-Bundesliga":     "GER-Bundesliga",
    "ITA-Serie A":        "ITA-Serie_A",
    "FRA-Ligue 1":        "FRA-Ligue_1",
}


def _understat_schedule(league: str, season: str):
    import soccerdata as sd
    us = sd.Understat(leagues=league, seasons=season)
    return us.read_schedule()


def shape(league="ENG-Premier League", season="2023"):
    sched = _understat_schedule(league, season)
    xg_cols = [c for c in sched.columns if "xg" in c.lower()]
    print(f"{league} {season}: {len(sched)} matches")
    print("xG columns:", xg_cols)
    print("date range:", sched["date"].min(), "→", sched["date"].max())
    cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]
    print(sched[cols].head(6).to_string(index=False))


def _corpus_teams(league: str, season: str):
    """Unique team names from the matching football-data MatchHistory CSV.

    season "2023" (Understat = 2023/24) → file stem `_23-24`.
    """
    import csv, glob
    stem = LEAGUE_TO_CORPUS.get(league)
    if not stem:
        return []
    yy = int(season) % 100
    fname = os.path.join(ROOT, "data_sets", "MatchHistory", f"{stem}_{yy}-{yy+1}.csv")
    if not os.path.exists(fname):
        cands = glob.glob(os.path.join(ROOT, "data_sets", "MatchHistory", f"{stem}_*.csv"))
        fname = sorted(cands)[-1] if cands else None
    if not fname:
        return []
    rows = list(csv.DictReader(open(fname)))
    col = "home_team" if rows and "home_team" in rows[0] else "HomeTeam"
    return sorted({r[col] for r in rows if r.get(col)}), os.path.basename(fname)


def join(league="ENG-Premier League", season="2023"):
    """Live name-reconciliation demo: Understat names → corpus names (WRatio)."""
    from rapidfuzz import process, fuzz

    sched = _understat_schedule(league, season)
    us_names = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    corpus_names, fname = _corpus_teams(league, season)
    print(f"Understat: {len(us_names)} teams | corpus ({fname}): {len(corpus_names)} teams\n")
    print(f"{'understat':<26}{'-> corpus (WRatio)':<22}{'score'}")
    print("-" * 56)
    matched = 0
    for u in us_names:
        best = process.extractOne(u, corpus_names, scorer=fuzz.WRatio)
        score = best[1] if best else 0.0
        ok = bool(best) and score >= 80
        matched += bool(ok)
        flag = "" if ok else "  ⚠ below 80 → needs override"
        print(f"{u:<26}{(best[0] if best else '-'):<22}{score:>5.0f}{flag}")
    print(f"\nmatch rate @80: {matched}/{len(us_names)} = {matched/len(us_names):.0%}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "shape"
    league = sys.argv[2] if len(sys.argv) > 2 else "ENG-Premier League"
    season = sys.argv[3] if len(sys.argv) > 3 else "2023"
    if cmd == "shape":
        shape(league, season)
    elif cmd == "join":
        join(league, season)
    else:
        print(__doc__)
