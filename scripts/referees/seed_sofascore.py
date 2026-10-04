#!/usr/bin/env python3
"""Backfill referee + cards from Sofascore into ``referee_matches.csv``.

football-data.co.uk only publishes ``Referee`` for ENG/SCO. Big-5 elsewhere
(ESP/ITA/FRA/GER) have HY/AY in MatchHistory but no referee name. Sofascore's
unofficial JSON API exposes both per finished event (probe 2026-09-21):

    GET /unique-tournament/{id}/seasons
    GET /unique-tournament/{id}/season/{sid}/events/last/{page}
    GET /event/{id}            → referee {id, name, country}
    GET /event/{id}/statistics → Yellow / Red cards (and corners)

Requires ``curl_cffi`` (plain ``requests`` → 403). Rows are upserted with
``match_key=ss:<event_id>``, ``source=sofascore``, via the same helpers as
``build_referee_history.py`` (idempotent).

Referee ids: Sofascore full names (``Miguel Angel Ortiz Arias``) are mapped
to Flashscore-like ``Surname I.`` slugs via ``normalize_referee`` so daily
``extract_referees`` names can hit the same catalog entry (best-effort;
Spanish double surnames use the last two tokens).

Usage::

    # Smoke (15 matches, current LaLiga season)
    python3 scripts/referees/seed_sofascore.py --tournament laliga --max-matches 15

    # Big-5, last 3 season years (23/24–25/26)
    python3 scripts/referees/seed_sofascore.py --all-big5 --years 23/24,24/25,25/26

    # One tournament + season year
    python3 scripts/referees/seed_sofascore.py --tournament serie_a --years 25/26
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import unicodedata
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "referees"))

from build_referee_history import (  # noqa: E402
    normalize_referee,
    rebuild_catalog,
    upsert_matches,
)

try:
    from curl_cffi import requests as creq
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "curl_cffi is required for Sofascore (TLS fingerprint). "
        "Install with: pip install curl_cffi\n"
        f"Original import error: {e}"
    ) from e

BASE = "https://api.sofascore.com/api/v1"
# Stable Sofascore uniqueTournament ids (probe 2026-09-21).
TOURNAMENTS: dict[str, dict[str, Any]] = {
    "laliga": {
        "id": 8,
        "league": "ESP-La Liga",
        "label": "LaLiga",
        # Verified 2026-09-21 against Sofascore /seasons.
        "seasons": {"25/26": 77559, "24/25": 61643, "23/24": 52376},
    },
    "serie_a": {
        "id": 23,
        "league": "ITA-Serie A",
        "label": "Serie A",
        # Filled on first successful --refresh-seasons (or leave empty → live).
        "seasons": {},
    },
    "ligue_1": {
        "id": 34,
        "league": "FRA-Ligue 1",
        "label": "Ligue 1",
        "seasons": {},
    },
    "bundesliga": {
        "id": 35,
        "league": "GER-Bundesliga",
        "label": "Bundesliga",
        "seasons": {},
    },
}
# Countries where the last two tokens are typically the surname pair.
_DOUBLE_SURNAME_CC = frozenset({"ES", "PT", "MX", "AR", "CL", "CO", "UY", "PE"})


def _get(path: str, timeout: float = 30.0, allow_404: bool = False) -> Optional[dict]:
    url = path if path.startswith("http") else f"{BASE}{path}"
    r = creq.get(url, impersonate="chrome120", timeout=timeout)
    if allow_404 and r.status_code == 404:
        return None
    if r.status_code in (403, 429):
        raise TransientHTTPError(r.status_code, r.reason)
    r.raise_for_status()
    return r.json()


class TransientHTTPError(Exception):
    def __init__(self, code: int, reason: str):
        self.code = code
        super().__init__(f"HTTP Error {code}: {reason}")


def _get_with_retries(path: str, sleep_s: float, tries: int = 6) -> Optional[dict]:
    last_err: Optional[Exception] = None
    for i in range(tries):
        try:
            return _get(path)
        except TransientHTTPError as e:
            last_err = e
            # Soft rate-limit: back off hard on 403/429.
            time.sleep(max(2.0, sleep_s * (5 * (i + 1))))
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (2 + i))
    if last_err:
        raise last_err
    return None


def _fold_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def sofascore_name_to_catalog_id(
    full_name: str, country_alpha2: Optional[str] = None
) -> tuple[Optional[str], str]:
    """Map Sofascore full name → (slug_id, display_kept_as_full_name).

    Flashscore daily extract uses ``Surname I.``; Sofascore uses full names.
    Synthesize a Flashscore-like string before ``normalize_referee``.
    Accented characters are folded (Sánchez → Sanchez) so slugs stay ASCII.
    """
    raw = (full_name or "").strip()
    if not raw:
        return None, ""
    folded = _fold_accents(raw)
    tokens = [t for t in folded.replace(".", " ").split() if t]
    if len(tokens) >= 3 and (country_alpha2 or "").upper() in _DOUBLE_SURNAME_CC:
        surname = f"{tokens[-2]} {tokens[-1]}"
        initial = tokens[0][0]
        synthetic = f"{surname} {initial}."
    elif len(tokens) >= 2:
        surname = tokens[-1]
        initial = tokens[0][0]
        synthetic = f"{surname} {initial}."
    else:
        synthetic = folded
    rid, _ = normalize_referee(synthetic)
    return rid, raw


def _stat_pair(stats_payload: dict, *names: str) -> tuple[Optional[int], Optional[int]]:
    want = {n.lower() for n in names}
    for block in stats_payload.get("statistics") or []:
        if block.get("period") not in (None, "ALL", "all", "Full time", "Match"):
            # Prefer ALL; fall back to any block if ALL absent.
            continue
        for group in block.get("groups") or []:
            for item in group.get("statisticsItems") or []:
                if (item.get("name") or "").lower() in want:
                    return _as_int(item.get("home")), _as_int(item.get("away"))
    # Fallback: scan all periods
    for block in stats_payload.get("statistics") or []:
        for group in block.get("groups") or []:
            for item in group.get("statisticsItems") or []:
                if (item.get("name") or "").lower() in want:
                    return _as_int(item.get("home")), _as_int(item.get("away"))
    return None, None


def _as_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(str(v).replace(",", ".")))
    except (TypeError, ValueError):
        return None


def list_season_ids(tournament_id: int) -> list[dict]:
    data = _get_with_retries(
        f"/unique-tournament/{tournament_id}/seasons", sleep_s=1.0, tries=10
    )
    return list((data or {}).get("seasons") or [])


def pick_seasons(seasons: list[dict], years: Optional[list[str]]) -> list[dict]:
    if not years:
        return seasons[:1]
    want = {y.strip() for y in years}
    return [
        s for s in seasons
        if str(s.get("year") or "") in want or str(s.get("name") or "") in want
    ]


def seasons_for_tournament(
    key: str, years: Optional[list[str]], refresh: bool = False
) -> list[dict]:
    meta = TOURNAMENTS[key]
    cached = meta.get("seasons") or {}
    if not refresh and cached:
        picked = []
        year_list = years or list(cached.keys())[:1]
        for y in year_list:
            sid = cached.get(y)
            if sid:
                picked.append({"id": sid, "year": y, "name": f"{meta['label']} {y}"})
        if picked:
            return picked
    # Live lookup (rate-limited).
    return pick_seasons(list_season_ids(meta["id"]), years)


def iter_season_events(tournament_id: int, season_id: int, sleep_s: float):
    page = 0
    while True:
        data = None
        last_err: Optional[Exception] = None
        for i in range(6):
            try:
                data = _get(
                    f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}",
                    allow_404=True,
                )
                last_err = None
                break
            except TransientHTTPError as e:
                last_err = e
                time.sleep(max(3.0, sleep_s * (6 * (i + 1))))
        if last_err:
            raise last_err
        # Sofascore returns 404 (not empty list) past the last page.
        if data is None:
            break
        events = data.get("events") or []
        if not events:
            break
        for ev in events:
            yield ev
        page += 1
        time.sleep(sleep_s)


def row_from_event(
    event_meta: dict,
    league: str,
    ingested_at: str,
    sleep_s: float,
) -> Optional[dict]:
    status = (event_meta.get("status") or {}).get("type")
    if status != "finished":
        return None
    eid = event_meta.get("id")
    if not eid:
        return None

    time.sleep(sleep_s)
    detail = _get_with_retries(f"/event/{eid}", sleep_s=sleep_s)
    if not detail:
        return None
    ev = detail.get("event") or detail
    ref = ev.get("referee")
    if not isinstance(ref, dict) or not ref.get("name"):
        return None

    time.sleep(sleep_s)
    stats_ok = False
    try:
        stats = _get_with_retries(f"/event/{eid}/statistics", sleep_s=sleep_s) or {}
        stats_ok = bool(stats.get("statistics"))
    except Exception:
        stats = {}

    hy, ay = _stat_pair(stats, "Yellow cards", "Yellow Cards")
    hr, ar = _stat_pair(stats, "Red cards", "Red Cards")
    hc, ac = _stat_pair(stats, "Corner kicks", "Corners")
    # Sofascore omits the Yellow/Red row when the count is zero — if the
    # stats payload loaded, treat missing card lines as 0 (not NaN).
    if stats_ok:
        if hy is None and ay is None:
            hy, ay = 0, 0
        if hr is None and ar is None:
            hr, ar = 0, 0

    cc = (ref.get("country") or {}).get("alpha2")
    ref_id, ref_name = sofascore_name_to_catalog_id(ref["name"], cc)
    if not ref_id:
        return None

    home = (ev.get("homeTeam") or event_meta.get("homeTeam") or {}).get("name") or ""
    away = (ev.get("awayTeam") or event_meta.get("awayTeam") or {}).get("name") or ""
    hs = (ev.get("homeScore") or event_meta.get("homeScore") or {}).get("current")
    aws = (ev.get("awayScore") or event_meta.get("awayScore") or {}).get("current")
    ts = ev.get("startTimestamp") or event_meta.get("startTimestamp")
    if not ts:
        return None
    date_iso = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).strftime("%Y-%m-%d")

    return {
        "match_key": f"ss:{eid}",
        "date": date_iso,
        "league": league,
        "home": home,
        "away": away,
        "score_home": _as_int(hs),
        "score_away": _as_int(aws),
        "referee_name": ref_name,
        "referee_id": ref_id,
        "hy": hy,
        "ay": ay,
        "hr": hr,
        "ar": ar,
        "hc": hc,
        "ac": ac,
        "source": "sofascore",
        "ingested_at": ingested_at,
    }


def existing_ss_keys() -> set[str]:
    path = os.path.join(ROOT, "data_sets", "referees", "referee_matches.csv")
    if not os.path.isfile(path):
        return set()
    import pandas as pd
    try:
        df = pd.read_csv(path, usecols=["match_key"], dtype=str)
    except (ValueError, OSError):
        return set()
    return {k for k in df["match_key"].dropna() if str(k).startswith("ss:")}


def seed_tournament(
    key: str,
    years: Optional[list[str]],
    max_matches: Optional[int],
    sleep_s: float,
    skip_existing: bool,
) -> dict[str, Any]:
    meta = TOURNAMENTS[key]
    tid = meta["id"]
    league = meta["league"]
    summary: dict[str, Any] = {
        "tournament": key,
        "league": league,
        "seasons": [],
        "n_rows": 0,
        "n_skipped_existing": 0,
        "n_no_ref": 0,
        "n_errors": 0,
    }
    seasons = seasons_for_tournament(key, years)
    if not seasons:
        summary["error"] = "no_seasons_matched"
        return summary

    known = existing_ss_keys() if skip_existing else set()
    ingested_at = dt.datetime.now(dt.timezone.utc).isoformat()
    batch: list[dict] = []
    budget = max_matches

    for season in seasons:
        sid = season["id"]
        syear = season.get("year")
        print(f"[sofascore] {meta['label']} season {syear} (id={sid})")
        n_season = 0
        for ev in iter_season_events(tid, sid, sleep_s=sleep_s):
            if budget is not None and budget <= 0:
                break
            mk = f"ss:{ev.get('id')}"
            if mk in known:
                summary["n_skipped_existing"] += 1
                continue
            try:
                row = row_from_event(ev, league, ingested_at, sleep_s=sleep_s)
            except Exception as e:
                summary["n_errors"] += 1
                if summary["n_errors"] <= 5:
                    print(f"  ! error event {ev.get('id')}: {e}")
                continue
            if row is None:
                summary["n_no_ref"] += 1
                continue
            batch.append(row)
            known.add(row["match_key"])
            n_season += 1
            summary["n_rows"] += 1
            if budget is not None:
                budget -= 1
            if len(batch) >= 40:
                net, total = upsert_matches(batch)
                print(f"  … flush +{net} (season so far {n_season}, csv {total})")
                batch.clear()
        summary["seasons"].append({"year": syear, "n_rows": n_season})
        if budget is not None and budget <= 0:
            break

    if batch:
        net, total = upsert_matches(batch)
        print(f"  … final flush +{net} (csv {total})")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tournament",
        choices=sorted(TOURNAMENTS),
        help="Single tournament key",
    )
    parser.add_argument(
        "--all-big5",
        action="store_true",
        help="Seed laliga + serie_a + ligue_1 + bundesliga",
    )
    parser.add_argument(
        "--years",
        default="23/24,24/25,25/26",
        help="Comma-separated Sofascore season years (default: 23/24,24/25,25/26)",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Cap finished matches processed (smoke tests)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Delay between HTTP calls (seconds); raise if Sofascore 403s",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-fetch even if match_key ss:<id> already in CSV",
    )
    args = parser.parse_args()

    if not args.tournament and not args.all_big5:
        parser.error("Pass --tournament KEY or --all-big5")

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    keys = list(TOURNAMENTS) if args.all_big5 else [args.tournament]
    totals = []
    for key in keys:
        print(f"\n=== {key} ===")
        s = seed_tournament(
            key,
            years=years,
            max_matches=args.max_matches,
            sleep_s=args.sleep,
            skip_existing=not args.no_skip_existing,
        )
        totals.append(s)
        print(
            f"[sofascore] {key}: rows={s.get('n_rows', 0)} "
            f"skip_existing={s.get('n_skipped_existing', 0)} "
            f"no_ref={s.get('n_no_ref', 0)} errors={s.get('n_errors', 0)}"
        )
        time.sleep(max(2.0, args.sleep * 10))

    catalog = rebuild_catalog()
    print(f"\n[sofascore] Catalog rebuilt: {len(catalog)} referees")
    n = sum(t.get("n_rows", 0) for t in totals)
    return 0 if n > 0 or any(t.get("n_skipped_existing", 0) for t in totals) else 1


if __name__ == "__main__":
    raise SystemExit(main())
