"""Winamax cards O/U odds (Número total de tarjetas, line 3.5).

Flashscore / BetExplorer / OddsPortal do not publish cards O/U; Pamestoixima
is Akamai-blocked for headless Playwright. Winamax.es embeds a full
``PRELOADED_STATE`` JSON in each match page — plain HTTPS, no browser.

Market note: Winamax counts yellow=1, red=2 (segunda amarilla ignored). Our
model target is HY+AY yellows. Still the only stable real-odds source found
(probe 2026-09-21); coverage is league-dependent (often present in LATAM /
lower tiers, sparse on far-out Big-5).

Usage:
    python3 -m ml_project.cards.winamax_odds 2026-09-21
    python3 -m ml_project.cards.winamax_odds --probe
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional
from zoneinfo import ZoneInfo

from rapidfuzz import fuzz

from .constants import CARD_LINE

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WINAMAX_SPORTS_URL = "https://www.winamax.es/apuestas-deportivas/sports/1"
WINAMAX_MATCH_URL = "https://www.winamax.es/apuestas-deportivas/match/{mid}"
ODDS_SOURCE = "winamax"
# betType for "Número total de tarjetas" (full-time booking points).
_BET_TYPE_TOTAL_CARDS = 2603
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_TZ = ZoneInfo("Europe/Madrid")
_MIN_NAME_SCORE = 80.0
_LINE_KEY = f"total={CARD_LINE:g}"  # "total=3.5"


def _fetch(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_preloaded_state(html: str) -> Optional[dict]:
    """Extract ``PRELOADED_STATE = {...}`` from a Winamax HTML page."""
    start = html.find("PRELOADED_STATE")
    if start < 0:
        return None
    eq = html.find("{", start)
    if eq < 0:
        return None
    depth = 0
    for i, ch in enumerate(html[eq:], eq):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[eq : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_cards_ou_3_5(state: dict) -> tuple[Optional[float], Optional[float]]:
    """Return (over, under) decimal odds for total cards line CARD_LINE."""
    bets = state.get("bets") or {}
    outcomes = state.get("outcomes") or {}
    odds = state.get("odds") or {}
    for bet in bets.values():
        if bet.get("betType") != _BET_TYPE_TOTAL_CARDS:
            continue
        if bet.get("specialBetValue") != _LINE_KEY:
            continue
        over = under = None
        for oid in bet.get("outcomes") or []:
            key = str(oid)
            oc = outcomes.get(key) or {}
            raw = odds.get(key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            code = (oc.get("code") or "").lower()
            if code == "over":
                over = val
            elif code == "under":
                under = val
        if over and under and over > 1.0 and under > 1.0:
            return over, under
    return None, None


def _norm_team(name: str) -> str:
    s = (name or "").lower().strip()
    s = re.sub(r"\s*\(f\)\s*$", "", s)
    s = s.replace("á", "a").replace("é", "e").replace("í", "i")
    s = s.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    s = s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_womens(name: str) -> bool:
    return bool(re.search(r"\(f\)|\bfemenin|\bwomen\b|\bw\b", name or "", re.I))


def load_football_index(state: Optional[dict] = None) -> list[dict]:
    """List of Winamax football match dicts (from sports/1 PRELOADED_STATE)."""
    if state is None:
        state = parse_preloaded_state(_fetch(WINAMAX_SPORTS_URL))
    if not state:
        return []
    out = []
    for mid, m in (state.get("matches") or {}).items():
        if m.get("sportId") != 1:
            continue
        ts = m.get("matchStart")
        if not ts:
            continue
        kick = dt.datetime.fromtimestamp(int(ts), tz=_TZ)
        out.append({
            "match_id": int(m.get("matchId") or mid),
            "home": m.get("competitor1Name") or "",
            "away": m.get("competitor2Name") or "",
            "kickoff": kick,
            "date": kick.date(),
            "status": m.get("status") or "",
            "tournament_id": m.get("tournamentId"),
            "womens": _is_womens(m.get("competitor1Name") or "")
                      or _is_womens(m.get("competitor2Name") or ""),
        })
    return out


def _parse_fs_date(start_time: str, fallback: dt.date) -> dt.date:
    """Flashscore ``DD.MM.YYYY HH:MM`` → date (Europe/Madrid calendar day)."""
    if not start_time:
        return fallback
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", start_time.strip())
    if not m:
        return fallback
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return fallback


def match_flashscore_row(
    home: str,
    away: str,
    match_date: dt.date,
    index: list[dict],
    min_score: float = _MIN_NAME_SCORE,
) -> Optional[tuple[dict, float]]:
    """Best Winamax match for a Flashscore fixture (same day ±1, name score)."""
    fs_w = _is_womens(home) or _is_womens(away)
    best: Optional[dict] = None
    best_s = 0.0
    for w in index:
        day_delta = abs((w["date"] - match_date).days)
        if day_delta > 1:
            continue
        if w["womens"] != fs_w:
            continue
        s_fwd = (
            fuzz.token_set_ratio(_norm_team(home), _norm_team(w["home"]))
            + fuzz.token_set_ratio(_norm_team(away), _norm_team(w["away"]))
        ) / 2.0
        s_rev = (
            fuzz.token_set_ratio(_norm_team(home), _norm_team(w["away"]))
            + fuzz.token_set_ratio(_norm_team(away), _norm_team(w["home"]))
        ) / 2.0
        s = max(s_fwd, s_rev * 0.85)
        if day_delta == 1:
            s -= 5.0
        if s > best_s:
            best_s, best = s, w
    if best is None or best_s < min_score:
        return None
    return best, best_s


def fetch_match_cards_odds(winamax_match_id: int) -> tuple[Optional[float], Optional[float]]:
    html = _fetch(WINAMAX_MATCH_URL.format(mid=winamax_match_id))
    state = parse_preloaded_state(html)
    if not state:
        return None, None
    return extract_cards_ou_3_5(state)


def enrich_matches_file(
    date: str,
    matches_path: Optional[str] = None,
    sleep_s: float = 0.15,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fill ``over_cards_3_5`` / ``under_cards_3_5`` on matches_<date>.json.

    Only writes when Winamax returns both sides > 1.0. Existing Flashscore
    values are kept unless ``overwrite`` is True.
    """
    path = matches_path or os.path.join(_ROOT, "output", f"matches_{date}.json")
    summary: dict[str, Any] = {
        "date": date,
        "path": path,
        "n_matches": 0,
        "n_linked": 0,
        "n_with_odds": 0,
        "n_skipped_existing": 0,
        "n_no_market": 0,
        "n_unmatched": 0,
        "rows": [],
    }
    if not os.path.isfile(path):
        summary["error"] = "missing_matches_file"
        return summary
    with open(path) as f:
        matches = json.load(f)
    if not isinstance(matches, list):
        summary["error"] = "matches_not_list"
        return summary
    summary["n_matches"] = len(matches)
    if not matches:
        return summary

    try:
        index = load_football_index()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        summary["error"] = f"winamax_index_failed:{e}"
        return summary

    fallback_date = dt.date.fromisoformat(date)
    changed = False
    for row in matches:
        home = row.get("home_team") or ""
        away = row.get("away_team") or ""
        mdate = _parse_fs_date(row.get("start_time") or "", fallback_date)
        existing_o = row.get("over_cards_3_5")
        existing_u = row.get("under_cards_3_5")
        if (
            not overwrite
            and existing_o not in (None, "", "null")
            and existing_u not in (None, "", "null")
        ):
            summary["n_skipped_existing"] += 1
            continue

        linked = match_flashscore_row(home, away, mdate, index)
        if not linked:
            summary["n_unmatched"] += 1
            continue
        wmatch, score = linked
        summary["n_linked"] += 1
        try:
            time.sleep(sleep_s)
            over, under = fetch_match_cards_odds(wmatch["match_id"])
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            summary["rows"].append({
                "fs": f"{home} vs {away}",
                "error": str(e),
            })
            continue
        if over is None or under is None:
            summary["n_no_market"] += 1
            summary["rows"].append({
                "fs": f"{home} vs {away}",
                "winamax_id": wmatch["match_id"],
                "score": round(score, 1),
                "market": False,
            })
            continue

        row["over_cards_3_5"] = f"{over:.2f}"
        row["under_cards_3_5"] = f"{under:.2f}"
        row["cards_odds_source"] = ODDS_SOURCE
        row["winamax_match_id"] = wmatch["match_id"]
        summary["n_with_odds"] += 1
        changed = True
        summary["rows"].append({
            "fs": f"{home} vs {away}",
            "winamax": f"{wmatch['home']} vs {wmatch['away']}",
            "winamax_id": wmatch["match_id"],
            "score": round(score, 1),
            "over": over,
            "under": under,
        })

    if changed:
        with open(path, "w") as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return summary


def probe(limit: int = 30, sleep_s: float = 0.12) -> dict[str, Any]:
    """Scan upcoming Winamax football for cards O/U 3.5; gate ≥2 hits."""
    index = load_football_index()
    now = dt.datetime.now(tz=_TZ)
    upcoming = [
        m for m in index
        if m["status"] == "PREMATCH"
        and 0 <= (m["kickoff"] - now).total_seconds() <= 72 * 3600
    ]
    upcoming.sort(key=lambda m: m["kickoff"])
    hits = []
    checked = 0
    for m in upcoming[:limit]:
        checked += 1
        try:
            time.sleep(sleep_s)
            over, under = fetch_match_cards_odds(m["match_id"])
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        if over and under:
            hits.append({
                "match_id": m["match_id"],
                "home": m["home"],
                "away": m["away"],
                "kickoff": m["kickoff"].isoformat(),
                "over": over,
                "under": under,
            })
    return {
        "checked": checked,
        "hits": hits,
        "n_hits": len(hits),
        "pass": len(hits) >= 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "date", nargs="?",
        help="YYYY-MM-DD — enrich output/matches_<date>.json",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Scan upcoming Winamax football for cards O/U 3.5 (gate ≥2)",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--matches", default=None, help="Override matches JSON path")
    args = parser.parse_args()

    if args.probe or not args.date:
        result = probe(limit=args.limit)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(
            f"[winamax-cards] probe: {result['n_hits']}/{result['checked']} "
            f"with O/U {CARD_LINE} → {'PASS' if result['pass'] else 'FAIL'}"
        )
        return 0 if result["pass"] else 1

    summary = enrich_matches_file(
        args.date, matches_path=args.matches, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(
        f"[winamax-cards] {summary.get('n_with_odds', 0)} odds written "
        f"(linked={summary.get('n_linked', 0)}, "
        f"unmatched={summary.get('n_unmatched', 0)}, "
        f"no_market={summary.get('n_no_market', 0)})"
    )
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
