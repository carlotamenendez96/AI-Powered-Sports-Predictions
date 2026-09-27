"""Euroleague / EuroCup odds probe — Flashscore, moneyline + totals ladder.

Writes ``output_euroleague/euroleague_odds_<date>.json``, the file
``predict_euroleague.py`` and the ``/euroleague/auto_wager`` route have been
waiting on since Phase 3. Until it exists, P(Over) is blank on every row and
every slip comes back empty.

WHY A LADDER, NOT A LINE
------------------------
Football's O/U market is a fixed 2.5, so "the line" is implicit. Basketball
totals are not: each bookmaker posts its OWN line and its own pair of prices,
and Flashscore lists every one of them — 17-18 rows spanning ~162.5 to ~173.5
on a typical EuroLeague game, with three different books sometimes sitting on
the same 169.5 at different prices. A bet of 1.48 on 162.5 and a bet of 1.91
on 169.5 are completely different wagers, so **line and price must always be
taken from the same row**; mixing one book's line with another's price records
a bet nobody offered. Every row here therefore carries its own ``book``.

We persist the FULL ladder rather than one chosen line, because the choice is
a strategy decision that we want to be able to change and — more importantly —
to measure, without re-scraping. See the caveat under `main_line` below.

MARKET / SETTLEMENT BASIS
-------------------------
``ft-including-ot``, not ``full-time``. Flashscore exposes both; basketball
bets settle including overtime, and picking the wrong tab would mis-settle
every total in a game that goes to OT.

NAVIGATION
----------
The odds URL is PATH-based (``…/odds/over-under/ft-including-ot/?mid=…``), so
it can be fetched directly — unlike the league standings pages, where the
window is in a hash fragment the SPA ignores. The path is discovered by
reading the match page's own odds link rather than being constructed, since
the slug contains team-name fragments.

Usage:
    python3 ml_project/euroleague/fetch_euroleague_odds.py [--date YYYY-MM-DD]
                                                           [--competition E|U|both]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright
from rapidfuzz import fuzz

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO, "data_sets", "Euroleague")
OUT_DIR = os.path.join(_REPO, "output_euroleague")

FIXTURE_URLS = {
    "E": "https://www.flashscore.com/basketball/europe/euroleague/fixtures/",
    "U": "https://www.flashscore.com/basketball/europe/eurocup/fixtures/",
}
COMPETITION_NAMES = {"E": "Euroleague", "U": "EuroCup"}

# Tie-break order when several books offer an equivalent price/line. Operator's
# call (2026-09-27): bwin first. Normalised comparison, so "Bwin.gr" matches.
BOOK_PREFERENCE = ["bwin", "bet365", "stoiximan", "pamestoixima", "novibet"]

SLEEP_S = 1.0          # same politeness budget as the other Flashscore paths

# Fixture join gates. The ABSOLUTE score is a weak signal here — Flashscore's
# short names and the API's sponsor-laden ones disagree a lot, and the worst
# true pair measured (2026-09-29) was "Venezia v Frankfurt" ->
# "UMANA REYER VENICE v SKYLINERS FRANKFURT" at just 66, because Venezia and
# Venice are different words. The MARGIN over the runner-up is what actually
# separates right from wrong: across all 13 pairs that day the correct fixture
# won by 37-72 points and no runner-up ever scored above 46. So accept on a
# modest floor plus a clear margin, which joins Venezia without dropping the
# threshold low enough to invent matches.
NAME_MATCH_MIN = 60
NAME_MATCH_MIN_MARGIN = 20


def _norm_book(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower()).replace("gr", "", 1) \
        if (name or "").lower().endswith(".gr") else re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _book_rank(book: str) -> int:
    n = _norm_book(book)
    for i, pref in enumerate(BOOK_PREFERENCE):
        if pref in n:
            return i
    return len(BOOK_PREFERENCE)


def _num(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _fixture_rows(page, comp: str, date_str: str):
    """[(match_id, flashscore_home, flashscore_away)] for the target date.

    The fixtures list renders dates as "29.09. 19:00" — day.month, no year —
    so the target is matched on day+month.
    """
    page.goto(FIXTURE_URLS[comp], wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("[id^='g_3_']", timeout=20000)
    except Exception as e:
        print(f"  ! {COMPETITION_NAMES[comp]}: fixtures page did not render ({e})")
        return None                                   # None = failure, [] = empty day
    rows = page.eval_on_selector_all("[id^='g_3_']", """els => els.map(e => ({
        id: e.id.replace('g_3_',''),
        text: e.innerText.replace(/\\n/g,'|')
    }))""")
    d = datetime.date.fromisoformat(date_str)
    want = f"{d.day:02d}.{d.month:02d}."
    out = []
    for r in rows:
        parts = [p.strip() for p in r["text"].split("|") if p.strip()]
        if not parts or not parts[0].startswith(want):
            continue
        if len(parts) >= 3:
            out.append((r["id"], parts[1], parts[2]))
    return out


def _odds_base(page, match_id: str):
    """The match's own /odds/ path (slug contains team fragments, so read it)."""
    page.goto(f"https://www.flashscore.com/match/{match_id}/",
              wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2000)
    link = page.locator("a[href*='/odds/']")
    if not link.count():
        return None
    href = link.first.get_attribute("href")
    return "https://www.flashscore.com" + href.split("/odds/")[0] + "/odds/"


# Plausibility bounds. These are not cosmetic: when a match's odds tab fails to
# render, the SPA leaves the LEAGUE STANDINGS table on the page and it matches
# `.ui-table__row` just as happily — observed 2026-09-30 on Lietkabelis v KK
# Bosna, which yielded 8 "totals" whose book was a team name ("Ulm", "PAOK"),
# whose line was the league rank (1, 2, 3…) and whose prices were 0.0. Without
# these checks that garbage lands in the ladder the strategy reads.
MIN_DECIMAL_ODDS = 1.01     # a decimal price is never <= 1
MAX_DECIMAL_ODDS = 1000.0
MIN_TOTAL_LINE = 100.0      # basketball totals sit ~120-230; a rank never does
MAX_TOTAL_LINE = 280.0


def _valid_ml(n) -> bool:
    return all(MIN_DECIMAL_ODDS <= v <= MAX_DECIMAL_ODDS for v in n)


def _valid_total(n) -> bool:
    return (MIN_TOTAL_LINE <= n[0] <= MAX_TOTAL_LINE
            and all(MIN_DECIMAL_ODDS <= v <= MAX_DECIMAL_ODDS for v in n[1:]))


def _scrape_market(page, url: str, n_cols: int, validator=None):
    """Rows of (book, *numbers) from one odds tab. n_cols = numbers per row.

    `validator` rejects rows that parse numerically but cannot be odds (see the
    bounds above); rejected rows are counted and reported, never stored.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(3000)
    try:
        page.wait_for_selector(".ui-table__row", timeout=12000)
    except Exception:
        return []
    raw = page.eval_on_selector_all(".ui-table__row", """els => els.map(e => ({
        book: (e.querySelector('img') ? (e.querySelector('img').getAttribute('title')
               || e.querySelector('img').getAttribute('alt')) : null)
              || (e.querySelector('a') ? e.querySelector('a').getAttribute('title') : null) || '',
        text: e.innerText.replace(/\\n/g,'|')
    }))""")
    out, rejected = [], 0
    for r in raw:
        nums = [_num(p) for p in r["text"].split("|")]
        nums = [n for n in nums if n is not None]
        if len(nums) < n_cols or not r["book"]:
            continue
        vals = nums[:n_cols]
        if validator and not validator(vals):
            rejected += 1
            continue
        out.append((r["book"].strip(), vals))
    if rejected:
        print(f"      (discarded {rejected} implausible row(s) — page was probably "
              f"not the odds table)")
    return out


def main_line(totals: list) -> dict | None:
    """The consensus total: the row whose over/under prices are most balanced.

    Recorded alongside whatever the strategy actually bets so the two can be
    compared on settled money later. Ladder-shopping maximises EV over ~17
    noisy rows, and this project has already measured that shape of selection
    picking out the model's own error rather than real edge (CLAUDE.md, "No
    measured edge over the market"); this column is how we find out whether
    that repeats for basketball totals, cheaply and without betting on it.
    """
    scored = [t for t in totals if t["over"] and t["under"]]
    if not scored:
        return None
    return sorted(scored, key=lambda t: (abs(t["over"] - t["under"]), _book_rank(t["book"])))[0]


def _preferred(rows: list, keys: tuple):
    """First row by book preference that has all `keys` populated."""
    ok = [r for r in rows if all(r.get(k) for k in keys)]
    return sorted(ok, key=lambda r: _book_rank(r["book"]))[0] if ok else None


def _join_fixture(home: str, away: str, fixtures: list):
    """Flashscore short names -> our fixture record.

    Accepts on score floor AND margin over the runner-up (see the constants).
    Returns (fixture_or_None, score, margin).
    """
    scored = sorted(
        ((fuzz.token_set_ratio(home.lower(), (fx.get("home_team") or "").lower())
          + fuzz.token_set_ratio(away.lower(), (fx.get("away_team") or "").lower())) / 2, i)
        for i, fx in enumerate(fixtures))
    if not scored:
        return None, 0.0, 0.0
    best_score, best_i = scored[-1]
    margin = best_score - (scored[-2][0] if len(scored) > 1 else 0.0)
    if best_score >= NAME_MATCH_MIN and margin >= NAME_MATCH_MIN_MARGIN:
        return fixtures[best_i], best_score, margin
    return None, best_score, margin


def fetch(date_str: str, comps: list) -> int:
    fx_path = os.path.join(DATA_DIR, f"fixtures_{date_str}.json")
    fixtures = json.load(open(fx_path)) if os.path.exists(fx_path) else []
    print(f"[odds] {date_str} — {len(fixtures)} known fixture(s) from {os.path.basename(fx_path)}")
    if not fixtures:
        print("[odds] no fixtures file; run fetch_euroleague_daily.py fixtures first.")

    records, failed = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for comp in comps:
            rows = _fixture_rows(page, comp, date_str)
            if rows is None:
                failed.append(comp)
                continue
            print(f"  {COMPETITION_NAMES[comp]}: {len(rows)} match(es) on {date_str}")
            for match_id, fs_home, fs_away in rows:
                time.sleep(SLEEP_S)
                base = _odds_base(page, match_id)
                if not base:
                    print(f"    - {fs_home} v {fs_away}: no odds tab")
                    continue
                ml = [{"book": b, "home": n[0], "away": n[1]}
                      for b, n in _scrape_market(page, f"{base}home-away/ft-including-ot/?mid={match_id}", 2, _valid_ml)]
                time.sleep(SLEEP_S)
                tot = [{"book": b, "line": n[0], "over": n[1], "under": n[2]}
                       for b, n in _scrape_market(page, f"{base}over-under/ft-including-ot/?mid={match_id}", 3, _valid_total)]

                fx, score, margin = _join_fixture(
                    fs_home, fs_away, [f for f in fixtures if f.get("competition") == comp])
                pref_ml = _preferred(ml, ("home", "away"))
                pref_tot = main_line(tot)
                rec = {
                    "gameId": (fx or {}).get("gameId"),
                    "competition": comp,
                    "date": date_str,
                    # Canonical names when the join worked, so the downstream
                    # (home_team, away_team) key matches predictions_*.csv.
                    "home_team": (fx or {}).get("home_team") or fs_home,
                    "away_team": (fx or {}).get("away_team") or fs_away,
                    "flashscore_home": fs_home,
                    "flashscore_away": fs_away,
                    "flashscore_match_id": match_id,
                    "name_match_score": round(score, 1),
                    "name_match_margin": round(margin, 1),
                    "moneyline": ml,
                    "totals": tot,
                    # Back-compatible scalars for the existing consumers, always
                    # taken as a matched (line, over, under) triple from ONE row.
                    "home_ml_decimal": (pref_ml or {}).get("home"),
                    "away_ml_decimal": (pref_ml or {}).get("away"),
                    "ml_book": (pref_ml or {}).get("book"),
                    "total": (pref_tot or {}).get("line"),
                    "over_ml_decimal": (pref_tot or {}).get("over"),
                    "under_ml_decimal": (pref_tot or {}).get("under"),
                    "total_book": (pref_tot or {}).get("book"),
                    "scraped_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                records.append(rec)
                flag = "" if fx else f"  [UNJOINED score={score:.0f} margin={margin:.0f}]"
                print(f"    {fs_home} v {fs_away}: {len(ml)} ML, {len(tot)} totals"
                      f" | main {rec['total']} {rec['over_ml_decimal']}/{rec['under_ml_decimal']}"
                      f" ({rec['total_book']}){flag}")
        browser.close()

    if failed and len(failed) == len(comps):
        print(f"[odds] EVERY competition failed ({', '.join(failed)}) — broken feed, "
              f"not an empty slate. No file written.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"euroleague_odds_{date_str}.json")

    # MERGE, never plain-overwrite: a --competition E run followed by a
    # --competition U run would otherwise leave the file holding only EuroCup,
    # silently dropping the EuroLeague odds the first run just collected
    # (observed 2026-09-27: 13 records became 5). Keyed on the Flashscore match
    # id, so re-running the same competition refreshes its rows in place.
    if os.path.exists(out_path):
        try:
            prior = json.load(open(out_path)) or []
        except (json.JSONDecodeError, OSError):
            prior = []
        fresh = {r["flashscore_match_id"] for r in records}
        kept = [r for r in prior if r.get("flashscore_match_id") not in fresh]
        if kept:
            print(f"[odds] merging with {len(kept)} record(s) already on file "
                  f"for {date_str} (other competition / earlier run)")
        records = kept + records

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    joined = sum(1 for r in records if r["gameId"])
    print(f"[odds] wrote {len(records)} record(s) ({joined} joined to a fixture) → {out_path}")
    if failed:
        print(f"[odds] NOTE: {', '.join(COMPETITION_NAMES[c] for c in failed)} unavailable this run.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: tomorrow)")
    ap.add_argument("--competition", default="both", choices=["E", "U", "both"])
    a = ap.parse_args()
    date_str = a.date or (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    comps = ["E", "U"] if a.competition == "both" else [a.competition]
    return fetch(date_str, comps)


if __name__ == "__main__":
    sys.exit(main())
