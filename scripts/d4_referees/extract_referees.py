"""D4 / Paso 3 — Referee assignment extractor (READ-ONLY).

Reads the day's scraped fixtures (``output/matches_<date>.json``), visits each
Flashscore match-summary page, and extracts the assigned referee from the
"Match information" block (``data-testid="wcl-summaryMatchInformation"``).

Output → ``output/referees_<date>.json``::

    { "<match_id>": {
        "home_team": str, "away_team": str, "league": str,
        "source_url": str, "ts": iso8601,
        "referee_name": str | null,
        "referee_id": str | null,
        "referee_country": str | null }, ... }

Soft-fail per match: missing referee / DOM change / navigation error does NOT
abort the run — the entry is still written with ``referee_name=null``. This
step does NOT touch the model, betting flow, justification, or UI.

Selector locked 2026-09-21 via ``probe_referee.py`` against Ligue 1 /
Bundesliga / Serie A summary pages. Structure:

    [data-testid="wcl-summaryMatchInformation"]
      div[class*="infoLabelWrapper"]  text "Referee:"
      div[class*="infoValue"]         "Letexier F." + "(Fra)"
        (no /referee/ link in current DOM → referee_id usually null)

Usage::

    python3 scripts/d4_referees/extract_referees.py [YYYY-MM-DD]
    # offline parser self-test against a cached summary HTML:
    python3 scripts/d4_referees/extract_referees.py --from-html <file.html>
"""

import asyncio
import datetime
import glob
import json
import os
import re
import sys

from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "output")

_MATCH_INFO_TESTID = "wcl-summaryMatchInformation"
_COUNTRY_RE = re.compile(r"^\((.+)\)$")


# ---------------------------------------------------------------------------
# Parsing (pure, testable offline against cached HTML)
# ---------------------------------------------------------------------------

def parse_referee(html: str) -> dict:
    """Extract referee from a match-summary page HTML.

    Returns ``{"referee_name", "referee_id", "referee_country"}`` — any field
    may be ``None`` when the Match information block is absent or has no
    Referee row (common for leagues that don't publish the assignment yet).
    """
    soup = BeautifulSoup(html, "html.parser")
    info = soup.select_one(f'[data-testid="{_MATCH_INFO_TESTID}"]')
    if info is None:
        return {"referee_name": None, "referee_id": None, "referee_country": None}

    # Children alternate label / value wrappers. Class suffixes are hashed
    # (wcl-infoValue_grawU), so match on the stable prefix.
    labels = info.select('[class*="infoLabelWrapper"]')
    values = info.select('[class*="infoValue"]')
    value_el = None
    for lab, val in zip(labels, values):
        lab_txt = lab.get_text(" ", strip=True).rstrip(":").strip().lower()
        if lab_txt == "referee":
            value_el = val
            break

    if value_el is None:
        # Fallback: text scan of the whole block ("Referee:\nName\n(Country)")
        text = info.get_text("\n", strip=True)
        m = re.search(
            r"Referee:\s*\n\s*([^\n(]+?)(?:\s*\n\s*\(([^)]+)\))?",
            text, re.IGNORECASE)
        if not m:
            return {"referee_name": None, "referee_id": None, "referee_country": None}
        name = m.group(1).strip() or None
        country = (m.group(2) or "").strip() or None
        return {"referee_name": name, "referee_id": None, "referee_country": country}

    # Prefer the bold name span; fall back to first non-country span / full text.
    name = None
    country = None
    spans = value_el.select('[data-testid="wcl-scores-simple-text-01"]')
    for sp in spans:
        t = sp.get_text(strip=True)
        if not t:
            continue
        cm = _COUNTRY_RE.match(t)
        if cm:
            country = cm.group(1).strip() or None
            continue
        if name is None:
            name = t

    if name is None:
        raw = value_el.get_text(" ", strip=True)
        # "Letexier F. (Fra)" → name + country
        cm = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", raw)
        if cm:
            name, country = cm.group(1).strip(), cm.group(2).strip()
        else:
            name = raw or None

    # Flashscore currently does not link the referee; keep the hook in case
    # a /referee/<slug>/<id>/ (or similar) appears later.
    referee_id = None
    a = value_el.select_one('a[href*="referee"]') or value_el.select_one("a[href]")
    if a is not None:
        href = a.get("href") or ""
        if "referee" in href.lower() or "/official" in href.lower():
            referee_id = href.rstrip("/").split("/")[-1] or None
            if name is None:
                name = a.get_text(strip=True) or None

    return {
        "referee_name": name,
        "referee_id": referee_id,
        "referee_country": country,
    }


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _summary_url(match: dict) -> str:
    base = (match.get("base_url") or "").rstrip("/")
    return base + "/"


def _latest_matches_file(date_str: str | None) -> str:
    if date_str:
        path = os.path.join(OUT_DIR, f"matches_{date_str}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    files = sorted(glob.glob(os.path.join(OUT_DIR, "matches_*.json")))
    if not files:
        raise FileNotFoundError("no output/matches_*.json found")
    return files[-1]


async def _accept_cookies(page):
    for sel in ("#onetrust-accept-btn-handler",
                'button:has-text("I Accept")',
                'button:has-text("Accept")'):
        try:
            btn = page.locator(sel)
            if await btn.count() > 0:
                await btn.first.click(timeout=3000)
                await page.wait_for_timeout(400)
                return
        except Exception:
            pass


async def _scrape_one(page, url):
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await _accept_cookies(page)
    for sel in (
        f'[data-testid="{_MATCH_INFO_TESTID}"]',
        '.detailScore__wrapper',
        '[class*="summary"]',
    ):
        try:
            await page.wait_for_selector(sel, timeout=8000)
            break
        except Exception:
            continue
    await page.wait_for_timeout(1500)
    html = await page.content()
    return html


async def run(date_str=None):
    from playwright.async_api import async_playwright

    matches_path = _latest_matches_file(date_str)
    date = re.search(r"matches_(\d{4}-\d{2}-\d{2})", matches_path).group(1)
    matches = json.load(open(matches_path))
    print(f"[refs] {len(matches)} fixtures from {os.path.basename(matches_path)}")

    result = {}
    n_with = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            locale="en-US",
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"))
        page = await ctx.new_page()
        for m in matches:
            mid = m.get("match_id")
            if not mid or not m.get("base_url"):
                continue
            url = _summary_url(m)
            ref = {"referee_name": None, "referee_id": None, "referee_country": None}
            try:
                html = await _scrape_one(page, url)
                ref = parse_referee(html)
            except Exception as e:
                print(f"  ! {mid} {m.get('home_team')} v {m.get('away_team')}: {e!r}")
            result[mid] = {
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "league": m.get("league"),
                "source_url": url,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "referee_name": ref.get("referee_name"),
                "referee_id": ref.get("referee_id"),
                "referee_country": ref.get("referee_country"),
            }
            if ref.get("referee_name"):
                n_with += 1
                print(f"  ✓ {mid} {m.get('home_team')} v {m.get('away_team')}: "
                      f"{ref['referee_name']}"
                      f"{' (' + ref['referee_country'] + ')' if ref.get('referee_country') else ''}")
            else:
                print(f"  · {mid} {m.get('home_team')} v {m.get('away_team')}: "
                      f"(no referee)")
        await browser.close()

    out_path = os.path.join(OUT_DIR, f"referees_{date}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    cov = (100.0 * n_with / len(result)) if result else 0.0
    print(f"\n[refs] {len(result)} matches, {n_with} with referee "
          f"({cov:.0f}% coverage) → {out_path}")


def _selftest(html_path):
    """Offline parser check against a cached summary HTML dump."""
    html = open(html_path).read()
    out = parse_referee(html)
    print(f"=== parse_referee({os.path.basename(html_path)}) ===")
    for k, v in out.items():
        print(f"  {k}: {v!r}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--from-html":
        _selftest(args[1])
    else:
        date_arg = args[0] if args and re.match(r"\d{4}-\d{2}-\d{2}", args[0]) else None
        asyncio.run(run(date_arg))
