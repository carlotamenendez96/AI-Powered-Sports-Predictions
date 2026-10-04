#!/usr/bin/env python3
"""Probe Flashscore for yellow-cards Over/Under 3.5 odds (READ-ONLY).

Locks the URL path + row selector before wiring them into
``flashscore_spider.py``. Same spirit as ``scripts/d4_referees/probe_referee.py``.

Candidate paths (tried in order per match)::

    {base}/odds/over-under/number-of-cards/full-time/?mid={id}
    {base}/odds/over-under/yellow-cards/full-time/?mid={id}
    {base}/odds/over-under/total-cards/full-time/?mid={id}
    hash fallbacks under #/odds-comparison/...

Looks for a ``.ui-table__row`` whose text contains the line ``3.5`` and at
least two decimal odds (Over | Under), mirroring the O/U 2.5 goals scrape.

Usage::

    python3 scripts/cards/probe_cards_odds.py --from-matches 2026-09-20
    python3 scripts/cards/probe_cards_odds.py --mid ClKl2gCt --base-url \\
        'https://www.flashscore.com/match/football/ac-milan-8Sa8HInO/lecce-G8lYsMgU/'

Gate: ≥2 matches with Over+Under > 1.0 on line 3.5 before treating the
market as live on Flashscore.

Locked 2026-09-21 (probe on Big-5 finished + Serie A upcoming, e.g.
Fiorentina–Genoa ``6k8N36mo``, Inter–Parma ``bqe9R9uh``)::

    # Flashscore odds-comparison does NOT currently publish Number of Cards /
    # Yellow Cards O/U. Candidate clean paths return empty/404; GraphQL
    # ``oce`` bettingTypes are only the standard 1X2/O/U-goals/AH/… set
    # (no CARD* type). Same DOM contract as goals O/U is kept for when/if
    # the market appears — spider attempt is non-fatal and leaves fields empty.
    URL_TEMPLATE = "{base}/odds/over-under/number-of-cards/full-time/?mid={mid}"
    ROW_SELECTOR = ".ui-table__row"
    LINE = "3.5"
    LINE_RE = r"(?:^|[^\\d])3\\.5(?:[^\\d]|$)"
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "output", "d4_probe")
MATCHES_DIR = os.path.join(ROOT, "output")

LINE = "3.5"
_LINE_RE = re.compile(r"(?:^|[^\d])3\.5(?:[^\d]|$)")

# Path suffixes tried after `{base}/odds/over-under/` …
_PATH_SUFFIXES = (
    "number-of-cards/full-time/",
    "yellow-cards/full-time/",
    "total-cards/full-time/",
    "cards/full-time/",
)

# Hash fallbacks (SPA routes) when clean paths fail.
_HASH_SUFFIXES = (
    "odds-comparison/over-under-odds/number-of-cards/full-time",
    "odds-comparison/over-under/number-of-cards/full-time",
    "odds-comparison/over-under-odds/yellow-cards/full-time",
)


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


def _norm_base(base_url: str) -> str:
    b = (base_url or "").split("#")[0].split("?")[0].rstrip("/")
    return b


def candidate_urls(base_url: str, match_id: str) -> list[tuple[str, str]]:
    """Return [(label, url), ...] to try."""
    base = _norm_base(base_url)
    out = []
    if base:
        for suf in _PATH_SUFFIXES:
            out.append((suf, f"{base}/odds/over-under/{suf}?mid={match_id}"))
        # Also try without mid query (some FS pages ignore it when path has slug)
        for suf in _PATH_SUFFIXES[:2]:
            out.append((f"{suf}(no-mid)", f"{base}/odds/over-under/{suf}"))
    # Hash routes off match id only
    for h in _HASH_SUFFIXES:
        out.append((f"hash:{h}", f"https://www.flashscore.com/match/{match_id}/#/{h}"))
    return out


def parse_line_35_from_rows(row_texts: list[str]) -> dict | None:
    """Find 3.5 line in row texts; return {over, under, row_text} or None."""
    for text in row_texts:
        clean = text.replace("\n", " ").strip()
        if not _LINE_RE.search(clean):
            continue
        nums = re.findall(r"\d+\.\d+", clean)
        # Expect line 3.5 + over + under, or just over+under if line is non-numeric text
        if len(nums) >= 3:
            # First float is usually the line itself
            over, under = nums[1], nums[2]
            try:
                if float(over) > 1.0 and float(under) > 1.0:
                    return {"over": over, "under": under, "row_text": clean[:120],
                            "all_nums": nums}
            except ValueError:
                continue
        elif len(nums) == 2:
            over, under = nums[0], nums[1]
            try:
                if float(over) > 1.0 and float(under) > 1.0:
                    return {"over": over, "under": under, "row_text": clean[:120],
                            "all_nums": nums}
            except ValueError:
                continue
    return None


_JS_ROWS = r"""
() => {
  const rows = Array.from(document.querySelectorAll('.ui-table__row, [class*="oddsRow"], [class*="ui-table"] tr'));
  const texts = rows.map(r => (r.innerText || '').trim()).filter(t => t.length > 0);
  // Also collect nav / tab hints that mention cards
  const links = Array.from(document.querySelectorAll('a[href*="odds"], a[href*="card"], button, [class*="tab"]'))
    .map(a => ({
      href: a.getAttribute('href') || '',
      text: (a.innerText || '').trim().slice(0, 80),
    }))
    .filter(x => /card|booking|amarill/i.test(x.text + x.href))
    .slice(0, 40);
  return {
    n_rows: rows.length,
    row_texts: texts.slice(0, 80),
    card_link_hints: links,
    title: document.title || '',
    url: location.href,
  };
}
"""


async def try_url(page, label: str, url: str) -> dict:
    result = {
        "label": label,
        "url": url,
        "ok": False,
        "error": None,
        "parsed": None,
        "n_rows": 0,
        "card_link_hints": [],
        "sample_rows": [],
    }
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await _accept_cookies(page)
        try:
            await page.wait_for_selector(".ui-table, [class*='oddsValue'], [class*='ui-table']",
                                        timeout=8000)
        except Exception:
            pass
        await page.wait_for_timeout(1500)
        data = await page.evaluate(_JS_ROWS)
        result["n_rows"] = data.get("n_rows", 0)
        result["card_link_hints"] = data.get("card_link_hints", [])
        result["sample_rows"] = (data.get("row_texts") or [])[:12]
        result["final_url"] = data.get("url")
        result["title"] = data.get("title")
        parsed = parse_line_35_from_rows(data.get("row_texts") or [])
        if parsed:
            result["ok"] = True
            result["parsed"] = parsed
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


async def probe_match(mid: str, base_url: str, page) -> dict:
    report = {
        "match_id": mid,
        "base_url": base_url,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempts": [],
        "winner": None,
    }
    for label, url in candidate_urls(base_url, mid):
        print(f"  try [{label}] {url}")
        att = await try_url(page, label, url)
        report["attempts"].append(att)
        if att["ok"]:
            report["winner"] = {
                "label": label,
                "url": att.get("final_url") or url,
                "over": att["parsed"]["over"],
                "under": att["parsed"]["under"],
                "row_text": att["parsed"]["row_text"],
            }
            print(f"  ✓ 3.5 Over={att['parsed']['over']} Under={att['parsed']['under']} "
                  f"via {label}")
            break
        hints = att.get("card_link_hints") or []
        if hints:
            print(f"    (no 3.5 row; {len(hints)} card-ish link hints)")
    if not report["winner"]:
        print(f"  ✗ no 3.5 cards line for {mid}")
        # Dump first attempt's sample rows for debugging
        for att in report["attempts"][:3]:
            if att.get("sample_rows"):
                print(f"    sample rows from {att['label']}:")
                for r in att["sample_rows"][:6]:
                    print(f"      | {r[:100]!r}")
                break
    return report


def load_from_matches(date: str, limit: int = 5) -> list[tuple[str, str]]:
    path = os.path.join(MATCHES_DIR, f"matches_{date}.json")
    with open(path) as f:
        matches = json.load(f)
    prefer = ("Ligue 1", "Bundesliga", "Serie A", "LaLiga", "La Liga",
              "Premier", "Liga Portugal", "Eredivisie", "Championship")
    picked = []
    for m in matches:
        lg = m.get("league") or ""
        if not any(p in lg for p in prefer):
            continue
        mid = m.get("match_id")
        base = m.get("base_url") or ""
        if mid and base:
            picked.append((mid, base, lg, m.get("home_team")))
        if len(picked) >= limit:
            break
    # fallback: any with base_url
    if len(picked) < 2:
        for m in matches:
            mid, base = m.get("match_id"), m.get("base_url") or ""
            if mid and base and (mid, base) not in [(p[0], p[1]) for p in picked]:
                picked.append((mid, base, m.get("league"), m.get("home_team")))
            if len(picked) >= limit:
                break
    return picked


async def main_async(args) -> int:
    from playwright.async_api import async_playwright

    targets: list[tuple[str, str, str, str]] = []
    if args.from_matches:
        for mid, base, lg, home in load_from_matches(args.from_matches, limit=args.limit):
            targets.append((mid, base, lg or "", home or ""))
            print(f"[probe] slate {args.from_matches}: {mid} {lg} {home}")
    if args.mid:
        if not args.base_url:
            print("[-] --mid requires --base-url", file=sys.stderr)
            return 1
        targets.append((args.mid, args.base_url, "", ""))

    if not targets:
        print("[-] No targets. Use --from-matches YYYY-MM-DD or --mid + --base-url",
              file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    reports = []
    n_ok = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
            locale="en-US",
        )
        page = await context.new_page()
        for mid, base, lg, home in targets:
            print(f"\n=== {mid} ({lg} / {home}) ===")
            rep = await probe_match(mid, base, page)
            reports.append(rep)
            if rep.get("winner"):
                n_ok += 1
            out_path = os.path.join(OUT_DIR, f"cards_odds_{mid}.json")
            with open(out_path, "w") as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)
            print(f"  wrote {out_path}")
        await browser.close()

    summary = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_targets": len(targets),
        "n_ok": n_ok,
        "gate_pass": n_ok >= 2,
        "winners": [
            {"match_id": r["match_id"], **r["winner"]}
            for r in reports if r.get("winner")
        ],
    }
    sum_path = os.path.join(OUT_DIR, "cards_odds_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY: {n_ok}/{len(targets)} with 3.5 line "
          f"(gate ≥2: {'PASS' if summary['gate_pass'] else 'FAIL'}) ===")
    print(f"wrote {sum_path}")
    for w in summary["winners"]:
        print(f"  {w['match_id']}: Over {w['over']} / Under {w['under']}  [{w['label']}]")
        print(f"    URL: {w['url']}")
    return 0 if summary["gate_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-matches", metavar="YYYY-MM-DD",
                        help="Pull Big-5-ish fixtures from output/matches_<date>.json")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--mid", help="Single match_id")
    parser.add_argument("--base-url", help="Flashscore match base URL (with --mid)")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
