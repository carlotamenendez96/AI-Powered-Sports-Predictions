#!/usr/bin/env python3
"""Probe Pamestoixima for Total Yellow Cards / Cards O/U 3.5 (READ-ONLY).

Uses the existing headed BrowserSession (headless is Akamai-blocked —
see real_betting/PAMESTOIXIMA_NOTES.md). No stake clicks, no slip writes.

Strategy:
  1. Open next24hCoupon (known-good football listing).
  2. Collect a few fixture URLs.
  3. Open each match page, scroll markets, list accordion labels.
  4. If a Cards / Yellow Cards / Κάρτες market exists, expand it and
     parse Over/Under 3.5 odds the same way O/U goals is scraped.

Usage:
    python3 scripts/cards/probe_cards_odds_pamestoixima.py
    python3 scripts/cards/probe_cards_odds_pamestoixima.py --limit 3
    python3 scripts/cards/probe_cards_odds_pamestoixima.py \\
        --url 'https://www.pamestoixima.gr/en/football/.../12345678'

Gate: ≥2 matches with Over+Under > 1.0 on line 3.5.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "output", "d4_probe")
LINE = "3.5"

# Market accordion labels we accept (EN + GR).
_CARD_LABEL_RE = re.compile(
    r"(yellow\s*cards?|total\s*cards?|number\s*of\s*cards?|"
    r"cards?\s*over/?under|booking|"
    r"κίτριν|κάρτ)",
    re.I,
)

# Class fragments that might appear on market-box-root for cards.
_CARD_CLASS_HINTS = (
    "YELLOW_CARD", "TOTAL_CARD", "CARDS_OVER", "CARD_OVER",
    "BOOKING", "KITRIN", "KART",
)


def _accept_cookies(page):
    for sel in (
        '#onetrust-accept-btn-handler',
        'button:has-text("Accept")',
        'button:has-text("I Accept")',
        'button:has-text("Συμφωνώ")',
        'button:has-text("Αποδοχή")',
    ):
        try:
            loc = page.locator(sel)
            if loc.count():
                loc.first.click(timeout=2500)
                time.sleep(0.4)
                return
        except Exception:
            pass


def _scroll_markets(page, passes: int = 8):
    for _ in range(passes):
        page.evaluate("""() => {
            const el = document.scrollingElement || document.body;
            el.scrollBy(0, Math.floor(window.innerHeight * 0.85));
        }""")
        time.sleep(0.45)


def _list_market_labels(page) -> list[str]:
    return page.evaluate("""() => {
        const out = [];
        const seen = new Set();
        const els = document.querySelectorAll(
          'button.event-page-market-box-collapseBtn, '
        + '.event-page-market-box-headerLabel, '
        + '[class*="marketName" i], h6, [class*="headerLabel" i]');
        for (const e of els) {
          const t = (e.innerText || '').trim().replace(/\\s+/g, ' ');
          if (!t || t.length > 80 || seen.has(t)) continue;
          seen.add(t);
          out.push(t);
        }
        // Also dump market-box-root class tokens that look like market ids
        for (const e of document.querySelectorAll('[class*="market-box-root" i]')) {
          const cls = (e.className || '').toString();
          const m = cls.match(/[A-Z][A-Z0-9_/]{6,}/g) || [];
          for (const tok of m) {
            if (!seen.has(tok)) { seen.add(tok); out.push('CLASS:' + tok); }
          }
        }
        return out.slice(0, 120);
    }""")


def _parse_35_from_market(page, label: str) -> dict | None:
    """Expand accordion matching label and scrape Over/Under 3.5."""
    # Click collapse button containing the label
    btn = page.locator(
        f'button.event-page-market-box-collapseBtn:has-text("{label}")'
    )
    if btn.count() == 0:
        btn = page.get_by_text(re.compile(re.escape(label), re.I)).first
        try:
            btn.click(timeout=3000)
        except Exception:
            return None
    else:
        try:
            btn.first.click(timeout=3000)
        except Exception:
            pass
    time.sleep(0.8)

    # Prefer class-hinted market boxes; fall back to any box with 3.5 line.
    data = page.evaluate(
        """(line) => {
        const boxes = Array.from(document.querySelectorAll(
          '[class*="market-box-root" i]'));
        const hits = [];
        for (const box of boxes) {
          const cls = (box.className || '').toString();
          const text = (box.innerText || '').replace(/\\n/g, ' ').trim();
          if (!text.includes(line)) continue;
          // Odds buttons
          const overBtn = box.querySelector('button[name="Over"], button[name="OVER"]');
          const underBtn = box.querySelector('button[name="Under"], button[name="UNDER"]');
          const oddSpans = Array.from(box.querySelectorAll(
            '.odd-button-oddValue, [class*="oddValue" i], [class*="oddsValue" i]'))
            .map(s => (s.innerText || '').trim());
          const nums = (text.match(/\\d+\\.\\d+/g) || []);
          hits.push({
            cls: cls.slice(0, 160),
            text: text.slice(0, 200),
            oddSpans,
            nums,
            hasOver: !!overBtn,
            hasUnder: !!underBtn,
          });
        }
        return hits;
    }""",
        LINE,
    )
    if not data:
        return None

    for hit in data:
        nums = hit.get("nums") or []
        # Typical: line 3.5 + over + under among floats
        # Filter out the line itself when picking odds
        odds = [n for n in nums if n != LINE and float(n) > 1.0]
        over = under = None
        if len(odds) >= 2:
            over, under = odds[0], odds[1]
        elif hit.get("oddSpans") and len(hit["oddSpans"]) >= 2:
            try:
                cand = [s for s in hit["oddSpans"] if re.match(r"^\d+\.\d+$", s)]
                if len(cand) >= 2:
                    over, under = cand[0], cand[1]
            except Exception:
                pass
        if over and under and float(over) > 1.0 and float(under) > 1.0:
            return {
                "over": over,
                "under": under,
                "label": label,
                "box_class": hit.get("cls"),
                "row_text": hit.get("text", "")[:120],
            }
    return None


def _collect_fixture_urls(page, limit: int) -> list[dict]:
    page.goto(
        "https://www.pamestoixima.gr/en/next24hCoupon",
        wait_until="domcontentloaded",
        timeout=60000,
    )
    _accept_cookies(page)
    time.sleep(2.0)
    # Scroll listing a bit
    for _ in range(4):
        page.evaluate("() => window.scrollBy(0, window.innerHeight)")
        time.sleep(0.5)

    rows = page.evaluate("""() => {
        const out = [];
        const boxes = document.querySelectorAll('[class*="event-box-root" i]');
        for (const box of boxes) {
          const a = box.querySelector('a[href*="/football/"]');
          if (!a) continue;
          const href = a.getAttribute('href') || '';
          if (!href.includes('-v-')) continue;
          const home = (box.querySelector('.homeTeam, .team.homeTeam') || {}).innerText || '';
          const away = (box.querySelector('.awayTeam, .team.awayTeam') || {}).innerText || '';
          const league = (box.querySelector('[class*="sportCompetitionName" i]') || {}).innerText || '';
          out.push({
            href: href.startsWith('http') ? href : ('https://www.pamestoixima.gr' + href),
            home: (home || '').trim(),
            away: (away || '').trim(),
            league: (league || '').trim().replace(/\\s+/g, ' '),
          });
        }
        return out;
    }""")
    # Prefer Big-5-ish leagues
    prefer = ("Serie A", "LaLiga", "La Liga", "Premier", "Bundesliga",
              "Ligue 1", "Championship", "Eredivisie", "Liga Portugal",
              "England", "Spain", "Italy", "Germany", "France")
    ranked = sorted(
        rows,
        key=lambda r: (0 if any(p in (r.get("league") or "") for p in prefer) else 1,
                       r.get("league") or ""),
    )
    # unique by href
    seen = set()
    out = []
    for r in ranked:
        if r["href"] in seen:
            continue
        seen.add(r["href"])
        out.append(r)
        if len(out) >= limit:
            break
    return out


def probe_match(page, url: str, meta: dict | None = None) -> dict:
    report = {
        "url": url,
        "meta": meta or {},
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "labels": [],
        "card_labels": [],
        "winner": None,
        "error": None,
    }
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        _accept_cookies(page)
        time.sleep(1.5)
        _scroll_markets(page, passes=10)
        labels = _list_market_labels(page)
        report["labels"] = labels
        card_labels = [
            lb for lb in labels
            if _CARD_LABEL_RE.search(lb) or any(h in lb for h in _CARD_CLASS_HINTS)
        ]
        report["card_labels"] = card_labels
        print(f"  markets listed: {len(labels)}; card-ish: {card_labels[:8]}")

        # Try each card label, then a few CLASS: tokens
        candidates = card_labels[:]
        if not candidates:
            # Try opening any accordion whose class hint suggests cards
            for lb in labels:
                if lb.startswith("CLASS:") and any(h in lb for h in _CARD_CLASS_HINTS):
                    candidates.append(lb)
        # Also try common English labels even if not listed (lazy render)
        for guess in (
            "Total Yellow Cards", "Yellow Cards Over/Under",
            "Total Cards Over/Under", "Cards Over/Under",
            "Number of Cards", "Total Cards",
            "Σύνολο Κίτρινων Καρτών", "Κίτρινες Κάρτες",
            "Σύνολο Καρτών",
        ):
            if guess not in candidates:
                candidates.append(guess)

        for label in candidates[:12]:
            if label.startswith("CLASS:"):
                continue
            print(f"  try expand: {label!r}")
            parsed = _parse_35_from_market(page, label)
            if parsed:
                report["winner"] = parsed
                print(f"  ✓ 3.5 Over={parsed['over']} Under={parsed['under']}")
                break
        if not report["winner"]:
            # Dump a few labels for debugging
            print(f"  ✗ no 3.5 cards line. sample labels: {labels[:15]}")
    except Exception as e:
        report["error"] = str(e)[:300]
        print(f"  ERR {e}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--url", action="append", default=[],
                        help="Explicit match URL (repeatable)")
    parser.add_argument("--login", action="store_true",
                        help="Log into Pamestoixima first (needs Keychain creds)")
    args = parser.parse_args()

    from real_betting.session import BrowserSession, session_lock
    from real_betting import config as rb_config

    os.makedirs(OUT_DIR, exist_ok=True)
    reports = []
    n_ok = 0

    storage = os.path.join(
        rb_config.SESSION_STATE_DIR, "pamestoixima.session_state.json")
    if not os.path.isfile(storage):
        storage = None

    with session_lock(), BrowserSession(
            headless=False, storage_state_path=storage) as session:
        page = session.page

        if args.login:
            try:
                from real_betting.bookmakers.pamestoixima import Pamestoixima
                pm = Pamestoixima(headless=False, reuse_session=True)
                # Drive login on OUR open page by temporarily swapping session.
                pm._session = session  # type: ignore[attr-defined]
                print("[probe] logging in…")
                pm.login()
                print("[probe] login ok")
            except Exception as e:
                print(f"[probe] login failed ({e}); continuing as guest")

        targets: list[tuple[str, dict]] = []
        if args.url:
            for u in args.url:
                targets.append((u, {}))
        else:
            print("[probe] collecting fixtures from next24hCoupon…")
            fixtures = _collect_fixture_urls(page, limit=args.limit)
            print(f"[probe] got {len(fixtures)} fixtures")
            for f in fixtures:
                print(f"  - {f.get('league')}: {f.get('home')} vs {f.get('away')}")
                targets.append((f["href"], f))

        if not targets:
            print("[-] No fixtures / URLs to probe", file=sys.stderr)
            return 1

        for url, meta in targets:
            print(f"\n=== {meta.get('home', '?')} vs {meta.get('away', '?')} ===")
            print(f"  {url}")
            rep = probe_match(page, url, meta)
            reports.append(rep)
            if rep.get("winner"):
                n_ok += 1
            mid = (meta.get("home") or "match").replace(" ", "_")[:40]
            out = os.path.join(OUT_DIR, f"cards_odds_pame_{mid}.json")
            with open(out, "w") as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)
            print(f"  wrote {out}")

    summary = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "pamestoixima",
        "n_targets": len(reports),
        "n_ok": n_ok,
        "gate_pass": n_ok >= 2,
        "winners": [
            {"url": r["url"], "meta": r.get("meta"), **r["winner"]}
            for r in reports if r.get("winner")
        ],
        "label_samples": {
            r["url"]: r.get("card_labels") or r.get("labels", [])[:20]
            for r in reports
        },
    }
    sum_path = os.path.join(OUT_DIR, "cards_odds_pame_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== SUMMARY: {n_ok}/{len(reports)} with 3.5 "
          f"(gate ≥2: {'PASS' if summary['gate_pass'] else 'FAIL'}) ===")
    print(f"wrote {sum_path}")
    for w in summary["winners"]:
        print(f"  Over {w['over']} / Under {w['under']}  [{w.get('label')}]")
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
