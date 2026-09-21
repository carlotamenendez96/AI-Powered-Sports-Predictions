"""D4 / Paso 3 probe — READ-ONLY exploration of Flashscore match-summary
pages for the assigned referee.

Touches NO production code, model, or UI. Dumps DOM hints + HTML so
``extract_referees.py`` can lock a stable selector from reality.

Output → ``output/d4_probe/referee_<mid>.{json,html}``

Usage::

    python3 scripts/d4_referees/probe_referee.py [URL ...]
    # or pull 3 fixtures from a matches_<date>.json:
    python3 scripts/d4_referees/probe_referee.py --from-matches 2026-09-20
"""

import asyncio
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "output", "d4_probe")
MATCHES_DIR = os.path.join(ROOT, "output")


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


# Locate anything that looks like a referee row / Match information block.
_JS_REFEREE = r"""
() => {
  const clsStr = (e) => {
    const c = e && e.className;
    if (!c) return '';
    if (typeof c === 'string') return c;
    if (typeof c.baseVal === 'string') return c.baseVal;  // SVGAnimatedString
    try { return String(c); } catch (_) { return ''; }
  };
  const res = {
    class_hints: [],
    testid_hints: [],
    hits: [],
    match_info_html: null,
    full_text_snippet: '',
  };
  const els = Array.from(document.querySelectorAll(
    'div,section,ul,li,span,a,dt,dd,tr,td,th,h1,h2,h3,h4'));
  const cls = new Set();
  const tids = new Set();
  for (const e of els) {
    const c = clsStr(e);
    if (/referee|matchInfo|match-info|summaryMatch|mi__|detailMS/i.test(c))
      cls.add(c.trim().slice(0, 120));
    const tid = e.getAttribute && e.getAttribute('data-testid');
    if (tid && /referee|matchInfo|match-info|summary|information/i.test(tid))
      tids.add(tid);
  }
  res.class_hints = Array.from(cls).slice(0, 40);
  res.testid_hints = Array.from(tids).slice(0, 40);

  // Label-based hits: element text is "Referee" (short) → climb to container.
  const RE = /^\s*Referee\s*$/i;
  const RE_SOFT = /Referee/i;
  const labels = els.filter(e => {
    const t = (e.textContent || '').trim();
    return (RE.test(t) || (RE_SOFT.test(t) && t.length < 40)) && t.length < 80;
  });
  const seen = new Set();
  for (const lab of labels.slice(0, 12)) {
    let node = lab;
    for (let i = 0; i < 5 && node.parentElement; i++) node = node.parentElement;
    if (seen.has(node)) continue;
    seen.add(node);
    const links = Array.from(node.querySelectorAll('a')).map(a => ({
      href: a.getAttribute('href') || '',
      text: (a.textContent || '').trim().slice(0, 80),
    }));
    res.hits.push({
      label: (lab.textContent || '').trim().slice(0, 80),
      label_testid: lab.getAttribute('data-testid') || null,
      label_class: clsStr(lab).slice(0, 120),
      container_class: clsStr(node).slice(0, 120),
      container_testid: node.getAttribute('data-testid') || null,
      container_html: node.outerHTML.slice(0, 5000),
      container_text: (node.innerText || '').trim().slice(0, 500),
      links: links.slice(0, 10),
    });
  }

  // Prefer a known Match Information region if present.
  const info =
    document.querySelector('[data-testid*="MatchInformation" i]') ||
    document.querySelector('[data-testid*="matchInformation" i]') ||
    document.querySelector('[class*="matchInfo"]') ||
    document.querySelector('[class*="mi__"]');
  if (info) {
    res.match_info_html = info.outerHTML.slice(0, 8000);
    res.match_info_testid = info.getAttribute('data-testid');
    res.match_info_class = clsStr(info).slice(0, 120);
    res.match_info_text = (info.innerText || '').trim().slice(0, 1500);
  }

  const body = document.body;
  res.full_text_snippet = (body ? body.innerText : '').slice(0, 4000);
  return res;
}
"""


async def _grab(page, url):
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await _accept_cookies(page)
    for sel in (
        '[data-testid*="MatchInformation" i]',
        '[data-testid*="matchInformation" i]',
        '[class*="matchInfo"]',
        '.detailScore__wrapper',
        '[class*="summary"]',
    ):
        try:
            await page.wait_for_selector(sel, timeout=8000)
            break
        except Exception:
            continue
    await page.wait_for_timeout(2500)
    data = await page.evaluate(_JS_REFEREE)
    html = await page.content()
    title = await page.title()
    return data, html, title


def _urls_from_matches(date_str, n=3):
    path = os.path.join(MATCHES_DIR, f"matches_{date_str}.json")
    matches = json.load(open(path))
    urls = []
    seen_leagues = set()
    for m in matches:
        base = (m.get("base_url") or "").rstrip("/")
        mid = m.get("match_id")
        if not base or not mid:
            continue
        lg = m.get("league") or ""
        # Prefer diverse leagues first, then fill.
        if lg in seen_leagues and len(urls) < n:
            continue
        seen_leagues.add(lg)
        urls.append({
            "url": base + "/",
            "match_id": mid,
            "home_team": m.get("home_team"),
            "away_team": m.get("away_team"),
            "league": lg,
        })
        if len(urls) >= n:
            break
    if len(urls) < n:
        for m in matches:
            base = (m.get("base_url") or "").rstrip("/")
            mid = m.get("match_id")
            if not base or not mid:
                continue
            if any(u["match_id"] == mid for u in urls):
                continue
            urls.append({
                "url": base + "/",
                "match_id": mid,
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "league": m.get("league"),
            })
            if len(urls) >= n:
                break
    return urls


async def main():
    args = sys.argv[1:]
    targets = []
    if args and args[0] == "--from-matches":
        date_str = args[1] if len(args) > 1 else None
        if not date_str:
            raise SystemExit("usage: --from-matches YYYY-MM-DD")
        targets = _urls_from_matches(date_str, n=3)
    elif args:
        for u in args:
            m = re.search(r"mid=([A-Za-z0-9]+)", u)
            mid = m.group(1) if m else (u.rstrip("/").split("/")[-1] or "unknown")
            targets.append({"url": u, "match_id": mid})
    else:
        targets = _urls_from_matches("2026-09-20", n=3)

    if not targets:
        raise SystemExit("no targets to probe")

    os.makedirs(OUT_DIR, exist_ok=True)

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            locale="en-US",
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"))
        page = await ctx.new_page()
        for t in targets:
            url = t["url"]
            mid = t["match_id"]
            print(f"[probe] {mid} {t.get('home_team')} v {t.get('away_team')} ({t.get('league')})")
            print(f"        {url}")
            try:
                data, html, title = await _grab(page, url)
            except Exception as e:
                print(f"  ! failed: {e!r}")
                continue
            with open(os.path.join(OUT_DIR, f"referee_{mid}.html"), "w") as f:
                f.write(html)
            report = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "match_id": mid,
                "url": url,
                "title": title,
                "home_team": t.get("home_team"),
                "away_team": t.get("away_team"),
                "league": t.get("league"),
                **data,
            }
            path = os.path.join(OUT_DIR, f"referee_{mid}.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"  title: {title[:80]!r}")
            print(f"  testid_hints: {data.get('testid_hints')}")
            print(f"  class_hints: {data.get('class_hints')[:8]}")
            print(f"  hits: {len(data.get('hits', []))}")
            for h in data.get("hits", [])[:4]:
                print(f"    label={h['label']!r} c_testid={h.get('container_testid')!r}")
                print(f"      text={h.get('container_text', '')[:120]!r}")
                print(f"      links={h.get('links')}")
            if data.get("match_info_testid") or data.get("match_info_class"):
                print(f"  match_info testid={data.get('match_info_testid')!r} "
                      f"class={data.get('match_info_class')!r}")
                print(f"    text={((data.get('match_info_text') or '')[:200])!r}")
            print(f"  → {path}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
