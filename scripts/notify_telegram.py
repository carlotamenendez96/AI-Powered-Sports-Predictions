#!/usr/bin/env python3
"""Send a daily football predictions + verification summary via Telegram.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment (never
hardcoded). Prefers the git-tracked JSONL log under history/football/; falls
back to output/predictions_*.csv and output/verification_*.csv when the log
has not been updated yet for those dates.

Usage:
    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python3 scripts/notify_telegram.py
    python3 scripts/notify_telegram.py --pred-date 2026-09-21 --verify-date 2026-09-20 --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_PATH = os.path.join(PROJECT_ROOT, "history", "football", "daily.jsonl")
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LEN = 4000  # Telegram hard limit is 4096; leave headroom


def _f(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(str(v).replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def predictions_for(date_str: str, log_rows: list[dict]) -> list[dict]:
    from_log = [r for r in log_rows if r.get("date") == date_str and r.get("pred_1x2")]
    if from_log:
        return from_log

    path = os.path.join(OUTPUT_DIR, f"predictions_{date_str}.csv")
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        out.append({
            "date": date_str,
            "league": r.get("League"),
            "home": r.get("Home Team"),
            "away": r.get("Away Team"),
            "pred_1x2": r.get("Prediction 1X2"),
            "conf_1x2": _f(r.get("Conf 1X2")),
            "odds_1x2": _f(r.get("Prediction 1X2 Odd")),
            "ev_1x2": _f(r.get("EV 1X2")),
            "pred_ou": r.get("Prediction O/U"),
            "odds_ou": _f(r.get("Prediction O/U Odd")),
            "ev_ou": _f(r.get("EV O/U")),
        })
    return out


def verifications_for(date_str: str, log_rows: list[dict]) -> list[dict]:
    from_log = [r for r in log_rows if r.get("date") == date_str and r.get("verified_at")]
    if from_log:
        return from_log

    path = os.path.join(OUTPUT_DIR, f"verification_{date_str}.csv")
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        hit = r.get("Correct 1X2")
        if isinstance(hit, str):
            hit = hit.strip().lower() in ("true", "1", "yes", "✅")
        out.append({
            "date": date_str,
            "home": r.get("Home"),
            "away": r.get("Away"),
            "league": r.get("League"),
            "score": r.get("Score"),
            "pred_1x2": r.get("Pred 1X2"),
            "actual_1x2": r.get("Actual 1X2"),
            "hit_1x2": bool(hit) if hit is not None and not (isinstance(hit, float) and pd.isna(hit)) else None,
            "pred_ou": r.get("Pred O/U"),
            "actual_ou": r.get("Actual O/U"),
        })
    return out


def _fmt_pct(x) -> str:
    if x is None:
        return "?"
    return f"{x:.0%}" if x <= 1.0 else f"{x:.0f}%"


def _fmt_ev(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2f}"


def _fmt_odd(x) -> str:
    if x is None:
        return "?"
    return f"{x:.2f}"


def _label_1x2(code) -> str:
    return {"1": "Home", "X": "Draw", "2": "Away"}.get(str(code), str(code or "?"))


def build_message(pred_date: str, verify_date: str, preds: list[dict], verifs: list[dict]) -> str:
    today = dt.date.today().isoformat()
    lines = [
        f"⚽ Football daily · {today}",
        "",
        f"📋 Predictions for {pred_date}",
    ]

    if not preds:
        lines.append("No matches / no predictions for this date.")
    else:
        lines.append(f"{len(preds)} fixtures:")
        # Group by league for readability.
        by_league: dict[str, list] = {}
        for p in preds:
            by_league.setdefault(str(p.get("league") or "Unknown"), []).append(p)
        for league in sorted(by_league):
            lines.append(f"\n[{league}]")
            for p in by_league[league]:
                pick = _label_1x2(p.get("pred_1x2"))
                lines.append(
                    f"• {p.get('home')} vs {p.get('away')}: "
                    f"{pick} @ {_fmt_odd(p.get('odds_1x2'))} "
                    f"(conf {_fmt_pct(p.get('conf_1x2'))}, EV {_fmt_ev(p.get('ev_1x2'))})"
                )

    lines += ["", f"✅ Verification for {verify_date}"]
    if not verifs:
        lines.append("No verification results available yet.")
    else:
        scored = [v for v in verifs if v.get("hit_1x2") is not None]
        hits = sum(1 for v in scored if v.get("hit_1x2"))
        n = len(scored)
        rate = (hits / n * 100) if n else 0.0
        lines.append(f"1X2 accuracy: {hits}/{n} ({rate:.1f}%)")
        misses = [v for v in scored if not v.get("hit_1x2")]
        if misses:
            lines.append("Notable misses:")
            for v in misses[:8]:
                lines.append(
                    f"• {v.get('home')} vs {v.get('away')} "
                    f"{v.get('score') or '?'} — pred {_label_1x2(v.get('pred_1x2'))}, "
                    f"actual {_label_1x2(v.get('actual_1x2'))}"
                )
            if len(misses) > 8:
                lines.append(f"  …and {len(misses) - 8} more")

    text = "\n".join(lines)
    if len(text) > MAX_MESSAGE_LEN:
        text = text[: MAX_MESSAGE_LEN - 20] + "\n…(truncated)"
    return text


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = TELEGRAM_API.format(token=token)
    resp = requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"Telegram API HTTP {resp.status_code}: {resp.text[:500]}"
        )
    body = resp.json()
    if not body.get("ok"):
        raise SystemExit(f"Telegram API error: {body}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-date", help="Predictions date (default: tomorrow)")
    parser.add_argument("--verify-date", help="Verification date (default: yesterday)")
    parser.add_argument("--log", default=LOG_PATH, help="JSONL log path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message and exit 0 without calling Telegram",
    )
    args = parser.parse_args()

    today = dt.date.today()
    pred_date = args.pred_date or (today + dt.timedelta(days=1)).isoformat()
    verify_date = args.verify_date or (today - dt.timedelta(days=1)).isoformat()

    log_rows = _load_jsonl(args.log)
    preds = predictions_for(pred_date, log_rows)
    verifs = verifications_for(verify_date, log_rows)
    message = build_message(pred_date, verify_date, preds, verifs)

    if args.dry_run:
        print(message)
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    if not chat_id:
        raise SystemExit("TELEGRAM_CHAT_ID is not set")

    send_telegram(token, chat_id, message)
    print(f"[notify] sent Telegram message ({len(message)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
