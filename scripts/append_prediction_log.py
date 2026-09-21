#!/usr/bin/env python3
"""Append / upsert football predictions + verification into the git-tracked log.

Reads the daily artifacts under output/ (gitignored, ephemeral) and writes a
cumulative JSONL under history/football/ (tracked, committed by the daily
Actions job). Rows are keyed by (date, match_id) so re-runs update in place
rather than duplicating; verification fields are filled when a verification
CSV exists for that date.

Usage:
    python3 scripts/append_prediction_log.py
    python3 scripts/append_prediction_log.py --pred-date 2026-09-21 --verify-date 2026-09-20
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "history", "football")
LOG_PATH = os.path.join(LOG_DIR, "daily.jsonl")


def _f(v):
    """Parse a float from CSV cell (EV/Conf/Odd may be str)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _s(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def _boolish(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "✅"):
        return True
    if s in ("0", "false", "no", "❌"):
        return False
    return bool(v)


def _row_key(date_str: str, match_id: str | None, home: str | None, away: str | None) -> str:
    if match_id:
        return f"{date_str}|id:{match_id}"
    return f"{date_str}|{home or ''}|{away or ''}"


def load_log(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _row_key(
                obj.get("date", ""),
                obj.get("match_id"),
                obj.get("home"),
                obj.get("away"),
            )
            rows[key] = obj
    return rows


def write_log(path: str, rows: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Stable order: date, then home, then match_id.
    ordered = sorted(
        rows.values(),
        key=lambda r: (r.get("date") or "", r.get("home") or "", r.get("match_id") or ""),
    )
    with open(path, "w", encoding="utf-8") as f:
        for obj in ordered:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def upsert_predictions(rows: dict[str, dict], pred_date: str) -> int:
    path = os.path.join(OUTPUT_DIR, f"predictions_{pred_date}.csv")
    if not os.path.isfile(path):
        print(f"[append_log] no predictions file at {path} — skip pred upsert")
        return 0

    df = pd.read_csv(path)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = 0
    for _, r in df.iterrows():
        match_id = _s(r.get("match_id"))
        home = _s(r.get("Home Team"))
        away = _s(r.get("Away Team"))
        key = _row_key(pred_date, match_id, home, away)
        existing = rows.get(key, {})
        entry = {
            **existing,
            "date": pred_date,
            "match_id": match_id,
            "league": _s(r.get("League")),
            "home": home,
            "away": away,
            "kickoff": _s(r.get("Date")),
            "pred_1x2": _s(r.get("Prediction 1X2")),
            "conf_1x2": _f(r.get("Conf 1X2")),
            "odds_1x2": _f(r.get("Prediction 1X2 Odd")),
            "ev_1x2": _f(r.get("EV 1X2")),
            "pred_ou": _s(r.get("Prediction O/U")),
            "conf_ou": _f(r.get("Conf O/U")),
            "odds_ou": _f(r.get("Prediction O/U Odd")),
            "ev_ou": _f(r.get("EV O/U")),
            "logged_at": existing.get("logged_at") or now,
        }
        rows[key] = entry
        n += 1
    print(f"[append_log] upserted {n} prediction rows for {pred_date}")
    return n


def upsert_verification(rows: dict[str, dict], verify_date: str) -> int:
    path = os.path.join(OUTPUT_DIR, f"verification_{verify_date}.csv")
    if not os.path.isfile(path):
        print(f"[append_log] no verification file at {path} — skip verify upsert")
        return 0

    df = pd.read_csv(path)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = 0
    for _, r in df.iterrows():
        home = _s(r.get("Home")) or (_s(r.get("Match")) or "").split(" vs ")[0]
        away = _s(r.get("Away"))
        if not away and " vs " in str(r.get("Match", "")):
            away = str(r.get("Match")).split(" vs ", 1)[1].strip()
        # Verification CSV has no match_id — join on date + team names.
        key = None
        for k, obj in rows.items():
            if obj.get("date") != verify_date:
                continue
            if obj.get("home") == home and (away is None or obj.get("away") == away):
                key = k
                break
        if key is None:
            # Create a verification-only stub so the day is not lost.
            key = _row_key(verify_date, None, home, away)
            rows[key] = {
                "date": verify_date,
                "match_id": None,
                "league": _s(r.get("League")),
                "home": home,
                "away": away,
                "logged_at": now,
            }

        rows[key].update({
            "score": _s(r.get("Score")),
            "actual_1x2": _s(r.get("Actual 1X2")),
            "hit_1x2": _boolish(r.get("Correct 1X2")),
            "actual_ou": _s(r.get("Actual O/U")),
            "hit_ou": _boolish(r.get("Correct O/U")),
            "verified_at": now,
        })
        # Prefer pred labels from verification if prediction row was missing.
        if not rows[key].get("pred_1x2"):
            rows[key]["pred_1x2"] = _s(r.get("Pred 1X2"))
        if not rows[key].get("pred_ou"):
            rows[key]["pred_ou"] = _s(r.get("Pred O/U"))
        n += 1
    print(f"[append_log] upserted {n} verification rows for {verify_date}")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-date",
        help="ISO date of predictions to append (default: tomorrow)",
    )
    parser.add_argument(
        "--verify-date",
        help="ISO date of verification to merge (default: yesterday)",
    )
    parser.add_argument(
        "--log",
        default=LOG_PATH,
        help=f"JSONL path (default: {LOG_PATH})",
    )
    args = parser.parse_args()

    today = dt.date.today()
    pred_date = args.pred_date or (today + dt.timedelta(days=1)).isoformat()
    verify_date = args.verify_date or (today - dt.timedelta(days=1)).isoformat()

    rows = load_log(args.log)
    before = json.dumps(rows, sort_keys=True)

    upsert_predictions(rows, pred_date)
    upsert_verification(rows, verify_date)

    after = json.dumps(rows, sort_keys=True)
    if before == after:
        print("[append_log] no changes — log untouched")
        return 0

    write_log(args.log, rows)
    print(f"[append_log] wrote {len(rows)} rows → {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
