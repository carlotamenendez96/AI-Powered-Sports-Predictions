#!/usr/bin/env python3
"""Merge predictions_cards_<date>.csv columns into predictions_<date>.csv.

Cards stay an isolated model head, but the dashboard / auto_wager / results
table should see Cards as a third market cluster next to 1X2 and O/U — same
file, same match rows.

Join key: match_id (preferred), else (Home Team, Away Team).

Usage:
    python3 -m ml_project.cards.merge_into_predictions 2026-09-21
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Columns copied from predictions_cards → predictions (rename for clarity
# where the main CSV already has Over % / Under % for goals).
_CARDS_COLS = (
    ("Prediction Cards", "Prediction Cards"),
    ("Prediction Cards Odd", "Prediction Cards Odd"),
    ("Conf Cards", "Conf Cards"),
    ("EV Cards", "EV Cards"),
    ("Over %", "Over Cards %"),
    ("Under %", "Under Cards %"),
    ("Over Odd", "Over Cards Odd"),
    ("Under Odd", "Under Cards Odd"),
    ("Odds Source", "Odds Source Cards"),
    ("Cal Source", "Cal Cards Source"),
    ("Referee", "Cards Referee"),
    ("Card Line", "Card Line"),
)


def _kelly(conf: float, odd: float) -> str:
    """Quarter-Kelly string matching predict_matches style."""
    try:
        conf = float(conf)
        odd = float(odd)
    except (TypeError, ValueError):
        return "0.00%"
    if odd <= 1.0 or conf <= 0:
        return "0.00%"
    b = odd - 1.0
    q = conf
    p = 1.0 - q
    f = (b * q - p) / b if b > 0 else 0.0
    f = max(f, 0.0) * 0.25
    return f"{f:.2%}"


def merge(date: str, output_dir: str = "output") -> str:
    main_path = os.path.join(output_dir, f"predictions_{date}.csv")
    cards_path = os.path.join(output_dir, f"predictions_cards_{date}.csv")
    if not os.path.isfile(main_path):
        print(f"[cards-merge] Missing {main_path}; nothing to merge into.")
        return ""
    if not os.path.isfile(cards_path):
        print(f"[cards-merge] Missing {cards_path}; main CSV unchanged.")
        return main_path

    main = pd.read_csv(main_path)
    cards = pd.read_csv(cards_path)
    if cards.empty:
        print("[cards-merge] Cards CSV empty; main unchanged.")
        return main_path

    # Drop any previous cards columns so a re-run is idempotent.
    drop = [dst for _, dst in _CARDS_COLS if dst in main.columns]
    drop += ["Kelly Cards"]
    main = main.drop(columns=[c for c in drop if c in main.columns], errors="ignore")

    keep_src = ["match_id", "Home Team", "Away Team"] + [s for s, _ in _CARDS_COLS]
    keep_src = [c for c in keep_src if c in cards.columns]
    sub = cards[keep_src].copy()
    rename = {src: dst for src, dst in _CARDS_COLS if src in sub.columns}
    sub = sub.rename(columns=rename)

    if "match_id" in main.columns and "match_id" in sub.columns:
        # Avoid colliding team cols from the right
        sub_join = sub.drop(
            columns=[c for c in ("Home Team", "Away Team") if c in sub.columns],
            errors="ignore",
        )
        merged = main.merge(sub_join, on="match_id", how="left")
    else:
        merged = main.merge(
            sub, on=["Home Team", "Away Team"], how="left", suffixes=("", "_cards")
        )

    # Kelly Cards from conf × odd (quarter-Kelly), blank when no odd.
    kellys = []
    for _, row in merged.iterrows():
        conf = row.get("Conf Cards", "")
        odd = row.get("Prediction Cards Odd", "")
        if conf == "" or odd == "" or pd.isna(conf) or pd.isna(odd):
            kellys.append("")
        else:
            kellys.append(_kelly(conf, odd))
    merged["Kelly Cards"] = kellys

    # Ensure the full Cards cluster exists even when odds were blank in the
    # sibling CSV (so the dashboard column order stays stable).
    for _, dst in _CARDS_COLS:
        if dst not in merged.columns:
            merged[dst] = ""
    if "Kelly Cards" not in merged.columns:
        merged["Kelly Cards"] = ""

    # Column order: keep existing, insert Cards cluster after O/U cluster.
    cards_cluster = [
        "Prediction Cards", "Prediction Cards Odd", "Conf Cards", "EV Cards",
        "Kelly Cards", "Over Cards %", "Under Cards %",
        "Over Cards Odd", "Under Cards Odd", "Odds Source Cards",
        "Cal Cards Source", "Cards Referee", "Card Line",
    ]
    cols = list(merged.columns)
    # Remove cards cols then re-insert after Kelly O/U (or Under %)
    for c in cards_cluster:
        if c in cols:
            cols.remove(c)
    anchor = None
    for a in ("Kelly O/U", "Under %", "Over %", "EV O/U"):
        if a in cols:
            anchor = a
            break
    if anchor is None:
        cols.extend([c for c in cards_cluster if c in merged.columns])
    else:
        i = cols.index(anchor) + 1
        insert = [c for c in cards_cluster if c in merged.columns]
        cols = cols[:i] + insert + cols[i:]
    merged = merged[cols]

    merged.to_csv(main_path, index=False)
    n = int(merged["Prediction Cards"].notna().sum()) if "Prediction Cards" in merged.columns else 0
    print(f"[cards-merge] Merged {n} Cards rows into {main_path}")
    return main_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    path = merge(args.date, args.output_dir)
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
