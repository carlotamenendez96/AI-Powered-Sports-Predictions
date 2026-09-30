"""Score one day's Euroleague/EuroCup predictions against final results.

    python3 ml_project/euroleague/evaluate_euroleague_predictions.py --date 2026-09-29

Reads ``output_euroleague/predictions_euroleague_<date>.csv``, joins the
finished games from the corpus (``team_game_stats.csv``) on ``gameId`` — the
same exact join the bet settler uses, never fuzzy names — and writes
``output_euroleague/verification_euroleague_<date>.csv``: one row per game with
the pick, the final score, and whether the winner pick and the O/U lean at the
bookmaker's main line were right. It is the Euroleague counterpart of
football's ``verification_<date>.csv`` and is what the dashboard's
Verification Reports column lists.

Games not yet finished are left out; if none are finished nothing is written
(so an early run does not produce an empty report). Exit 0 either way — the
verification wrapper treats this step as non-fatal.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from ml_project.resolve_basketball_bets import load_results  # noqa: E402

OUT_DIR = os.path.join(_REPO, "output_euroleague")
CORPUS = os.path.join(_REPO, "data_sets", "Euroleague", "team_game_stats.csv")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def evaluate(date: str) -> int:
    pred_path = os.path.join(OUT_DIR, f"predictions_euroleague_{date}.csv")
    if not os.path.exists(pred_path):
        print(f"[evaluate] no predictions for {date} ({os.path.basename(pred_path)})")
        return 0
    preds = pd.read_csv(pred_path)
    results = load_results(CORPUS)

    rows = []
    for _, p in preds.iterrows():
        res = results.get(str(p.get("gameId")))
        if not res:
            continue
        pick = str(p.get("Predicted Winner", "")).upper()
        actual = "HOME" if res["home_won"] else "AWAY"
        line = _f(p.get("Over Line"))
        p_over = _f(p.get("P(Over)"))
        ou_lean = ou_actual = ou_ok = ""
        if line is not None and p_over is not None:
            ou_lean = "Over" if p_over >= 0.5 else "Under"
            if res["total"] != line:                     # .5 lines never push
                ou_actual = "Over" if res["total"] > line else "Under"
                ou_ok = int(ou_lean == ou_actual)
        rows.append({
            "Date": p.get("Date"),
            "competition": p.get("competition"),
            "Home Team": p.get("Home Short") or p.get("Home Team"),
            "Away Team": p.get("Away Short") or p.get("Away Team"),
            "Home Win Prob": p.get("Home Win Prob"),
            "Predicted Winner": pick,
            "Final Score": f"{res['home_score']}-{res['away_score']}",
            "Actual Winner": actual,
            "Winner Correct": int(pick == actual),
            "Predicted Total": p.get("Predicted Total"),
            "Actual Total": res["total"],
            "Over Line": line if line is not None else "",
            "P(Over)": p_over if p_over is not None else "",
            "O/U Lean": ou_lean,
            "O/U Actual": ou_actual,
            "O/U Correct": ou_ok,
            "gameId": p.get("gameId"),
        })

    if not rows:
        print(f"[evaluate] {date}: none of {len(preds)} predicted games have finished yet — no report written")
        return 0

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, f"verification_euroleague_{date}.csv")
    df.to_csv(out, index=False)
    ou = df[df["O/U Correct"] != ""]
    brier = ((pd.to_numeric(df["Home Win Prob"]) - (df["Actual Winner"] == "HOME").astype(float)) ** 2).mean()
    print(f"[evaluate] {date}: {len(df)}/{len(preds)} games finished · "
          f"winner {df['Winner Correct'].sum()}/{len(df)} · "
          f"O/U {int(ou['O/U Correct'].sum()) if len(ou) else 0}/{len(ou)} · "
          f"Brier {brier:.4f} → {os.path.relpath(out, _REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    return evaluate(ap.parse_args().date)


if __name__ == "__main__":
    sys.exit(main())
