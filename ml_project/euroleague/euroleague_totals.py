"""Totals-ladder selection — the single source of truth for BOTH sides.

Flashscore gives a ladder, not a line: each bookmaker posts its own total and
its own pair of prices (17-19 rows on a typical EuroLeague game). Something has
to choose which row to bet, and that choice must be identical in the predictor
(which displays it) and in ``/euroleague/auto_wager`` (which stakes it) — a
serve-time/display skew here would mean the dashboard advertises one bet and
the slip records another. Same reasoning as ``FORM_WINDOWS`` on the football
side: one constant, imported by both paths.

Two policies live here, and both are computed on every slate:

``best_ev``    the operator's chosen strategy (2026-09-27) — evaluate P(Over)
               at every offered line and take the highest-EV side/row.
``main_line``  the consensus total: the row whose over/under prices are most
               balanced. NOT bet; recorded alongside as the counterfactual.

The counterfactual is the point of this module, not decoration. Maximising
``prob × odds − 1`` across ~34 candidate sides is a maximum over noisy
estimates, and it will preferentially select rows where our own total is
furthest from the market — which is precisely where a miscalibrated model is
most wrong, the mechanism documented in CLAUDE.md's "No measured edge over the
market" (bets claiming EV > 0.50 returned −25%). Recording what the main line
would have paid, on the same games, is how we find out within one season
whether ladder-shopping is real edge or our own error, without betting twice.
"""
from __future__ import annotations

try:                                              # standalone (ml_project/euroleague on path)
    from euroleague_calibration import prob_over
except ImportError:                               # package import (web_ui)
    from ml_project.euroleague.euroleague_calibration import prob_over

# Tie-break order when rows are otherwise equivalent. Operator's call: bwin
# first. Matching is substring-on-normalised, so "Bwin.gr" hits "bwin".
BOOK_PREFERENCE = ["bwin", "bet365", "stoiximan", "pamestoixima", "novibet"]


def book_rank(book: str) -> int:
    n = "".join(c for c in (book or "").lower() if c.isalnum())
    for i, pref in enumerate(BOOK_PREFERENCE):
        if pref in n:
            return i
    return len(BOOK_PREFERENCE)


def _rows(totals) -> list:
    """Usable ladder rows: a line and both prices present."""
    out = []
    for t in (totals or []):
        try:
            line, over, under = float(t["line"]), float(t["over"]), float(t["under"])
        except (KeyError, TypeError, ValueError):
            continue
        if over > 1.0 and under > 1.0:
            out.append({"book": t.get("book", ""), "line": line,
                        "over": over, "under": under})
    return out


def main_line(totals) -> dict | None:
    """Consensus row: most balanced prices, book preference as tie-break."""
    rows = _rows(totals)
    if not rows:
        return None
    return sorted(rows, key=lambda r: (abs(r["over"] - r["under"]), book_rank(r["book"])))[0]


def evaluate_ladder(pred_total, sigma, totals) -> list[dict]:
    """Every (row, side) as a candidate bet, with model probability and EV.

    Returns dicts: {book, line, side ('Over'|'Under'), odds, prob, ev}.
    """
    out = []
    for r in _rows(totals):
        p_over = prob_over(pred_total, r["line"], sigma)
        if p_over is None:
            continue
        for side, prob, odds in (("Over", p_over, r["over"]),
                                 ("Under", 1.0 - p_over, r["under"])):
            out.append({"book": r["book"], "line": r["line"], "side": side,
                        "odds": odds, "prob": prob, "ev": prob * odds - 1.0})
    return out


def best_ev(pred_total, sigma, totals) -> dict | None:
    """Highest-EV side across the whole ladder (the strategy that gets staked).

    Ties break on book preference, then on the line closest to our own
    prediction, so the pick is deterministic run to run.
    """
    cands = evaluate_ladder(pred_total, sigma, totals)
    if not cands:
        return None
    return sorted(
        cands,
        key=lambda c: (-c["ev"], book_rank(c["book"]), abs(c["line"] - float(pred_total))),
    )[0]


def counterfactual(pred_total, sigma, totals) -> dict | None:
    """What the MAIN line would have been bet at — same shape as `best_ev`.

    Side is whichever the model favours at the consensus line, so the two
    policies are compared on the same decision, differing only in which row
    they took.
    """
    row = main_line(totals)
    if row is None:
        return None
    p_over = prob_over(pred_total, row["line"], sigma)
    if p_over is None:
        return None
    side, prob, odds = (("Over", p_over, row["over"]) if p_over >= 0.5
                        else ("Under", 1.0 - p_over, row["under"]))
    return {"book": row["book"], "line": row["line"], "side": side,
            "odds": odds, "prob": prob, "ev": prob * odds - 1.0}
