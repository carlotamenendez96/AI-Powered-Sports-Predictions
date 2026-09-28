"""Settle basketball bet slips (Euroleague/EuroCup + NBA) against the corpus.

Sibling to ``resolve_daily_bets.py``, which is FOOTBALL-ONLY and cannot be
reused here — not by preference but because it is wrong for basketball in two
ways that both fail silently:

* ``res_ou = "OVER" if (h_score + a_score) > 2.5`` hardcodes football's single
  O/U line. A basketball total of 160 is always over 2.5, so **every** totals
  bet would settle WON regardless of the line actually taken.
* 1X2 assumes a draw exists and expects ``1``/``X``/``2``; basketball has no
  draw and our moneyline selection is a team NAME.

Differences from the football resolver that are improvements, not just
adaptations:

* **Join on ``gameId``, not fuzzy team names.** Bets carry the fixture's
  ``gameId`` (``E2026_13``) in ``match_id`` and the corpus is keyed by the same
  string, so settlement is exact. Football needs rapidfuzz because its two
  sources disagree on club names; here they cannot.
* **Each totals bet settles against ITS OWN line.** Flashscore posts a ladder
  and the staked row is whatever ``euroleague_totals.best_ev`` chose, so the
  line lives on the bet (``line``/``side``), not in a constant.
* **PUSH is a real outcome.** Integer lines exist; total == line refunds the
  stake. Football's 2.5-style lines can never push, so it has no such state.
* **The counterfactual is scored but never paid.** O/U bets carry the
  main-line bet that was deliberately not placed; this settles it too and
  stores the result under ``counterfactual_result`` so ladder-shopping can be
  compared against the consensus line on identical games. Nothing about it
  touches a bankroll.

Idempotent: only ``OPEN`` bets are touched, so a re-run credits nothing twice,
and a game missing from the corpus leaves its bet OPEN for a later run (same
partial-friendly contract as football's).

Usage:
    python3 ml_project/resolve_basketball_bets.py --sport euroleague [--date YYYY-MM-DD]
    python3 ml_project/resolve_basketball_bets.py --sport euroleague --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPORTS = {
    "euroleague": {"corpus": "data_sets/Euroleague/team_game_stats.csv",
                   "bets_dir": "output_euroleague"},
    "nba":        {"corpus": "data_sets/NBA/team_game_stats.csv",
                   "bets_dir": "output_basketball"},
}


def load_results(corpus_path: str) -> dict:
    """{gameId: {home_score, away_score, total, home_won, date}} for finished games."""
    df = pd.read_csv(corpus_path, low_memory=False,
                     usecols=["gameId", "date", "home", "teamScore", "opponentScore", "win"])
    out = {}
    for gid, g in df.groupby("gameId"):
        home = g[g["home"] == 1]
        if home.empty:
            continue
        r = home.iloc[0]
        try:
            hs, as_ = int(r["teamScore"]), int(r["opponentScore"])
        except (TypeError, ValueError):
            continue
        out[str(gid)] = {"home_score": hs, "away_score": as_, "total": hs + as_,
                         "home_won": bool(int(r["win"])), "date": str(r["date"])}
    return out


def _parse_ou(bet: dict):
    """(side, line) for a totals bet. Prefers the explicit fields written by
    auto_wager; falls back to parsing 'Over 168.5' for slips placed before
    those existed."""
    side, line = bet.get("side"), bet.get("line")
    if side and line is not None:
        try:
            return str(side).capitalize(), float(line)
        except (TypeError, ValueError):
            pass
    sel = str(bet.get("selection") or "")
    parts = sel.split()
    if len(parts) >= 2 and parts[0].lower() in ("over", "under"):
        try:
            return parts[0].capitalize(), float(parts[1])
        except ValueError:
            return None, None
    return None, None


def settle_one(bet: dict, res: dict):
    """(status, payout_multiplier_on_stake) — WON / LOST / PUSH / None.

    None means 'cannot decide', which leaves the bet OPEN rather than guessing.
    """
    btype = (bet.get("type") or "").upper()

    if btype in ("ML", "MONEYLINE"):
        sel = (bet.get("selection") or "").strip()
        # Decide by SIDE, not by matching the selection against corpus team
        # names: the bet's own `home`/`away` came from the same fixture record
        # as the selection, so this cannot drift, whereas corpus `teamName`
        # spellings can.
        if sel and sel == (bet.get("home") or "").strip():
            won = res["home_won"]
        elif sel and sel == (bet.get("away") or "").strip():
            won = not res["home_won"]
        else:
            return None, 0.0
        return ("WON", float(bet.get("odds") or bet.get("odd") or 0)) if won else ("LOST", 0.0)

    if btype in ("O/U", "OU", "TOTAL", "TOTALS"):
        side, line = _parse_ou(bet)
        if side is None or line is None:
            return None, 0.0
        total = res["total"]
        if total == line:
            return "PUSH", 1.0                      # stake back
        over_won = total > line
        won = over_won if side == "Over" else not over_won
        return ("WON", float(bet.get("odds") or bet.get("odd") or 0)) if won else ("LOST", 0.0)

    return None, 0.0


def _score_counterfactual(bet: dict, res: dict):
    """Settle the unplaced main-line bet at the SAME stake. Never credited."""
    cf = bet.get("counterfactual")
    if not isinstance(cf, dict):
        return None
    shadow = {"type": "O/U", "side": cf.get("side"), "line": cf.get("line"),
              "odds": cf.get("odds")}
    status, mult = settle_one(shadow, res)
    if status is None:
        return None
    stake = float(bet.get("stake") or bet.get("stake_units") or 0)
    return {"status": status, "line": cf.get("line"), "side": cf.get("side"),
            "odds": cf.get("odds"), "pnl": round(stake * mult - stake, 2)}


def resolve(sport: str, date: str | None = None, dry_run: bool = False) -> int:
    cfg = SPORTS[sport]
    corpus_path = os.path.join(_REPO, cfg["corpus"])
    bets_dir = os.path.join(_REPO, cfg["bets_dir"])
    if not os.path.exists(corpus_path):
        print(f"[resolve] no corpus at {corpus_path}", file=sys.stderr)
        return 1

    results = load_results(corpus_path)
    print(f"[resolve] {sport}: {len(results):,} finished games in the corpus")

    paths = ([os.path.join(bets_dir, f"bets_{date}.json")] if date
             else sorted(glob.glob(os.path.join(bets_dir, "bets_*.json"))))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"[resolve] no bet slips in {bets_dir}" + (f" for {date}" if date else ""))
        return 0

    # Bankroll credits go through sports_config, never the JSON directly
    # (CLAUDE.md). Imported lazily so --dry-run and the unit path work even if
    # the web_ui package cannot be imported.
    if _REPO not in sys.path:
        sys.path.append(_REPO)
    web_ui_dir = os.path.join(_REPO, "web_ui")
    if web_ui_dir not in sys.path:
        sys.path.append(web_ui_dir)
    from sports_config import update_bankroll               # noqa: E402

    grand = {"settled": 0, "open": 0, "won": 0, "lost": 0, "push": 0}
    for path in paths:
        with open(path) as f:
            slip = json.load(f)
        bets = slip.get("bets") or []
        credit_by_lane: dict = {}
        changed = False

        for bet in bets:
            if (bet.get("status") or "OPEN").upper() != "OPEN":
                continue                                   # idempotent
            gid = str(bet.get("match_id") or "")
            res = results.get(gid)
            if not res:
                grand["open"] += 1
                continue                                   # not finished yet
            status, mult = settle_one(bet, res)
            if status is None:
                print(f"    ? undecidable: {bet.get('type')} {bet.get('selection')!r} ({gid})")
                grand["open"] += 1
                continue

            stake = float(bet.get("stake") or bet.get("stake_units") or 0)
            payout = round(stake * mult, 2)
            bet["status"] = status
            bet["result"] = status
            bet["final_score"] = f"{res['home_score']}-{res['away_score']}"
            bet["final_total"] = res["total"]
            bet["payout"] = payout
            bet["pnl"] = round(payout - stake, 2)
            bet["profit"] = bet["pnl"]                     # football-compatible alias
            cf = _score_counterfactual(bet, res)
            if cf:
                bet["counterfactual_result"] = cf
            credit_by_lane[bet.get("lane", "value")] = \
                credit_by_lane.get(bet.get("lane", "value"), 0.0) + payout
            grand["settled"] += 1
            grand[status.lower()] = grand.get(status.lower(), 0) + 1
            changed = True

        if not changed:
            continue

        slip["pnl"] = round(sum(float(b.get("pnl") or 0) for b in bets), 2)
        slip["return_by_lane"] = {k: round(v, 2) for k, v in credit_by_lane.items()}
        still_open = [b for b in bets if (b.get("status") or "OPEN").upper() == "OPEN"]
        slip["status"] = "OPEN" if still_open else "CLOSED"
        slip["settled"] = not still_open

        print(f"  {os.path.basename(path)}: settled {len(bets) - len(still_open)}/{len(bets)}"
              f" | slip P/L {slip['pnl']:+.2f} | credits {slip['return_by_lane']}")
        if dry_run:
            continue
        with open(path, "w") as f:
            json.dump(slip, f, indent=4)
        for lane, amount in credit_by_lane.items():
            if amount:
                update_bankroll(sport, amount, lane=lane)

    print(f"[resolve] {grand['settled']} settled "
          f"(won {grand['won']}, lost {grand['lost']}, push {grand['push']}), "
          f"{grand['open']} left OPEN" + ("  [DRY RUN — nothing written]" if dry_run else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sport", required=True, choices=sorted(SPORTS))
    ap.add_argument("--date", default=None, help="settle only bets_<date>.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would settle; write nothing, credit nothing")
    a = ap.parse_args()
    return resolve(a.sport, a.date, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
