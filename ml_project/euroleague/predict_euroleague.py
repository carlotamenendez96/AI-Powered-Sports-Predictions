"""Euroleague/EuroCup daily predictor — winner + total, serve-time parity.

Mirrors ``ml_project/nba/predict_nba.py``. Reads
``data_sets/Euroleague/fixtures_<date>.json`` (from ``fetch_euroleague_daily.py
fixtures``), recomputes per-fixture features from the **same corpus**
``team_game_stats.csv`` the model trained on (kills train/serve skew), runs the
combined winner + total models, applies the **per-competition** Platt
calibrator, and writes ``output_euroleague/predictions_euroleague_<date>.csv``.

Euroleague specifics vs NBA:
* fixtures carry ``competition`` + per-competition ``home/away_team_id`` (the
  TeamIdRegistry namespaces ids by competition, so ELO/rolling lookups by id are
  already competition-correct).
* the combined model's ``is_eurocup`` game-level feature is set from the fixture.
* calibration is per competition (``apply_home_win_platt(..., competition=...)``).

CSV columns match the NBA/contract shape downstream ``auto_wager`` understands:
``gameId, competition, Home Team, Away Team, Home Win Prob, Home Win Prob (raw),
Cal Source, Predicted Winner, Predicted Total`` (+ ELO/rest for explainability).
"""

import argparse
import datetime
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_REPO, "data_sets", "Euroleague")
CORPUS = os.path.join(DATA, "team_game_stats.csv")
ELO_CACHE = os.path.join(DATA, "euroleague_elo.json")
FIXTURES_TMPL = os.path.join(DATA, "fixtures_{date}.json")
CALIBRATION_PATH = os.path.join(DATA, "euroleague_calibration.json")

MODEL_DIR = os.path.join(_REPO, "models", "euroleague")
WINNER_MODEL = os.path.join(MODEL_DIR, "winner_model.pkl")
TOTAL_MODEL = os.path.join(MODEL_DIR, "total_model.pkl")
WINNER_FEATURES = os.path.join(MODEL_DIR, "features_winner.json")
TOTAL_FEATURES = os.path.join(MODEL_DIR, "features_total.json")

OUT_DIR = os.path.join(_REPO, "output_euroleague")

ELO_INIT = 1500.0
ROLL_METRICS = {
    "pts": "teamScore", "pts_allowed": "opponentScore", "win": "win",
    "fg_pct": "fieldGoalsPercentage", "fg3_pct": "threePointersPercentage",
    "ft_pct": "freeThrowsPercentage", "reb": "reboundsTotal",
    "ast": "assists", "tov": "turnovers", "plus_minus": "plusMinusPoints",
}
L10_METRICS = ("pts", "pts_allowed", "win", "plus_minus")
VENUE_METRICS = ("pts", "pts_allowed", "win")


def team_features(team_id: int, is_home: bool, game_date: pd.Timestamp,
                  team_rows: pd.DataFrame, elo: dict) -> dict:
    """Serve-time feature vector for one team (mirrors euroleague_feature_engineering)."""
    prior = team_rows[team_rows["date"] < game_date.strftime("%Y-%m-%d")].sort_values("date")
    out: dict = {}
    if len(prior):
        last_date = pd.to_datetime(prior["date"].iloc[-1])
        rd = max(0, min(14, (game_date - last_date).days))
    else:
        rd = 7
    out["rest_days"] = int(rd)
    out["b2b"] = 1 if rd == 1 else 0
    out["elo_pre"] = float(elo.get(str(team_id), ELO_INIT))

    last5 = prior.tail(5)
    for short, col in ROLL_METRICS.items():
        out[f"l5_{short}"] = float(last5[col].mean()) if len(last5) else np.nan
    last10 = prior.tail(10)
    for short in L10_METRICS:
        col = ROLL_METRICS[short]
        out[f"l10_{short}"] = float(last10[col].mean()) if len(last10) else np.nan
    venue_prior = prior[prior["home"] == (1 if is_home else 0)].tail(5)
    for short in VENUE_METRICS:
        col = ROLL_METRICS[short]
        out[f"venue_l5_{short}"] = float(venue_prior[col].mean()) if len(venue_prior) else np.nan
    return out


def build_predict_row(home_team_id, away_team_id, competition, game_date, corpus, elo) -> dict:
    home_rows = corpus[corpus["teamId"] == home_team_id]
    away_rows = corpus[corpus["teamId"] == away_team_id]
    row: dict = {"is_eurocup": 1 if competition == "U" else 0}
    for k, v in team_features(home_team_id, True, game_date, home_rows, elo).items():
        row[f"home_{k}"] = v
    for k, v in team_features(away_team_id, False, game_date, away_rows, elo).items():
        row[f"away_{k}"] = v
    return row


def predict(date_str: str | None = None) -> int:
    date_str = date_str or (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    fix_path = FIXTURES_TMPL.format(date=date_str)
    if not os.path.exists(fix_path):
        print(f"[predict] no fixtures file: {fix_path}\n"
              f"  run: python3 ml_project/euroleague/fetch_euroleague_daily.py fixtures --date {date_str}",
              file=sys.stderr)
        return 1
    fixtures = json.load(open(fix_path))
    print(f"[predict] {len(fixtures)} fixtures from {os.path.basename(fix_path)}")
    if not fixtures:
        print("[predict] no games — nothing to predict.")
        return 0

    corpus = pd.read_csv(CORPUS, low_memory=False, usecols=[
        "gameId", "date", "teamId", "home", "teamScore", "opponentScore",
        "fieldGoalsPercentage", "threePointersPercentage", "freeThrowsPercentage",
        "reboundsTotal", "assists", "turnovers", "plusMinusPoints", "win"])
    elo = json.load(open(ELO_CACHE)) if os.path.exists(ELO_CACHE) else {}

    with open(WINNER_MODEL, "rb") as f: winner = pickle.load(f)
    with open(TOTAL_MODEL, "rb") as f: total = pickle.load(f)
    win_features = json.load(open(WINNER_FEATURES))
    tot_features = json.load(open(TOTAL_FEATURES))

    from euroleague_calibration import load_calibration_data, apply_home_win_platt, prob_over, total_sigma
    from euroleague_totals import best_ev, counterfactual
    # The SAME completeness contract the trainer enforces. euroleague_feature_
    # engineering drops every game with a NaN in these columns, so the model has
    # literally never seen one — serving such a fixture is extrapolation, not
    # prediction. Imported rather than restated so the two cannot drift.
    from euroleague_feature_engineering import L10_METRICS, VENUE_METRICS
    required_feats = ([f"home_l10_{m}" for m in L10_METRICS]
                      + [f"away_l10_{m}" for m in L10_METRICS]
                      + [f"home_venue_l5_{m}" for m in VENUE_METRICS]
                      + [f"away_venue_l5_{m}" for m in VENUE_METRICS])
    cal = load_calibration_data(CALIBRATION_PATH)
    print(f"[predict] calibration: {'loaded (per-competition)' if cal else 'none — raw probs served'}")

    # Odds (optional, season-gated): join euroleague_odds_<date>.json for the
    # total LINE so we can emit P(Over). No odds file yet → P(Over) blank.
    odds_path = os.path.join(OUT_DIR, f"euroleague_odds_{date_str}.json")
    odds_by_pair = {}
    if os.path.exists(odds_path):
        try:
            odds_by_pair = {(r.get("home_team"), r.get("away_team")): r
                            for r in (json.load(open(odds_path)) or [])}
        except Exception:
            pass

    game_date = pd.Timestamp(date_str)
    rows: list[dict] = []
    for fx in fixtures:
        h_id, a_id = fx.get("home_team_id"), fx.get("away_team_id")
        comp = fx.get("competition")
        if not h_id or not a_id:
            print(f"  ⚠ skipping fixture with missing team_id: {fx}")
            continue
        feats = build_predict_row(int(h_id), int(a_id), comp, game_date, corpus, elo)
        x_win = pd.DataFrame([{f: feats.get(f, np.nan) for f in win_features}])
        x_tot = pd.DataFrame([{f: feats.get(f, np.nan) for f in tot_features}])
        raw_p = float(winner.predict_proba(x_win)[0, 1])
        cal_p, _, cal_src = apply_home_win_platt(raw_p, cal, competition=comp, enabled=bool(cal))
        pred_total = float(total.predict(x_tot)[0])
        sigma = total_sigma(cal, competition=comp)   # per-competition σ, pooled fallback
        odds_row = odds_by_pair.get((fx.get("home_team"), fx.get("away_team"))) or {}
        over_line = odds_row.get("total")
        p_over = prob_over(pred_total, over_line, sigma) if (over_line is not None and sigma) else None

        # Ladder selection. `Over Line` / `P(Over)` above stay the MAIN line —
        # they are the audit trail and the counterfactual reference — while the
        # Bet* columns carry the row the strategy would actually stake, chosen
        # by EV across every offered line. Both come from euroleague_totals so
        # the dashboard can never advertise a different bet than auto_wager
        # stakes. Cf* is the main-line bet we deliberately do NOT place, kept so
        # the two policies can be compared on settled money later.
        # Training-contract check. Measured 2026-09-28: 75% of EuroCup fixtures
        # (vs 10% of EuroLeague) carry a NaN in these columns, because 13 of 32
        # EuroCup teams have ZERO rows in the corpus — the competition churns
        # its field every season. On those the total model predicts 178.6 mean
        # against its own training mean of 164.1, and sits +8.6 above the market
        # (n=15, t=+4.13) where EuroLeague sits at +0.6 (t=+0.37). That is not a
        # bias to calibrate away, it is the model extrapolating outside the
        # shape it was fitted on, so those fixtures are not priced.
        gaps = [c for c in required_feats
                if feats.get(c) is None or (isinstance(feats.get(c), float) and np.isnan(feats.get(c)))]
        totals_ok = not gaps
        # Display names. The API's are sponsor-laden ("PANATHINAIKOS AKTOR
        # ATHENS", "LDLC ASVEL VILLEURBANNE"); Flashscore's short forms
        # ("Panathinaikos", "Villeurbanne") are already carried on the odds
        # record from the join. `Home Team`/`Away Team` stay canonical because
        # the odds join and the fixtures key on them — this is display only,
        # falling back to the canonical name when a fixture has no odds.
        short_h = odds_row.get("flashscore_home") or fx.get("home_team")
        short_a = odds_row.get("flashscore_away") or fx.get("away_team")

        ladder = odds_row.get("totals") or []
        pick = best_ev(pred_total, sigma, ladder) if (sigma and totals_ok) else None
        cf = counterfactual(pred_total, sigma, ladder) if (sigma and totals_ok) else None
        rows.append({
            "Date": date_str,
            "competition": comp,
            "Home Team": fx.get("home_team"),
            "Away Team": fx.get("away_team"),
            "Home Short": short_h,
            "Away Short": short_a,
            "Home ELO": int(feats.get("home_elo_pre", ELO_INIT)),
            "Away ELO": int(feats.get("away_elo_pre", ELO_INIT)),
            "Home Win Prob": round(cal_p, 4),
            "Home Win Prob (raw)": round(raw_p, 4),
            "Cal Source": cal_src or "",
            "Predicted Winner": "HOME" if cal_p >= 0.5 else "AWAY",
            "Predicted Total": round(pred_total, 2),
            "Total Sigma": round(sigma, 2) if sigma else "",
            "Over Line": over_line if over_line is not None else "",
            "P(Over)": round(p_over, 4) if p_over is not None else "",
            "P(Under)": round(1.0 - p_over, 4) if p_over is not None else "",
            # Staked pick (ladder, best EV) ...
            "Feature Gaps": len(gaps),
            "Totals Eligible": 1 if totals_ok else 0,
            "Bet Line": pick["line"] if pick else "",
            "Bet Side": pick["side"] if pick else "",
            "Bet Odds": pick["odds"] if pick else "",
            "Bet Book": pick["book"] if pick else "",
            "Bet Prob": round(pick["prob"], 4) if pick else "",
            "Bet EV": round(pick["ev"], 4) if pick else "",
            # ... and the main-line bet NOT placed, for the A/B.
            "Cf Line": cf["line"] if cf else "",
            "Cf Side": cf["side"] if cf else "",
            "Cf Odds": cf["odds"] if cf else "",
            "Cf Prob": round(cf["prob"], 4) if cf else "",
            "Cf EV": round(cf["ev"], 4) if cf else "",
            "Home Rest": feats.get("home_rest_days"),
            "Away Rest": feats.get("away_rest_days"),
            "gameId": fx.get("gameId"),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"predictions_euroleague_{date_str}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[predict] wrote {len(rows)} predictions → {out_path}")
    if len(df):
        print(df[["competition", "Home Team", "Away Team", "Home ELO", "Away ELO",
                  "Home Win Prob", "Home Win Prob (raw)", "Cal Source", "Predicted Total"]]
              .to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; default tomorrow")
    args = ap.parse_args()
    sys.exit(predict(args.date))
