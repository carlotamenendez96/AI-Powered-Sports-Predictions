# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All Python commands must run inside the project's `venv`. Most shell wrappers in `bin/` source `venv/bin/activate` automatically; when running Python directly, activate it first:

```bash
source venv/bin/activate
```

Scrapy spiders need Playwright browsers installed (`playwright install`) — this is a one-time setup separate from `pip install -r requirements.txt`.

`PYTHONPATH` needs to include both the repo root and `ml_project/` for the ML scripts to resolve imports. The `bin/run_predictions.sh` and `bin/run_nba_predictions.sh` wrappers export this; replicate it when invoking `ml_project/*.py` files directly:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ml_project
```

## Common Commands

| Task | Command |
| :--- | :--- |
| Football predictions (tomorrow) | `./bin/run_predictions.sh` |
| Football predictions for a date | `./bin/run_predictions.sh 2026-05-15` |
| Force-rescrape (ignore cached JSON) | `./bin/run_predictions.sh --force` |
| Verify yesterday's predictions | `./bin/run_verification.sh` |
| Full football retrain pipeline | `./bin/retrain_pipeline.sh` (data update → standings → train → fit calibrators → validate calibrators) — recommended cadence: **weekly** (see Operational cadence below) |
| Train football models only | `python3 ml_project/train_model.py` |
| Tune football hyperparameters | `python3 ml_project/tune_model.py` |
| Update standings/form only | `./bin/update_leagues_data.sh` |
| Download a season of football history | `./bin/setup_data.sh 2526` (season code = end year; back-compat with old positional invocation) |
| Full data-rollout for all sports | `./bin/setup_data.sh` (football + nba + euroleague + nt; idempotent, safe to re-run; `--sport <slug>` for one sport) |
| NBA predictions / verification / retrain | `./bin/run_nba_predictions.sh`, `./bin/run_nba_verification.sh`, `./bin/retrain_nba_pipeline.sh` |
| Live in-play snapshot (one-shot) | `python3 scripts/run_live_analysis.py` — also exposed in the UI as the "Refresh Live Snapshot" button |
| Cashout backtest harness | `python3 scripts/run_backtest.py --start YYYY-MM-DD --end YYYY-MM-DD` (CLI only; see `ml_project/backtest/`) |
| Evaluate auto-cashout decisions | `python3 scripts/evaluate_auto_cashout.py` (joins `output/auto_cashout_log.jsonl` to `verification_*.csv`: cash vs hold-to-settlement, net Δ; run after verification) |
| Start/stop web UI (Flask, port 5001) | `./bin/manage_server.sh {start\|stop\|restart\|status}` — logs to `logs/ui.log` |

Run a single spider manually:
```bash
scrapy crawl flashscore -O output/matches_YYYY-MM-DD.json -L WARNING -a filter_leagues=true -a day_diff=1
scrapy crawl flashscore -a live_ids="id1,id2,..." -a mode=verification -O <file>
scrapy crawl standings -L WARNING
scrapy crawl nba -O output_basketball/nba_matches_YYYY-MM-DD_final.json -L WARNING
```

There is no test suite. Treat correctness checks as: run the relevant pipeline end-to-end, then inspect `output/predictions_*.csv`, `output/report_*.txt`, or `output/verification_*.csv`.

## Architecture

The project is three loosely-coupled subsystems sharing a filesystem-based data layer (CSV/JSON files under `data_sets/`, `output/`, `output_basketball/`, `models/`). There is no database.

### 1. Scrapers — `flashscore_scraper/` (Scrapy + Playwright)

- `spiders/flashscore_spider.py` — daily football matches, 1X2 + O/U 2.5 odds, results. Modes:
  - **Date-based**: `-a day_diff=N` (relative to today: `+1` = tomorrow, `-1` = yesterday).
  - **ID-based**: `-a live_ids="..."` — scrape specific match IDs (used by verification to reuse the IDs captured during prediction).
  - `-a mode=verification` flips parsing to read final scores.
  - `-a filter_leagues=true` restricts to `data_sets/target_leagues.json`.
- `spiders/standings_spider.py` — league tables and form; outputs are written via `pipelines.StandingsPipeline`, so do not use `-O`.
- `spiders/nba_spider.py` — NBA equivalent (spider name = `nba`).
- `settings.py` sets `CONCURRENT_REQUESTS=4`, `DOWNLOAD_DELAY=1`, headless Chromium via `scrapy_playwright`. `ROBOTSTXT_OBEY=False`.
- `scripts/update_football_data.py` and `scripts/setup_historical_data.py` pull CSVs from football-data.co.uk — not Scrapy spiders, just HTTP downloaders feeding `data_sets/MatchHistory/`.

### 2. ML pipeline — `ml_project/`

Football flow (1X2 + Over/Under 2.5):
1. `data_loader.py` reads `data_sets/MatchHistory/**.csv`. When Bet365 odds are absent (some extra leagues), it falls back to `AvgC*` then `MaxC*` columns.
2. `elo_engine.py` builds ELO ratings across full history (~since 2010) with goal-margin K-factor; cached at `data_sets/elo_ratings.json`.
3. `feature_engineering.py` produces: implied probabilities (`IP_H/D/A`), ELO, rolling last-5 form (points/GF/GA/OU/shots/corners), season-to-date PPG and attack/defense strength vs league average (`H_ppg`/`A_ppg`/`H_att`/`A_att`/`H_def`/`A_def`/`ppg_diff`/`abs_ppg_diff`/`att_def_diff`), and home-only / away-only specific form.
4. `train_model.py` trains XGBoost models: multi-class 1X2, binary draw, and Poisson O/U 2.5. Time-series 5-fold CV. Models saved to `models/xgb_model_{1x2,draw,ou}.json`, with feature lists in `models/features_*.json`. **Note (2026-05-19)**: the binary draw model is **trained but no longer used at inference**. It used to be averaged into the 1X2 multi-class output as a "second opinion" on draws, but investigation showed it was acting as an implicit base-rate regularizer — exactly what the per-league Platt calibrator (Phase C4) handles in a more principled, league-specific way. The model file is preserved on disk for backward compatibility and possible future reactivation, but `predict_matches.py` no longer loads or calls it. See FOOTBALL_NEXT_STEPS D0 for full reasoning. `tune_model.py` runs a 6-stage stepwise hyperparameter search and writes `models/best_params_*.json`. **Calibration refresh** (Phase C5): `bin/retrain_pipeline.sh` automatically chains `scripts/run_fit_calibration.py` (refit Platt per league × market, ~6 min) and `scripts/run_validate_calibration.py` (chronological-holdout validate, auto-filter entries that regress >5%, ~6 min) after the model retrain, so `data_sets/league_calibration.json` stays in sync with the model. Both calibration steps are non-fatal — if either fails, the prior calibration file remains in place. **Model registry / estimator seam (2026-06-23)**: the model *family* for each football head is decoupled from the pipeline behind `ml_project/model_registry.py`. `REGISTRY[market][family]` (markets `1x2`/`ou`/`draw`) returns a `ModelSpec` whose `build()` yields a fresh estimator implementing a uniform contract — `predict_proba` for `1x2`/`draw`, `predict`→Poisson λ for `ou`. `train_model.py`, `predict_matches.py`, and `scripts/benchmark_models.py` all go through this seam and never construct XGBoost directly. Families: 1X2 `xgboost`/`logreg`/`rf`; O/U `xgboost`/`poisson_glm`; draw `xgboost`/`logreg`. Pick a family per head via `ModelTrainer(..., model_family=, ou_family=, draw_family=)` or the env vars `MODEL_FAMILY_1X2` / `MODEL_FAMILY_OU` / `MODEL_FAMILY_DRAW` (each defaults to `xgboost`, which is **byte-identical** to the pre-seam path — verified by diffing model files + prediction CSVs). Training writes a `models/model_meta_<market>.json` sidecar (`{family, artifact, market}`); XGBoost saves its native JSON, other families `joblib.dump` to `models/sk_model_<market>_<family>.joblib`. `predict_matches.py` loads each head via `load_1x2_model` / `load_ou_model`, which read the sidecar and fall back to the legacy `xgb_model_<market>.json` when none exists (so a pre-seam checkout is unchanged). `league_cat` is XGBoost-only categorical; non-XGBoost families drop it from the feature list, and `known_leagues` is decoded from whichever served head is XGBoost (1X2 preferred, O/U fallback) so a mixed-family setup still handles categories correctly. **Caveat**: swapping the production model invalidates `league_calibration.json` (fit against the deployed model's OOF preds) — roll real model changes through `bin/retrain_pipeline.sh` so calibration refits in the same run. `scripts/benchmark_models.py` runs a cross-model value-bet backtest (5-fold TimeSeriesSplit, EV-gated flat stakes) with `implied`/`prior` baselines; writes `output/benchmarks/<ts>.{json,txt}`.
5. `predict_matches.MatchPredictor` (invoked by `run_predictions.sh`) loads models, fetches/derives features for the scraped upcoming-matches JSON, and writes `output/predictions_YYYY-MM-DD.csv`. Season-to-date PPG/strength at inference is mirrored from the standings JSON via `HeuristicAdjuster.get_team_strength` to avoid train/serve skew. **Calibration step (Phase C4)**: between the raw model output and the heuristic adjuster, `MatchPredictor` calls `apply_platt_1x2 / apply_platt_ou` (`ml_project/calibration/apply.py`) which looks up per-league Platt scaling parameters from `data_sets/league_calibration.json`. Leagues without a calibrator (or when the per-sport `use_league_calibration` flag in `betting_config.json` is `false`) pass raw probs through unchanged. `predictions_*.csv` carries both: `Home Win %` / `Draw %` / `Away Win %` / `Over %` / `Under %` are the calibrated-then-heuristically-adjusted final values, and `Home Win % (raw)` / `Draw % (raw)` / `Away Win % (raw)` / `Over % (raw)` / `Under % (raw)` are the model's pre-calibration output for audit. Two extra columns — `Cal 1X2 Source` and `Cal O/U Source` — show whether the calibrator came from full-features OOF (`full`), minimal-features OOF (`minimal`), or no calibrator was applied (empty). Flashscore-format league names map to calibration keys via `ml_project/calibration/league_aliases.py`; add new entries there to enable calibration for newly-scraped leagues.
6. `heuristic_adjuster.py` post-processes raw model probabilities with league-aware draw calibration + cap, rank-gap boosts, symmetric form-momentum (winning streak → boost; losing streak → split fade between opponent and Draw via `_fade()`), heating/cooling trend, and a goal-fest O/U boost. All H1–H6 deltas are accumulated and capped at `MAX_TOTAL_BOOST_PER_CLASS = 0.15` per outcome before being applied. Probabilities re-normalized; the *adjusted* confidence drives the final pick. See `docs/training_process.md` for the heuristic table. **Note**: post-C4, the heuristic's "league-aware draw calibration + cap" partially overlaps with the upstream Platt calibration — running them in series is currently safe, but worth revisiting in C6 (FOOTBALL_NEXT_STEPS) whether one or both should be tuned together.
7. `resolve_daily_bets.py` (called by `run_verification.sh:96`) settles open bet slips in `output/bets_*.json` against scraped results. **Note**: `ml_project/betting_engine.py` is legacy/reference code — the *active* bet-placement and settlement flow lives in `web_ui/app.py` (`/auto_wager`, `/place_bets`, `process_bet_verification`).
8. `live_adjuster.py` applies in-play heuristics on top of pre-match model output. Two adjusters: `adjust_probabilities()` for 1X2 (shots/xG/possession/dominance + time decay) and `adjust_ou_probabilities()` for Over/Under 2.5 (Poisson goal model from observed xG pace, blended with the pre-match Over % via a sigmoid centred at minute 30). Invoked by `scripts/run_live_analysis.py` (one-shot, user-triggered via the dashboard's **Refresh Live Snapshot** button or auto-triggered every 10 min when the dashboard's "Auto 10m" checkbox is on and the tab is visible). Each refresh writes the latest snapshot to `output/live_data.json` (consumed by the UI) AND appends one JSON line per scraped match to `output/live_history/live_history_<YYYY-MM-DD>.jsonl` (append-only, feeds the cashout backtest harness). The history lives in its own `output/live_history/` subdir to keep it visibly separate from the throwaway `live_data.json` the UI's Clear button zeroes — it's accruing data the weekly backtest depends on, never auto-deleted.

**Auto-cashout (functionality test, 2026-05-26)**: a dashboard **"Auto-cashout" checkbox** arms **server-side autonomous** auto-cashout — it does NOT depend on a browser tab. The checkbox POSTs to `POST /football/auto_cashout/arm` (`on=1|0`), persisting `output/auto_cashout_armed.json`. A daemon thread `_auto_cashout_scheduler()` (started in `__main__`, so it runs under `bin/manage_server.sh`) refreshes Flashscore (reusing `_launch_live_refresh()`) then runs `_run_auto_cashout_sweep()` every `_AUTO_CASHOUT_INTERVAL_S` (10 min) while armed. The sweep evaluates every OPEN bet on a live match and fires the lane-cascading `VirtualBettingBackend.execute_cashout` on any whose decision is non-`hold`. `POST /football/auto_cashout` runs one sweep on demand (diagnostic). The browser "Auto 10m" checkbox is now just a UI-refresh convenience, independent of cashout. The decision is `_cashout_decision()` in `web_ui/app.py` — the **single source of truth** shared with the display badge. It's driven by `adj_prob` (the LiveAdjuster's synthesis of the live stats — score, minute, xG, shots, possession, dominance, red cards — so the decision *is* stats-based without re-reading raw stats): `lock_in` if in profit AND (`adj_prob ≥ _AUTO_CASHOUT_LOCK_IN_PROB` (0.85, near-certain, odds-independent) OR `fair/stake ≥ _AUTO_CASHOUT_LOCK_IN_RATIO` (1.5, big unrealized profit)); `stop_loss` if `adj_prob < _AUTO_CASHOUT_STOP_LOSS_PROB` (0.20); else `hold`. A minute floor (`_AUTO_CASHOUT_MIN_MINUTE`, 30) suppresses both before in-play stats are reliable. (The probability branch exists because `fair/stake = odds×adj_prob×0.95` is odds-capped — a ≥1.5× rule alone can never lock in a bet at odds <~1.58, even at near-certain win.) **Prices at the synthetic estimate** (`get_cashout_amount` = `stake × odds × adj_prob × 0.95`), so it tests cashout TIMING/MECHANISM, not real bookmaker economics — **virtual money only, no real bet placed**. Every evaluation (fired, would-have-fired, or held) is appended to `output/auto_cashout_log.jsonl` for audit/threshold-tuning. The auto loop is Flashscore-only (never triggers Pamestoixima). See FOOTBALL_NEXT_STEPS phase 7c.

**Shadow mode (2026-09-10)**: a second **"Shadow" checkbox** next to Auto-cashout controls whether armed sweeps actually fire. In shadow, the sweep evaluates and logs exactly as before but calls no `execute_cashout` — bets stay `OPEN`, no bankroll moves. State lives alongside arming in `output/auto_cashout_armed.json` as `{"armed": bool, "shadow": bool}`; **a missing `shadow` key defaults to `true`**, so a sweep never starts moving money on its own. Rationale: cashout is priced at fair value (`stake × odds × adj_prob × 0.95`), which is EV-negative by exactly that haircut — it reduces variance, not expected loss (measured: −€26.12 over 64 executed cashouts, of which −€25.44 was `stop_loss`; `scripts/evaluate_auto_cashout.py`). Shadow accrues decision data at zero cost. Log entries carry `shadow`, `would_fire`, and — recorded but never used to price or decide — `bookmaker_offer` / `bookmaker_age_s` from `_load_bookmaker_offers`, which is the data needed to test whether `adj_prob` beats the real live price. Because nothing fires, a shadowed bet stays `OPEN` and is re-logged every sweep, so **scoring takes the FIRST non-`hold` entry per `bet_id`** as the would-have-fired moment; `evaluate_auto_cashout.py` dedupes accordingly and reports executed and shadow decisions in separate sections. Both `on` and `shadow` are **optional** on `POST /football/auto_cashout/arm` and an omitted field is preserved — each checkbox posts only the field it owns, so a stale tab can't silently re-enable real firing. `POST /football/auto_cashout` accepts `shadow=0|1` to override for one sweep. **Startup disarms real firing only** (`__main__`): shadow arming survives restarts on purpose — the unconditional disarm is what stalled live collection between 2026-06-14 and 2026-09-10.

**Per-lane bankroll impact (2026-09-10)**: `evaluate_auto_cashout.py` reports how each lane's bankroll moved (executed) or *would have* moved (shadow) — the "what if live cashout were on" question. Necessary because the audit log stores only ONE representative bet per `bet_id` while `execute_cashout` cascades across every lane holding it, so **the log's headline understates total money movement**: over the 64 historical cashouts the representative-only net Δ is −€26.12, but the true across-lane total is **−€48.62** (value −€29.26, conviction −€12.12, model −€7.24). `_load_lane_bets()` recovers each lane's own stake/odds/`cashout_amount` from the slips; `_lane_impact()` prefers the **recorded** `cashout_amount` as ground truth and only scales the representative's amount by stake share when none exists (the shadow case). Scaling is exact for shadow — a cascade prices every lane at one instant off the same `adj_prob`, and price is linear in stake — but **invalid for the historical executed rows**, whose lanes were cashed at different times pre-cascade (e.g. `2026-05-20:freiburg:aston_villa:o_u:over_2_5`: value cashed 22:58, model 08:23 next morning, ratios 1.2578 vs 1.0859). The report prints per-lane `recorded`/`scaled` counts so the basis is always visible.

**Live-adjuster tracing (2026-09-10)**: `LiveAdjuster.adjust_probabilities` / `adjust_ou_probabilities` take an optional `trace=dict`; when passed it records the branch taken, the dominance score, and the probability vector after each of the four 1X2 stages (`time_decay` → `dominance` → `sterile_possession` → `red_card`), or the pace/blend internals for O/U. Results are byte-identical with and without a trace, and existing callers (`backtest/simulator.py`) are unaffected. `run_live_analysis.py` passes traces and persists them to `live_history` as `trace_1x2` / `trace_ou`, alongside `priors_synthesized` (previously present in the UI record but **dropped** from the persisted history, so lower-quality synthesized-prior snapshots were indistinguishable in stored data). Without the trace only the before/after was stored and per-stage attribution was unrecoverable. Note `LiveAdjuster` emits no log strings of its own — the `adj_logs` in `predictions_*.csv` come from the separate `HeuristicAdjuster` in the prediction path.

**Live-window gating (2026-09-10)**: the scheduler used to scrape Flashscore every 10 min around the clock while armed. `_live_window_active()` now gates each tick on whether any OPEN bet's match is plausibly in play — from its kickoff (`date`, `"YYYY-MM-DD HH:MM"`) until `_LIVE_MATCH_WINDOW_S` (150 min: 90' + half-time + stoppage + buffer) after it. On a typical slate this cuts ~144 scrapes/day to ~59, because a day leaves a long dead stretch (2026-09-10: an 11.8-hour gap between the early matches ending ~08:00 and the evening kickoffs at 19:50) that a "start at the earliest kickoff" rule would scrape straight through. **Fails OPEN**: OPEN bets with no parseable kickoff return `True`, since silently suppressing collection is worse than a wasted scrape. The scheduler logs only on window transitions, not every tick. The manual `POST /football/auto_cashout` is ungated — it's a diagnostic and always runs.

NBA pipeline lives under `ml_project/nba/`. Reactivated 2026-05-28 (Phase 1+2 — data layer + enhanced model; Phase 3 = UI/betting integration on a separate branch).

**Data layer** (replaces the pbpstats/stats.nba.com path — geo-blocked for advanced data from this machine):
1. **Historical corpus** comes from a **local archive** (`data_sets/NBA/archive/`, gitignored — Kaggle-style snapshot with `Games.csv`, `TeamStatistics(Extended).csv`, `PlayByPlay.parquet`, etc.). `process_archive.py` reads `TeamStatisticsExtended.csv`, filters to **2000+** competitive games (drops All-Star), and writes the canonical long-format corpus `data_sets/NBA/team_game_stats.csv` (one row per team per game; ~32k games × 27 seasons × 30 teams). **Merge-safe**: a re-run preserves any daily-only appended rows.
2. **Daily refresh** uses `nba_api` (proper headers; `data.nba.com` works, raw `stats.nba.com` requests don't). `fetch_nba_daily.py` has two modes — `append-results` (yesterday's finished games via `LeagueGameLog`, idempotent dedup keeping the archive's richer rows) and `fixtures` (tomorrow's schedule via `ScoreboardV3` — V2 has known 2025-26 issues). `time.sleep(1)` before every API call.

**Model** (`nba_feature_engineering.py` → `train_nba_models.py` → `nba_calibration.py`):
- Features (~40, all leakage-free / shifted, in `data_sets/NBA/training_data.csv`): rolling **L5** (pts/allowed/win + FG%/3P%/FT% + reb/ast/tov/plus_minus) + **L10** (subset) + **venue-matched L5** (team's last-5 home games when at home, last-5 away when on the road) + **rest_days** / **back-to-back** + per-game **ELO_pre** (K=20, home_adv=100, cached at `data_sets/NBA/nba_elo.json`).
- Two models: winner = `XGBClassifier` on `home_win`; total = `XGBRegressor` on `total_points`. 5-fold `TimeSeriesSplit` CV. Saved to `models/nba/{winner,total}_model.pkl` with feature manifests `models/nba/features_{winner,total}.json` so the predictor reads exactly the columns the trainer wrote.
- **Calibration** (`nba_calibration.py`): single global Platt on `P(home_win)` (NBA = one league); `apply_home_win_platt` returns the `(prob, applied, source)` 3-tuple — same contract as football's `apply_platt_1x2`, so Phase 3 can be sport-agnostic. Fit OOF; saved to `data_sets/NBA/nba_calibration.json`. Total regressor: diagnostic only (well-calibrated, no Platt fit needed).

**Predictor** (`predict_nba.py`): reads `data_sets/NBA/fixtures_<date>.json` (from `fetch_nba_daily.py fixtures`), computes serve-time features from the **same corpus** the model trained on (mirrors `nba_feature_engineering`'s shift+rolling exactly — **kills the train/serve skew** the old Flashscore-standings serve path had), applies Platt calibration if present, writes `output_basketball/predictions_nba_<date>.csv` carrying `Home Win Prob` (calibrated) + `Home Win Prob (raw)` + `Cal Source` for audit. Odds / EV / Kelly are Phase 3 (the betting flow).

**Odds** (`fetch_espn_odds.py`, rewritten 2026-05-28): plain-HTTP JSON client over ESPN's public scoreboard API (`site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD`). Replaces the brittle Playwright/DOM scraper. Pulls moneyline + spread + total + per-side juice, picks a preferred provider (ESPN BET → DraftKings → FanDuel → …), American→decimal converted. Writes per-date `output_basketball/espn_odds_<date>.json` (no more singleton overwrite). Wired as a non-fatal step in `run_nba_predictions.sh` between fixtures and predict — the predictor itself works without odds, Phase 3 joins odds to predictions for EV/Kelly. **Caveat**: ESPN doesn't preserve historical odds (`odds: []` for completed games), so this is a forward-only source.

**Retired** (under `ml_project/nba/legacy/`): `fetch_nba_history_stats.py` (pbpstats), `fetch_nba_results.py` (broken ESPN scrape), `fetch_nba_stats_tables.py` (Flashscore standings — no longer needed; features come from the corpus). `pbpstats` dropped from `requirements.txt`, replaced by `nba_api==1.11.4`. The Flashscore `nba` spider is preserved in-tree as an unwired fallback. `evaluate_nba_predictions.py` and `tune_nba_models.py` are kept but unwired in v1 (evaluator needs CSV-schema updates for the new columns; tuner is deferred until the new feature set has stabilized for a couple of cycles).

Bin scripts (`run_nba_predictions.sh`, `run_nba_verification.sh`, `retrain_nba_pipeline.sh`) export `PYTHONPATH=...:ml_project/nba`, use portable date handling, and are `set -u` safe. `retrain_nba_pipeline.sh` chains `process_archive → features → train → calibrate` (calibration step non-fatal, mirroring football's pipeline).

**NBA UI is currently DETACHED** — the blueprint code lives at `web_ui/nba/routes.py` (template at `web_ui/templates/nba/index.html`), but `app.py` does not register it. Old `output_basketball/` artifacts have been soft-archived to `output_basketball/history/`. To reactivate the NBA UI: re-add `from nba.routes import nba_bp, NBA_TASKS` and `app.register_blueprint(nba_bp, url_prefix='/nba')` in `web_ui/app.py`, plus restore the nav link in `web_ui/templates/layout.html`.

`entity_resolver.py` + `team_mapping.py` + `data_sets/team_mappings.json` handle name reconciliation between Flashscore and football-data.co.uk (fuzzy match via `rapidfuzz`/`thefuzz`).

### 3. Web UI — `web_ui/app.py` (Flask, port 5001)

Multi-sport-aware. Run as a backgrounded process via `bin/manage_server.sh` — direct `python3 web_ui/app.py` works but won't daemonize.

**URL structure:**
- `/` — sport-picker landing page. Lists everything in `SPORTS = [...]` (defined near the top of `app.py`); active sports are clickable cards, dormant ones are greyed out. Also renders a **Portfolio Summary table** aggregating bets / stake / P/L / ROI / bankroll across every sport.
- `/football/*` — football blueprint (`football_bp`). Every football route — dashboard, betting, predict, verify, retrain, place_bets, auto_wager, view, delete_file, refresh_live, etc. — lives here.
- `/nba/*` — NBA blueprint (currently DETACHED; code at `web_ui/nba/routes.py`, see "NBA reactivation" below).
- `/status`, `/stop/<task>`, `/server/<action>` — sport-agnostic, registered directly on `app`.

**Theme (light + dark)**: visual styling lives in `web_ui/static/css/theme.css`, loaded after Bootstrap 5.3 in `layout.html`. Overrides Bootstrap's semantic CSS custom properties (`--bs-primary`, `--bs-warning`, `--bs-info`, `--bs-success`, `--bs-danger`, plus surface tokens `--bs-body-bg`, `--bs-card-bg`, `--bs-border-color`) with a calmer palette. Dark mode rides Bootstrap 5.3's native `data-bs-theme` attribute: an inline `<script>` in `<head>` reads `localStorage['theme']` (falls back to `prefers-color-scheme: dark`) and sets the attribute *before* CSS evaluates so there's no flash. A 🌙/☀️ toggle in the navbar flips and persists choice. OS-level theme changes are reflected live while no manual override is stored. Custom utility classes added by the theme: `.bg-lane-value` / `.border-lane-value` (soft lavender for the Value lane so it reads distinct from the Model lane's sky blue) and `.acc-badge` (forces dark text on the Cumulative League Performance % badges in dark mode only). The navbar uses `sticky-top` so it stays pinned while scrolling.

**Adding a new sport** (forward-compatible by design):
1. Create `web_ui/<sport>/routes.py` with a `Blueprint('<sport>', __name__)` and the routes you want.
2. Add an entry to `SPORTS` in `app.py`: `{'slug': '<sport>', 'label': '...', 'icon': '...', 'active': True, 'tagline': '...'}`.
3. Import + register: `from <sport>.routes import <sport>_bp` then `app.register_blueprint(<sport>_bp, url_prefix='/<sport>')`.
4. The landing page card and navbar Sport ▾ dropdown pick it up automatically.

**NBA reactivation** (next NBA season): in `web_ui/app.py` flip `nba` entry's `active: False` → `True`, uncomment the `from nba.routes import nba_bp, NBA_TASKS` and `app.register_blueprint(nba_bp, url_prefix='/nba')` lines.

**Betting flow** (lives in `football_bp`, all under `/football/...`):
- `/football/auto_wager` reads the latest `output/predictions_*.csv` and builds three parallel slips:
  - **Value lane** — EV-gated, Option B sizing (`value_bankroll × EV × Conf × stake_multiplier`), per-bet cap 3%.
  - **Conviction lane** — Conf ≥ 0.65 AND odds ≥ 1.40 (EV ignored), flat 0.5% of conviction bankroll.
  - **Model lane** — broad coverage: stakes every prediction with `model_bankroll × model_base_pct × Conf × ev_factor` where `ev_factor = clamp(Conf × odds, 0.5, 1.5)`. Has its own lower min-stake floor (`model_min_stake_eur`, default €1) so wide coverage isn't killed by the shared €2 floor.
  - Each lane has its **own bankroll** (independent buckets) and its **own daily exposure cap** (defaults: value 10%, conviction 10%, model 15%). Per-session overrides via query params `bankroll_<lane>` / `cap_<lane>` (cap as 0–1 fraction). When a lane's stakes exceed its cap, that lane scales pro-rata; the other lanes are untouched.
- `/football/place_bets` writes a combined slip to `output/bets_<date>.json` with each bet tagged `lane: 'value' | 'conviction' | 'model'` and a `stake_by_lane` summary. Debits each lane's bankroll separately. The slip preview has a per-bet **"Live" checkbox**; ticked bets persist `mark_for_real: true` on the saved bet and show a ⚡ live badge in the slip history. `mark_for_real` records *intent only* — it does not place a real bet.
- `/football/place_real_bets` is a **DORMANT** route (added 2026-05-25). The "⚡ Place Marked as Real" button POSTs the ticked bets here; it logs + reports them and returns `{placed: 0, dormant: true}` — places nothing real. It's the designated dispatch point for the future real-betting flow (would call into `real_betting/` behind a confirmation modal + per-bet stake caps + `EXECUTE_*` gating). Real placement is out of scope per `FOOTBALL_NEXT_STEPS.md` until re-evaluation.
- `process_bet_verification` (called after a verification CSV is produced) settles bets and credits returns **per lane** back to that lane's bankroll. Stores `return_by_lane` and `pnl_by_lane` on the closed slip. **Only looks in `output/`** — archived slips will not settle.
- `/football/betting` page shows a three-row Strategy Comparison table ("Strategy Comparison · Football (cumulative)"). Aggregation logic lives in `compute_sport_summary(bets_dir)` (module-level in `app.py`); the same helper feeds the landing page's Portfolio Summary table.
- `/football/delete_file/<filename>` is a **soft delete**: moves the file to `output/history/`. The Archive button only appears on CLOSED slips so OPEN slips can't be archived before settlement.

**Bet-status taxonomy** (per-bet `status` field inside `output/bets_<date>.json`):
- `OPEN` — placed but match not yet settled. Eligible for cashout (Phase 7) or normal settlement. **If the match doesn't appear in the verification CSV (because it hasn't finished yet), the bet STAYS `OPEN` across verification runs** — settlement is idempotent and partial-friendly. Earlier behaviour wrongly marked these as VOID, which prevented re-running verification later.
- `WON` / `LOST` — set by `ml_project/resolve_daily_bets.py:resolve_all_bets` (the canonical settlement path, invoked by `bin/run_verification.sh` at the end of the verification flow). `pnl` reflects payout − stake or −stake; `profit` is kept as a backward-compatible alias. Once set, re-running settlement leaves the bet alone (no double-credit).
- `VOID` — for truly canceled / postponed matches that will never settle. Set via the **⊘ Void button** on the `/football/betting` page (OPEN bets only). Backend: `VirtualBettingBackend.void_bet` refunds the stake to the lane bankroll and stamps `status='VOID'`, `result='VOID'`, `pnl=0.0`, `voided_timestamp=<iso>`. `process_bet_verification` leaves VOID bets alone on subsequent runs.
- `CASHED_OUT` — bet manually cashed out via Phase 7 endpoint. Carries `cashout_amount` (bookmaker payout), `cashout_timestamp`, and `pnl = cashout_amount − stake`. **Bankroll is credited at cashout time** (inside `VirtualBettingBackend.execute_cashout`), NOT at settlement. `resolve_all_bets` skips CASHED_OUT bets for re-credit but includes them in the recomputed slip totals (Phase B of its two-pass loop). **Cashout cascades across lanes**: a single click cashes out every OPEN bet sharing the same bet_id (same conceptual wager held in multiple lanes — e.g. value + model both on Over 2.5). Each lane's bet gets its own per-bet `cashout_amount` and its own lane bankroll credit. Same cascading behaviour applies to `void_bet`. `process_bet_verification` *includes* CASHED_OUT bets in `return_by_lane` / `pnl_by_lane` for reporting parity but tracks `cashed_out_already_credited` and subtracts it from the per-lane bankroll update so the amount isn't credited twice. `compute_sport_summary` aggregates them via a separate `cashed_out` counter.

**Tunables** live in `data_sets/betting_config.json`, **sport-keyed**, **lane-aware**:

```json
{
    "sports": {
        "football": {
            "bankrolls": {
                "value":      {"current": 1000.0, "initial": 1000.0},
                "conviction": {"current": 1000.0, "initial": 1000.0},
                "model":      {"current": 1000.0, "initial": 1000.0}
            },
            "min_confidence": 0.45, "stake_multiplier": 0.4,
            "min_stake_eur": 2.0, "max_stake_pct": 0.03,
            "conviction_min_confidence": 0.65, "conviction_min_odds": 1.4, "conviction_stake_pct": 0.005,
            "model_base_pct": 0.005, "model_max_stake_pct": 0.015, "model_min_stake_eur": 1.0,
            "model_ev_factor_min": 0.5, "model_ev_factor_max": 1.5,
            "value_max_daily_exposure_pct": 0.10,
            "conviction_max_daily_exposure_pct": 0.10,
            "model_max_daily_exposure_pct": 0.15
        }
    }
}
```

Each sport has three independent lane bankrolls (no cross-contamination between lanes or sports) and its own tunables. Adding a fourth lane = new entry under `bankrolls`, new entry in `LANES` in `sports_config.py`, and a new builder in `/auto_wager`.

**All bankroll/config access goes through `web_ui/sports_config.py`** — never read or mutate the JSON directly. The schema lives only in this module:
- `get_sport_config(slug)` — full per-sport config dict (defaults merged with overrides, includes `bankrolls`).
- `get_bankroll(slug, lane='value')` / `update_bankroll(slug, delta, lane='value')` — atomic lane-scoped mutations.
- `lane_bankrolls(slug)` → `{lane: current}` for one sport.
- `sport_total(slug)` → sum across lanes.
- `all_bankrolls()` → `{sport: sport_total}` (flat per-sport view for navbar/landing).
- `all_lane_bankrolls()` → `{sport: {lane: current}}` (detailed breakdown).
- `total_bankroll()` → sum across every lane of every sport.

The legacy flat keys (`base_unit`, `confidence_threshold_*`, `max_kelly_fraction`, `ev_threshold`, `league_performance_threshold`, `min_matches_for_stats`) used to live at the root for the orphan `betting_engine.py`; dropped in the per-sport migration. The orphan module is fully dead.

### Data layout cheatsheet

- `data_sets/MatchHistory/` — raw historical CSVs (one per league/season). Source of truth for training.
- `data_sets/standings/` — JSON files written by `standings_spider` (current standings + form tables).
- `data_sets/target_leagues.json` — whitelist for the daily match filter.
- `data_sets/elo_ratings.json`, `league_analytics.json`, `team_mappings.json` — derived/config caches.
- `data_sets/betting_config.json` — bankroll + strategy tunables (see Web UI section).
- `data_sets/bets.json` — present but unused (read by orphan `betting_engine.py`).
- `output/matches_<date>.json` — scraper output (predictions or results, depending on mode).
- `output/predictions_<date>.csv`, `output/verification_<date>.csv`, `output/report_<date>.txt` — prediction artifacts.
- `output/bets_<date>.json` — placed bet slips (active). Each bet has a `lane` tag.
- `output/live_data.json` — latest live-snapshot results (overwritten each refresh; what the UI reads).
- `output/live_history/live_history_<date>.jsonl` — append-only per-match snapshots from every Refresh Live Snapshot run (own subdir; the UI Clear button never touches it). One JSON object per line. Schema: `ts, date, match_id, home_team, away_team, league, minute, score, stats, pre_probs, adj_probs, pre_ou_probs, adj_ou_probs`. `stats` carries xg, xgot, possession, shots, shots_on_target, shots_inside_box, shots_outside_box, big_chances, corners, touches_opp_box, woodwork, saves, yellow_cards, red_cards, fouls (each `_home` and `_away`; absent stats omitted, not nulled). Source data for cashout backtesting and adjuster v2 work. **Red cards feed `LiveAdjuster._apply_red_card_modifier`** — net red cards shift 1X2 probability away from the man-down team toward the opponent (~65%) and draw (~35%), scaled by remaining time, capped at `MAX_RED_SHIFT`. Shown as a 🟥 badge on the live dashboard / live-analysis rows.
- `output/backtests/<timestamp>.{json,txt}` — cashout backtest outputs from `scripts/run_backtest.py`. JSON has raw outcomes + aggregate by `(rule, lane)`; .txt is the pretty-printed report. Engine lives in `ml_project/backtest/` (`trajectories.py`, `simulator.py`, `rules.py`, `report.py`). Built-in rules: `null` (self-validation, must equal stored P/L), `lock_in_profit`, `stop_loss`, `late_drift`, and `momentum_fade` (EXPERIMENTAL stub — trajectory pace/slope via `ctx['history']`; untuned, gated on data accrual per FOOTBALL_NEXT_STEPS phase 8d). Both 1X2 and O/U bets are evaluated — see `LiveAdjuster.adjust_ou_probabilities` (Poisson goal model from observed xG pace, blended with the pre-match Over % via a sigmoid centred at minute 30).

### 4. Real-betting integration — `real_betting/` (DORMANT)

Playwright-driven bookmaker automation. Currently scoped to **read-only operations** against `pamestoixima.gr` (OPAP). Bet placement is officially out of scope per `FOOTBALL_NEXT_STEPS.md` until a separate re-evaluation. **Several end-to-end tests have shipped** (audit dumps under `output/real_betting/`, all gitignored — local-only):

- 2026-05-20 — single €10 bet on Freiburg vs Aston Villa (`dryrun_freiburg_villa.py`).
- 2026-05-22 — cashout commit on Machida vs Urawa (`dryrun_cashout_discovery.py`).
- 2026-05-25 — €2+€2 scenario-#5 batch placement (`dryrun_batch_placement.py`) + fixture discoverer (`discover_fixtures.py` + `find_fixture_url()` helper) + open-bets scraper (`read_open_bets.py` writing `output/real_betting/open_bets_snapshot.json`).

The working selectors, DOM structures, anti-patterns, and corrections (e.g. "placement success signal is the receipt overlay, NOT slip-empty"; Pamestoixima ↔ Flashscore use different `match_id` schemes — fuzzy team-name join is the actual path) are preserved in **`real_betting/PAMESTOIXIMA_NOTES.md`**. Start there before adding any new Pamestoixima-driving code.

CLI surface (`python -m real_betting --help`): `set-credentials`, `login`, `discover-fixtures`, `read-open-bets`, plus the three `dry-run-*` one-shots for placement / cashout / batch tests.

**Bookmaker-cashout integration into the live UI** (scenario #3 from `real_betting/test_case_scenarios.md`, shipped 2026-05-25):
- Snapshot file: `output/real_betting/open_bets_snapshot.json` (latest-wins) + append-only `open_bets_history.jsonl`.
- Consumer: `_load_bookmaker_offers` + `_match_offer_by_teams` + `_attach_open_bets` in `web_ui/app.py`. Joins by Pamestoixima `match_id` first, then fuzzy team-name fallback.
- Per-sport flag in `data_sets/betting_config.json`: `cashout_source: 'synthetic' | 'bookmaker'` (default `'synthetic'`; flipped to `'bookmaker'` for football 2026-05-25).
- **Two freshness windows** (decoupled): the `🔗 linked` badge persists once established (stamped onto the bet as `linked_to_bookmaker` + `pamestoixima_uuid`) and lasts until the bet resolves — "link until resolved", not a timer. The real €offer *value* shows whenever the bet is linked (no separate short cap — the on-disk value only changes on a manual re-scrape), with the snapshot age shown on the badge (`real · 30m`). `cashout_snapshot_max_age_s` (default 600 s) still bounds the initial link *establishment*; `_BOOKMAKER_LINK_MAX_AGE_S` (4 h in `app.py`) bounds how old a snapshot can be to first establish a link.
- UI badges in `_open_bets_fragment.html`: green `real · <age>` / grey `est` (cashout value source), green `🔗 linked` (matched bookmaker record exists).
- Manual `Refresh Live Snapshot` POSTs to `/football/refresh_live?with_bookmaker=1`, which passes `--with-bookmaker` to `run_live_analysis.py`. That script chains the `read-open-bets` scrape **only if a live match intersects an OPEN bet** (skips the ~25s headed scrape when nothing relevant is live; Pamestoixima then runs after Flashscore, not in parallel). The button's JS (`triggerLive`) must fetch the form's own `action` URL so the `?with_bookmaker=1` param survives. Auto-5m stays Flashscore-only (headless Pamestoixima is **Akamai-blocked** — returns "Access Denied"; real-betting step 6d ⛔, see `PAMESTOIXIMA_NOTES.md`).
- `/football/live_analysis` standalone page filters to bets with `linked_to_bookmaker=True` only (focused "skin in the game" view); dashboard keeps the full listing.

## Roadmap

Cashout feature is built in phases. See **`FOOTBALL_NEXT_STEPS.md`** at the repo root for the current state of each phase and the data-accrual wait that gates phases 3, 6, 7. Update that file when phases complete.
- `output/history/` — soft-delete destination. Files moved here are hidden from UI lists but still counted by `/betting` Strategy Comparison stats.
- `output_basketball/` — NBA artifacts: `predictions_nba_<date>.csv` (from `predict_nba.py`); the old slate is soft-archived under `output_basketball/history/`. (Pre-Phase-3: predictions only; once the betting flow lands, this will also carry `bets_<date>.json` etc., paralleling football's `output/`.)
- `models/` — trained XGBoost JSON / sklearn pickle artifacts and tuned hyperparameters.
- `logs/` — pipeline, scraper status, UI logs.

### Operational cadence (suggested)

Different parts of the pipeline benefit from different cadences:

| Task | Cadence | Why |
| ---- | ------- | --- |
| `./bin/run_predictions.sh` | **daily** (typically the night before, for next-day fixtures) | Predictions are produced for matches happening tomorrow. |
| `./bin/run_verification.sh` | **daily** (after yesterday's matches finish) | Settles bet slips and updates `league_analytics.json`. |
| `./bin/update_leagues_data.sh` | **daily before predictions** | Standings + form are inputs to feature engineering at inference time. |
| `./bin/retrain_pipeline.sh` | **weekly** | Steps 1–2 refresh fresh CSV results (~50–100 new matches/day, ~0.6% training-set growth/week — too slow to matter daily, fast enough that monthly lags). Steps 3–5 retrain the model + refit calibrators, ~20–30 min total. Reasonable to run on a fixed day (e.g., Monday morning), or any time training data has materially grown. |
| `python3 ml_project/tune_model.py` | **every 3–6 months** | Hyperparameters are stable; tuning takes ~30–60 min and grids are coarse. Pulling forward when training data has grown >50% since the last tune (e.g., end of season, new league imports). |

The minimum-viable run cadence is: daily predict/verify, weekly retrain, quarterly tune. Skipping the weekly retrain doesn't break anything — production keeps using the last-trained model, which is still calibration-correct because the calibrators were fit against it.

### Date handling gotcha

`run_predictions.sh` and `run_verification.sh` compute `day_diff` (target − today, in days) and pass it to the spider. Scripts use MacOS `date -v` syntax with a Linux `date -d` fallback — when modifying these wrappers preserve both branches. The scraper treats `day_diff` as the source of truth, not the date string, so the date in the output filename and the day actually scraped can drift if the diff math is wrong.
