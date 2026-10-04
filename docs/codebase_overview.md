# Flashscore Predictor: Codebase Overview

This document provides a summary of all Python (`.py`) and Shell (`.sh`) scripts in the project.

## 1. Root Directory Scripts

| File | Description |
| :--- | :--- |
| `manage_server.sh` | **Server Control**: Starts, stops, and restarts the Flask Web UI in the background (`nohup`). |
| `retrain_pipeline.sh` | **Automation**: Runs the full pipeline: Update Results &rarr; Update Standings &rarr; Retrain Model. |
| `run_predictions.sh` | **Prediction**: Daily driver. Scrapes tomorrow's matches, refreshes the serve-time feature inputs (results CSVs + standings/form, age-gated at 12h; `--no-refresh` / `--refresh`), and generates `predictions_YYYY-MM-DD.csv`. |
| `run_verification.sh` | **Verification**: Scrapes results for a past date (default: yesterday) and compares them with predictions. |
| `update_leagues_data.sh` | **Data Update**: Runs the `standings` spider to update league tables and form JSONs. |
| `run_live_analysis.py` | **Live Mode**: Standalone script to fetch live match stats and predict outcome in real-time. |
| `run_live_loop.py` | **Live Mode**: Daemon that runs `run_live_analysis.py` in a loop (every 10 mins). |
| `setup_data.sh` | **Setup**: Wrapper script to download historical data for a specific season. |


## 2. Machine Learning Project (`ml_project/`)

| File | Description |
| :--- | :--- |
| `betting_engine.py` | **(Legacy / orphan)** Reference implementation of bet placement + resolution + league filter. Not instantiated anywhere; the active betting logic lives in `web_ui/app.py`. Kept for design reference until the league-performance filter is migrated. |
| `data_loader.py` | **IO**: Utility class to load raw CSV match data into Pandas DataFrames. |
| `elo_engine.py` | **Feature**: Calculates historical ELO ratings for all teams. |
| `elo_scraper.py` | **Utility**: (Deprecated/Optional) Scraper for external ELO sources. |
| `entity_resolver.py` | **Utility**: Fuzzy matching logic to map team names between different data sources. |
| `evaluate_predictions.py` | **Verification**: Compares predicted vs actual results and generates accuracy reports. |
| `feature_engineering.py` | **Core**: Transforms raw match data into rolling features (Form, PPG, Attack/Defense Strength) for the model. |
| `generate_target_leagues.py`| **Config**: Helper to generate the list of active leagues (not actively used in runtime). |
| `heuristic_adjuster.py` | **Logic**: Post-prediction adjuster (calibration, draw cap, rank/form/trend boosts with cumulative-magnitude cap, value/EV logging). Also exposes `get_team_strength()` consumed by `predict_matches.py` to derive PPG/Att/Def from current standings at inference. |
| `live_adjuster.py` | **Live**: Heuristics specifically for in-play stats (Analysis of Shots/xG). |
| `predict_matches.py` | **Core**: Main prediction CLI. Loads model, fetches features for upcoming games, and predicts. |
| `resolve_daily_bets.py` | **Settlement**: Settles open `output/bets_*.json` slips against scraped results. Called by `bin/run_verification.sh`. |
| `team_mapping.py` | **Config**: Static dictionary for known team name variations. |
| `train_model.py` | **Training**: Trains XGBoost models (1X2, draw, O/U Poisson), saving them to JSON. |
| `tune_model.py` | **Optimization**: Performs stepwise hyperparameter tuning for XGBoost and saves best parameters. |
| `nba/` (subpackage) | All NBA pipeline modules: `predict_nba.py`, `train_nba_models.py`, `tune_nba_models.py`, `evaluate_nba_predictions.py`, `nba_feature_engineering.py`, `nba_utils.py`, `fetch_nba_results.py`, `fetch_nba_history_stats.py`, `fetch_nba_stats_tables.py`, `fetch_espn_odds.py`. Models at `models/nba/`, outputs at `output_basketball/`. NBA UI is currently detached. |

## 3. Web Interface (`web_ui/`)

| File | Description |
| :--- | :--- |
| `app.py` | **Flask App**: Main web server. `/` is the sport-picker landing + portfolio summary; `/football/*` (blueprint) hosts dashboard, betting (`/auto_wager`, `/place_bets`, `/betting`), live-loop controls, and soft-delete via `/football/delete_file/<filename>`. Module-level helper `compute_sport_summary(bets_dir)` aggregates per-lane stats from any sport's slips folder — used by both `/football/betting` and the landing page. |
| `sports_config.py` | **Bankroll/config helpers**. Single source of truth for `data_sets/betting_config.json` access. `get_sport_config(slug)`, `get_bankroll(slug)`, `update_bankroll(slug, delta)`, `all_bankrolls()`, `total_bankroll()`. Routes never read or mutate the JSON directly. |
| `nba/routes.py` | **(Detached)** Flask Blueprint for NBA routes. Code preserved but not registered in `app.py` — NBA UI is currently dormant. Template at `templates/nba/index.html`. |

## 4. Scrapers (`flashscore_scraper/`, `scripts/`)

| File | Description |
| :--- | :--- |
| `spiders/flashscore_spider.py` | **Scraper**: Main spider. Scrapes Daily Matches, 1X2 Odds, O/U 2.5 Odds, and Results using Playwright. |
| `spiders/standings_spider.py`| **Scraper**: Scrapes League Standings and Form Tables. |
| `scripts/update_football_data.py` | **Data Update**: Downloads and updates the historical CSV dataset from *Football-Data.co.uk*. |
| `scripts/setup_historical_data.py` | **Setup**: Downloads main and extra league CSVs for a specific season. |
