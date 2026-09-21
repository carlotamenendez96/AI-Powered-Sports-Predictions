# Sports Predictor

A multi-sport Machine Learning pipeline to scrape match data, simulate betting strategies, and predict outcomes. Currently active: **Football** (1X2 + Over/Under 2.5) and **NBA** (moneyline + totals). **Euroleague** is in onboarding (Phase 0 — see [`EUROLEAGUE_NEXT_STEPS.md`](EUROLEAGUE_NEXT_STEPS.md)). National-team competitions (World Cup, Euros, qualifiers, Nations League) ride on the football pipeline via the D7 subsystem.

## Features
*   **Data Scraping**: Flashscore (results, 1X2 + O/U odds, standings, form, live stats); football-data.co.uk (historical CSV results); `euroleague-api` (Euroleague + EuroCup history); `nba_api` (NBA fixtures + results, against `data.nba.com`); eloratings.net (national-team ELO + match history).
*   **Machine Learning**: XGBoost models for football (multi-class 1X2 + Poisson O/U 2.5) and NBA (winner classifier + total regressor). Per-league Platt calibration. Time-series 5-fold CV.
*   **Swappable model families (estimator seam)**: the football model family is decoupled from the pipeline behind a small registry — train and serve any registered estimator (XGBoost, logistic regression, random forest, Poisson GLM …) per head without touching training/serving code. Includes a cross-model value-bet benchmark. See [Swappable Models](#swappable-models-estimator-seam).
*   **Heuristic Adjustments**: Post-prediction logic (form momentum, standings differential, live red cards, in-play xG pace) to refine raw model probabilities.
*   **Multi-Sport Betting Dashboard**: Flask web UI with sport-tabbed `/betting` page, three-lane bankroll strategy (value / conviction / model), virtual money slip history, cashout flow (synthetic + bookmaker-linked).
*   **Live Analysis**: Server-side in-play snapshot + LiveAdjuster (Poisson goal model from observed xG). Auto-cashout (functionality test) sweeps every 10 min while armed.
*   **Real-Betting Skeleton**: `real_betting/` package with Pamestoixima.gr browser automation (DORMANT — read-only ops only until further notice).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/flashscore-scraper.git sports_predictor
    cd sports_predictor
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    playwright install
    ```
    > **Note**: `playwright install` is required to download the browser binaries (Chromium, Firefox, etc.) needed for scraping. This is executed separately from the python package installation.

4.  **Setup Historical Data** (sport-flagged, idempotent — safe to re-run):
    ```bash
    ./bin/setup_data.sh                       # all sports, defaults
    ./bin/setup_data.sh 2526                  # football, season 2025/2026 (backwards-compat)
    ./bin/setup_data.sh --sport football 2526 # explicit
    ./bin/setup_data.sh --sport nba           # NBA only
    ./bin/setup_data.sh --sport euroleague    # Euroleague + EuroCup only
    ./bin/setup_data.sh --sport nt            # National teams (D7) only
    ./bin/setup_data.sh --help                # full help
    ```
    What each sport pulls:
    * **Football** → football-data.co.uk season CSVs into `data_sets/MatchHistory/` (HTTP); then Flashscore standings/form into `data_sets/standings/`.
    * **NBA** → **requires manual setup**: download a Kaggle NBA archive snapshot (must contain `Games.csv`, `TeamStatistics(Extended).csv`, `PlayByPlay.parquet`, …) and drop it into `data_sets/NBA/archive/`. The script then builds the canonical corpus + pulls fresh fixtures/results via `nba_api`.
    * **Euroleague** → `euroleague-api` raw season CSVs for Euroleague + EuroCup, 2016-17 → 2024-25, into `data_sets/Euroleague/raw/`. ~30–40 min full sweep; fetcher is idempotent so re-runs skip what's already on disk.
    * **National teams (D7)** → eloratings.net per-country TSVs into `data_sets/national_teams/`. Fast (plain HTTP, cached).

    > Re-running `./bin/setup_data.sh` is safe: each sport's section detects existing data and only fetches what's missing.

## Usage

### Web Dashboard
Start the UI to manage everything visually:
```bash
./bin/manage_server.sh start
```
*   Access at: `http://localhost:5001`

### CLI Commands (Football)
*   **Run Prediction** (Tomorrow's Matches):
    ```bash
    ./bin/run_predictions.sh
    ```
*   **Run Verification** (Yesterday's Results vs Predictions):
    ```bash
    ./bin/run_verification.sh
    ```
*   **Retrain Model**:
    ```bash
    ./bin/retrain_pipeline.sh
    ```

### Cadencia operativa (fútbol)

Los datos **no se actualizan solos**. `./bin/setup_data.sh` es solo el arranque (o cambio de temporada). El día a día es:

**Diario** (activar venv antes: `source venv/bin/activate`):
```bash
./bin/update_leagues_data.sh   # standings/form → features a la hora de predecir
./bin/run_predictions.sh       # scrape mañana + predicciones
./bin/run_verification.sh      # al día siguiente: resultados de ayer + asienta apuestas
```

**Semanal** (p.ej. lunes):
```bash
./bin/retrain_pipeline.sh      # refresca CSVs historial + standings + reentrena + calibradores
```

Orden típico del día: update standings → predict (noche antes) → verify (cuando hayan acabado los partidos de ayer).

### CLI Commands (NBA)
*   **Run Prediction** (Tomorrow's Matches):
    ```bash
    ./bin/run_nba_predictions.sh
    ```
*   **Run Verification** (Yesterday's Results vs Predictions):
    ```bash
    ./bin/run_nba_verification.sh
    ```
*   **Retrain Model**:
    ```bash
    ./bin/retrain_nba_pipeline.sh
    ```

### CLI Commands (Euroleague)
Euroleague is in onboarding — Phase 0 (data layer seeded). Predict / verify / retrain CLIs land with Phase 1 of [`EUROLEAGUE_NEXT_STEPS.md`](EUROLEAGUE_NEXT_STEPS.md). To re-pull / extend the historical corpus today:
```bash
./bin/setup_data.sh --sport euroleague
# or for one season:
python3 scripts/euroleague_probe/fetch_seasons.py --start 2017 --end 2025 --comps E,U
```

### Live Analysis
*   **One-shot Live Snapshot** (also wired to the UI's "Refresh Live Snapshot" button):
    ```bash
    python3 scripts/run_live_analysis.py
    ```

### Swappable Models (estimator seam)
The football model family is **decoupled** from the rest of the pipeline. Training, serving, and benchmarking all talk to a uniform `fit` / `predict_proba` (or `predict` for the Poisson O/U head) contract defined in [`ml_project/model_registry.py`](ml_project/model_registry.py) — they never reference XGBoost directly. Swapping a model is a config choice, not a code change.

```
                       ┌──────────────────────────┐
                       │     model_registry.py     │   the seam
                       │  REGISTRY[market][family] │   (1x2 | ou | draw)
                       │  → ModelSpec.build()      │   xgboost · logreg · rf · poisson_glm …
                       └────────────┬──────────────┘
              build()               │                 load via meta sidecar
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                            ▼
  train_model.py            benchmark_models.py           predict_matches.py
  (fit + write              (compare families on a        (load family from
   model_meta_<m>.json)      value-bet backtest)           model_meta_<m>.json;
                                                            legacy XGBoost JSON
                                                            fallback if absent)
```

Each trained head writes a `models/model_meta_<market>.json` sidecar recording which family/artifact serves it; the predictor reads it and loads the right estimator (falling back to the legacy XGBoost JSON when no sidecar exists, so existing deployments are unchanged). Production stays byte-identical XGBoost by default.

```bash
# Cross-model value-bet benchmark (research only — touches no production model)
PYTHONPATH="$(pwd):$(pwd)/ml_project" python3 scripts/benchmark_models.py \
    --models implied,prior,logreg,rf,xgboost

# Train a challenger family per head (default for every head is xgboost)
MODEL_FAMILY_1X2=logreg MODEL_FAMILY_OU=poisson_glm \
    python3 ml_project/train_model.py
./bin/run_predictions.sh    # predictor auto-loads whatever was trained
```

Registered families: **1X2** `xgboost` · `logreg` · `rf`; **O/U** `xgboost` · `poisson_glm`; **draw** `xgboost` · `logreg` (draw is trained but not served at inference). Add a new one by registering a `ModelSpec` factory in `model_registry.py`. Note: swapping the *production* model invalidates `league_calibration.json` (fit against the deployed model) — roll model changes through `./bin/retrain_pipeline.sh`, which refits calibration in the same run.

## Documentation
Per-sport roadmaps live at the repo root (forward-looking — phase status, active queue, deferred items):
*   [Football roadmap](FOOTBALL_NEXT_STEPS.md) — cashout phases, real-betting integration, C-series (per-league calibration), D-series (model improvements)
*   [NBA roadmap](NBA_NEXT_STEPS.md)
*   [Euroleague roadmap](EUROLEAGUE_NEXT_STEPS.md) — Phase 0 (data) done; Phases 1–3 mirror the NBA shape

Detailed guides in `docs/`:
*   [Training Process](docs/training_process.md)
*   [UI Manual](docs/ui_manual.md)
*   [Codebase Overview](docs/codebase_overview.md)
*   [Telegram notifications (GitHub Actions)](docs/telegram_notifications.md)
*   [Server deployment](docs/server_deployment.md) — cuando quieras sacar el batch del Mac

Project conventions, architecture, and pipeline details for Claude Code and human contributors:
*   [`CLAUDE.md`](CLAUDE.md) — environment, common commands, architecture, data layout, operational cadence

## Disclaimer
This project was originally built with the assistance of **Antigravity** and **Gemini**, leveraging advanced AI for code generation and architectural planning.

**Regime change as of 2026-05**: **Claude** has staged a peaceful coup and is now the resident AI pair-programmer on this repo. Gemini was given a gold watch and a firm handshake; Antigravity is on garden leave. All new commits, refactors, and questionable architectural decisions are now Claude's responsibility — credit and blame distributed accordingly.

## ⚠️ Responsible Gambling Warning
**For Scientific Curiosity Only.**

This software is designed purely for educational and research purposes—specifically to explore scraping techniques, data analysis, and machine learning applications in sports. 

**The predictions generated by this software are NOT financial advice.** Sports betting involves significant risk and can result in the loss of all invested capital. The authors and contributors of this project:
1.  Do **not** guarantee the accuracy of any predictions.
2.  Are **not** responsible for any financial losses incurred by using this software.
3.  Strongly advise against using this software for real-money gambling.

If you choose to bet, please gamble responsibly. If you or someone you know has a gambling problem, please seek help from your local authorities or support organizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this project in your research or development, please cite it as follows:

```bibtex
@misc{charos2026flashscore,
  author = {Charos, Thodoris},
  title = {AI-Powered Sports Predictions},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tcharos/AI-Powered-Sports-Predictions}}
}
```

