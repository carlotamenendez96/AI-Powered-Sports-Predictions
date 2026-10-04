#!/bin/bash
# Euroleague/EuroCup daily prediction pipeline (Phase 2 — odds/EV wired in Phase 3).
#
#   tomorrow's fixtures (euroleague-api, both E + U)
#     → predictor (corpus-derived features, per-competition calibrated P(home_win))
#     → output_euroleague/predictions_euroleague_<date>.csv
#
# Odds are NOT fetched yet: the Flashscore Euroleague odds probe is deferred to
# season start (off-season now). Phase 3 joins odds for EV/Kelly; the predictor
# works without them.
#
# Usage: ./bin/run_euroleague_predictions.sh [YYYY-MM-DD]   (default: tomorrow)

set -u
cd "$(dirname "$0")/.." || exit 1

VENV_PATH="venv/bin/activate"

# Portable date (macOS date -v / Linux date -d), same pattern as run_predictions.sh.
if [ -n "${1:-}" ] && [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    TARGET_DATE="$1"
elif date -v+1d >/dev/null 2>&1; then
    TARGET_DATE=$(date -v+1d +%Y-%m-%d)
else
    TARGET_DATE=$(date -d "tomorrow" +%Y-%m-%d)
fi

echo "========================================"
echo "    Euroleague Prediction Pipeline      "
echo "========================================"
echo "Target Date: $TARGET_DATE"
echo "Pipeline Started: $(date "+%Y-%m-%d %H:%M:%S")"

if [ ! -f "$VENV_PATH" ]; then
    echo "[-] venv not found at $VENV_PATH"; exit 1
fi
source "$VENV_PATH"

mkdir -p logs output_euroleague
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/ml_project:$(pwd)/ml_project/euroleague"

# 1. Fixtures (writes data_sets/Euroleague/fixtures_<date>.json for both competitions).
echo ""
echo "[*] Fetching fixtures (euroleague-api, E + U) ..."
if ! python3 ml_project/euroleague/fetch_euroleague_daily.py fixtures --date "$TARGET_DATE"; then
    echo "[-] Fixture fetch failed."
    exit 1
fi

# 1b. Odds (Flashscore: moneyline + the full totals ladder, both competitions).
# NON-FATAL by design: the predictor works without odds — it just leaves
# Over Line / P(Over) blank and auto_wager builds an empty slip — so a
# Flashscore DOM change should cost us the betting surface for a day, not the
# predictions. Writes output_euroleague/euroleague_odds_<date>.json.
echo ""
echo "[*] Fetching odds (Flashscore, E + U) ..."
if python3 ml_project/euroleague/fetch_euroleague_odds.py --date "$TARGET_DATE"; then
    echo "[+] Odds fetched."
else
    echo "[!] Odds fetch failed (non-fatal) — predictions will have no line, slips will be empty."
fi

# 2. Predict (corpus-derived features + per-competition Platt).
echo ""
echo "[*] Running predictor ..."
if python3 ml_project/euroleague/predict_euroleague.py --date "$TARGET_DATE"; then
    echo "[+] Prediction complete."
else
    echo "[-] Prediction failed."
    exit 1
fi

echo ""
echo "========================================"
echo "    Euroleague Pipeline Finished        "
echo "========================================"
