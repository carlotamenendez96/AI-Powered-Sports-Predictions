#!/bin/bash
# Euroleague/EuroCup daily verification.
#
#   yesterday's finished games (euroleague-api, both E + U)
#     → append-results into team_game_stats.csv (idempotent dedup)
#     → refresh euroleague_elo.json (feature step) → settle slips → report
#
# v1 = data-side verification only: it keeps the corpus current so the next
# retrain / serve-time feature computation sees the latest games. A
# predictions-vs-results EVALUATOR (Brier/acc on settled games) and bet
# settlement are Phase 3 / a later evaluator — flagged in EUROLEAGUE_NEXT_STEPS.
#
# Usage: ./bin/run_euroleague_verification.sh [YYYY-MM-DD]   (default: yesterday)

set -u
cd "$(dirname "$0")/.." || exit 1

VENV_PATH="venv/bin/activate"

# Portable date (macOS date -v / Linux date -d).
if [ -n "${1:-}" ] && [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    TARGET_DATE="$1"
elif date -v-1d >/dev/null 2>&1; then
    TARGET_DATE=$(date -v-1d +%Y-%m-%d)
else
    TARGET_DATE=$(date -d "yesterday" +%Y-%m-%d)
fi

echo "========================================"
echo "    Euroleague Verification             "
echo "========================================"
echo "Target Date: $TARGET_DATE"
echo "Started: $(date "+%Y-%m-%d %H:%M:%S")"

if [ ! -f "$VENV_PATH" ]; then
    echo "[-] venv not found at $VENV_PATH"; exit 1
fi
source "$VENV_PATH"

mkdir -p logs output_euroleague
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/ml_project:$(pwd)/ml_project/euroleague"
# euroleague-api draws a tqdm bar per round; in a log file the \r-redrawn bars
# collapse into one multi-KB line that buries the actual output.
export TQDM_DISABLE=1

# Append finished games to the corpus (idempotent — dedups on gameId,teamId).
echo ""
echo "[*] Appending finished results (euroleague-api, E + U) ..."
if python3 ml_project/euroleague/fetch_euroleague_daily.py append-results --date "$TARGET_DATE"; then
    echo "[+] Results appended."
else
    echo "[-] Result append failed."
    exit 1
fi

# Refresh the ELO cache from the updated corpus. predict_euroleague reads ELO
# from euroleague_elo.json, which only this feature step writes — before this
# was chained here it changed only on retrain, so every game since the last
# retrain was missing from the ratings (measured 2026-10-03: 42 of 93 ladders
# behind by up to 17 points, 11 teams absent). Rolling form needs nothing: the
# predictor derives it from the corpus directly. ~0.5s. Also rewrites
# training_data.csv, which the models only read at retrain. Non-fatal: a stale
# cache degrades predictions, it does not invalidate the appended results.
echo ""
echo "[*] Refreshing ELO ratings ..."
if python3 ml_project/euroleague/euroleague_feature_engineering.py >/dev/null; then
    echo "[+] ELO cache refreshed ($(python3 -c 'import json;print(len(json.load(open("data_sets/Euroleague/euroleague_elo.json"))))') ladders)."
else
    echo "[!] ELO refresh failed (non-fatal) — predictions will use the previous ratings."
fi

# Settle bet slips against the freshly-appended results.
#
# Added 2026-09-28. Until now this script appended results and stopped, so
# Euroleague bets debited a bankroll and then stayed OPEN forever — placing
# them was spending, not betting. NOT the football resolver: that one
# hardcodes the O/U line at 2.5 (every basketball total is "over 2.5", so
# every totals bet would settle WON) and assumes a draw exists.
#
# Non-fatal: the corpus append above is the irreplaceable step (the API only
# serves the current season), whereas settlement is idempotent and simply
# re-runs tomorrow for anything it could not decide today.
echo ""
echo "[*] Settling bet slips ..."
if python3 ml_project/resolve_basketball_bets.py --sport euroleague; then
    echo "[+] Settlement complete."
else
    echo "[!] Settlement failed (non-fatal) — results are appended; re-run to settle."
fi

# Prediction-vs-result report (verification_euroleague_<date>.csv) — what the
# dashboard's Verification Reports column lists. Non-fatal: the corpus append
# and settlement above are what matter; the report can be regenerated.
echo ""
echo "[*] Writing verification report ..."
python3 ml_project/euroleague/evaluate_euroleague_predictions.py --date "$TARGET_DATE" \
    || echo "[!] Verification report failed (non-fatal)."

echo ""
echo "========================================"
echo "    Verification Finished               "
echo "========================================"
