#!/bin/bash

# Change directory to project root
cd "$(dirname "$0")/.." || exit

# Configuration
VENV_PATH="venv/bin/activate"

# Date for verification (default: yesterday)
# Usage: ./run_verification.sh [YYYY-MM-DD]
if [ -z "$1" ]; then
    if date -v-1d >/dev/null 2>&1; then
        # macOS
        TARGET_DATE=$(date -v-1d +%Y-%m-%d)
    else
        # Linux (GitHub Actions / Ubuntu)
        TARGET_DATE=$(date -d "yesterday" +%Y-%m-%d)
    fi
else
    TARGET_DATE=$1
fi

# Results go to their OWN file. They used to be written back over
# output/matches_<date>.json with `-O`, which destroyed the pre-match slate:
# verification-mode parsing reads final scores and does not re-read the
# 1X2 odds, so every verified day lost its odds and the market baseline
# ("how would backing the favourite have done?") became unanswerable after
# the fact. Keeping them apart preserves the odds for every settled day.
RESULTS_JSON="output/results_$TARGET_DATE.json"
MATCHES_JSON="output/matches_$TARGET_DATE.json"
PREDICTIONS_CSV="output/predictions_$TARGET_DATE.csv"
BETS_FILE="output/bets_$TARGET_DATE.json"
VERIFICATION_CSV="output/verification_$TARGET_DATE.csv"

echo "========================================"
echo "    Flashscore Prediction Verification  "
echo "========================================"
echo "Verifying Date: $TARGET_DATE"

# 1. Activate Virtual Environment
if [ -f "$VENV_PATH" ]; then
    source $VENV_PATH
else
    echo "[-] Error: Virtual Environment not found at $VENV_PATH"
    exit 1
fi

# 2. Determine source of match IDs to scrape.
#    Preferred: predictions CSV (full data including odds, league, probs).
#    Fallback: open bets in bets_<date>.json — lets verification settle
#    bets even when the predictions file was archived/deleted.
HAVE_PREDICTIONS=0
LIVE_IDS=""
if [ -f "$PREDICTIONS_CSV" ]; then
    LIVE_IDS=$(python3 -c "import pandas as pd; df=pd.read_csv('$PREDICTIONS_CSV'); print(','.join(df['match_id'].dropna().astype(str).tolist()))" 2>/dev/null)
    HAVE_PREDICTIONS=1
    echo "[*] Predictions file found — using its match IDs."
elif [ -f "$BETS_FILE" ]; then
    LIVE_IDS=$(python3 -c "
import json
with open('$BETS_FILE') as f:
    slip = json.load(f)
ids = set()
for b in slip.get('bets', []):
    # Settle any non-terminal bet; OPEN is the usual case but a bet
    # could be missing 'status' on very old slips.
    if b.get('status') in ('OPEN', '', None) and b.get('match_id'):
        ids.add(str(b.get('match_id')))
print(','.join(sorted(ids)))
" 2>/dev/null)
    echo "[!] No predictions file. Falling back to match IDs from open bets in $BETS_FILE."
else
    echo "[-] Neither predictions ($PREDICTIONS_CSV) nor bets ($BETS_FILE) "
    echo "    exists for $TARGET_DATE. Nothing to verify."
    exit 1
fi

if [ -z "$LIVE_IDS" ]; then
    echo "[-] No match IDs to scrape. Nothing to verify."
    exit 1
fi

# 3. Calculate Day Offset for Scraper (still useful as a sanity log).
CURRENT_DATE_SEC=$(date +%s)
if date -j -f "%Y-%m-%d" "$TARGET_DATE" +%s >/dev/null 2>&1; then
    # macOS
    TARGET_DATE_SEC=$(date -j -f "%Y-%m-%d" "$TARGET_DATE" +%s)
else
    # Linux
    TARGET_DATE_SEC=$(date -d "$TARGET_DATE" +%s)
fi
DIFF_SEC=$((TARGET_DATE_SEC - CURRENT_DATE_SEC))
DAY_DIFF=$(( (DIFF_SEC - 43200) / 86400 ))
echo "[*] Target is $DAY_DIFF days from today. Running Scraper with ID-based mode..."

# 4. Run Scraper to get Results — always by ID here, since we have IDs
#    from either predictions or open bets.
scrapy crawl flashscore -a live_ids="$LIVE_IDS" -a mode=verification -O $RESULTS_JSON -L WARNING
if [ $? -ne 0 ]; then
    echo "[-] Scraper Failed!"
    exit 1
fi
echo "[+] Results saved to $RESULTS_JSON"

# Confirm the pre-match slate (and with it the 1X2 odds) survived this run.
# Not fatal — verification works without it — but a missing/odds-less slate
# means this day can never be scored against the market baseline.
if [ -s "$MATCHES_JSON" ]; then
    ODDS_N=$(python3 -c "import json;d=json.load(open('$MATCHES_JSON'));print(sum(1 for m in d if m.get('interaction_1x2_1')))" 2>/dev/null || echo "?")
    echo "[+] Pre-match slate preserved at $MATCHES_JSON ($ODDS_N matches with odds)."
else
    echo "[!] No pre-match slate at $MATCHES_JSON — market-baseline scoring unavailable for $TARGET_DATE."
fi

# 5. Run Evaluation — only if we have predictions to compare against.
# The evaluator streams its accuracy summary to stdout (still visible
# during the run); the verification CSV at $VERIFICATION_CSV is the
# structured source of truth that the rest of the pipeline reads.
# We no longer persist the stdout summary to a report_<date>.txt file —
# those accumulated without anything downstream reading them.
if [ "$HAVE_PREDICTIONS" -eq 1 ]; then
    echo ""
    echo "[*] Comparing Predictions vs Results..."
    python3 ml_project/evaluate_predictions.py --preds $PREDICTIONS_CSV --results $RESULTS_JSON --output $VERIFICATION_CSV
    echo "[+] Verification CSV saved to $VERIFICATION_CSV"
else
    echo "[!] Skipping prediction evaluation (no predictions file)."
fi

# 6. Resolve Bets — always runs. Uses results JSON directly; the
#    verification CSV is optional (skipped when predictions weren't
#    available so we couldn't generate it).
echo ""
echo "[*] Resolving Open Bets across all slips..."
if [ "$HAVE_PREDICTIONS" -eq 1 ]; then
    python3 ml_project/resolve_daily_bets.py --bets_dir output --results $RESULTS_JSON --verification_csv $VERIFICATION_CSV
else
    python3 ml_project/resolve_daily_bets.py --bets_dir output --results $RESULTS_JSON
fi

# 7. Referee history append (D4 Paso 4, ver docs/enriched_match_data_roadmap.md).
#    Grows data_sets/referees/referee_matches.csv with today's officiated
#    matches when the referee assignment is known (output/referees_$TARGET_DATE.json,
#    D4 Paso 3). Non-fatal and self-skipping — if Paso 3 didn't run that day,
#    or the results file is missing, it logs why and leaves the corpus alone.
echo ""
echo "[*] Referee history append..."
python3 scripts/referees/build_referee_history.py --append-verification "$TARGET_DATE" || true

# 8. Collection health + backtest trigger. Both are reports and must never
#    break this pipeline, hence `|| true`. Settlement above is what turns a
#    stored trajectory into a scoreable one, so this is the right moment to
#    ask whether enough has accrued for a backtest re-run -- the trigger is
#    new bound trajectories, not elapsed days (see check_backtest_due.py).
echo ""
echo "[*] Live collection health..."
python3 scripts/check_live_collection.py --days 14 || true

echo ""
echo "[*] Cashout backtest trigger..."
python3 scripts/check_backtest_due.py --run || true

echo ""
echo "========================================"
echo "           Verification Finished        "
echo "========================================"
