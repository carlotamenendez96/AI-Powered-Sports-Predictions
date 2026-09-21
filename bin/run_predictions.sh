#!/bin/bash

# Change directory to project root (one level up from bin/)
cd "$(dirname "$0")/.." || exit

# Configuration
VENV_PATH="venv/bin/activate"
# Check for --force flag and Date Arg
FORCE_SCRAPE=false
TARGET_DATE=""

for arg in "$@"; do
    if [ "$arg" == "--force" ] || [ "$arg" == "-f" ]; then
        FORCE_SCRAPE=true
    elif [[ "$arg" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        TARGET_DATE="$arg"
    fi
done

# Date Logic
if [ -z "$TARGET_DATE" ]; then
    # Default: Tomorrow
    if date -v+1d >/dev/null 2>&1; then
        # MacOS
        DATE=$(date -v+1d +%Y-%m-%d)
    else
        # Linux
        DATE=$(date -d "tomorrow" +%Y-%m-%d)
    fi
else
    DATE="$TARGET_DATE"
    echo "[*] Using Custom Date: $DATE"
fi

OUTPUT_JSON="output/matches_$DATE.json"

echo "========================================"
echo "    Flashscore ML Prediction Pipeline   "
echo "========================================"
echo "Date: $DATE"
echo "Pipeline Started: $(date "+%Y-%m-%d %H:%M:%S")"

# Calculate Day Difference (for Spiderman)
# CURRENT - TARGET ?? No, we want TARGET - CURRENT.
# If Target is tomorrow, Diff = +1.
CURRENT_SEC=$(date +%s)
# Need portable date to sec? MacOS `date -j -f ...` vs Linux `date -d ...`
if date -j -f "%Y-%m-%d" "$DATE" +%s >/dev/null 2>&1; then
    # MacOS
    TARGET_SEC=$(date -j -f "%Y-%m-%d" "$DATE" +%s)
else
    # Linux
    TARGET_SEC=$(date -d "$DATE" +%s)
fi

DIFF_SEC=$((TARGET_SEC - CURRENT_SEC))
# Rounding
DAY_DIFF=$(( (DIFF_SEC + 43200) / 86400 ))

echo "[*] Target Offset: $DAY_DIFF days from today."


# 1. Activate Virtual Environment
if [ -f "$VENV_PATH" ]; then
    source $VENV_PATH
    echo "[+] Virtual Environment Activated"
else
    echo "[-] Error: Virtual Environment not found at $VENV_PATH"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Redirect all output to log file (and stdout) - Overwrite mode
exec > >(tee logs/pipeline_output.log) 2>&1

# 2. Run Scraper or Skip
NEED_SCRAPE=false

# Check if we should scrape
if [ "$FORCE_SCRAPE" == "true" ]; then
    NEED_SCRAPE=true
elif [ ! -s "$OUTPUT_JSON" ]; then
    echo "[*] Output file missing or empty."
    NEED_SCRAPE=true
else
    # File exists and size > 0. Check for JSON Corruption.
    if ! python3 -c "import json; json.load(open('$OUTPUT_JSON'))" > /dev/null 2>&1; then
        echo "[!] Output file exists but contains corrupt JSON. Forcing re-scrape."
        NEED_SCRAPE=true
    # An empty `[]` is 4 bytes on disk, so it passes both the -s and the
    # json.load checks above and would silently short-circuit the scraper into
    # reusing a failed run's output forever. Treat it as no cache at all.
    elif [ "$(python3 -c "import json; print(len(json.load(open('$OUTPUT_JSON'))))" 2>/dev/null)" == "0" ]; then
        echo "[!] Output file exists but holds 0 matches (failed prior scrape). Forcing re-scrape."
        NEED_SCRAPE=true
    else
        echo "[*] valid Output file found. Skipping Scraper."
    fi
fi

if [ "$NEED_SCRAPE" == "true" ]; then
    echo "[*] Starting Scraper..."
    start_ts=$(date +%s)
    start_date=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$start_date] Status: Started" >> logs/scraper_status.log

    # Pass day_diff
    scrapy crawl flashscore -O $OUTPUT_JSON -L WARNING -a filter_leagues=true -a day_diff=$DAY_DIFF
    # Capture immediately: any intervening command (even `date`) clobbers $?.
    SCRAPY_RC=$?

    end_ts=$(date +%s)
    end_date=$(date "+%Y-%m-%d %H:%M:%S")
    duration=$((end_ts - start_ts))

    # Scrapy exits 0 even when every request errored out (e.g. Playwright's
    # browser binary is missing after a version bump), leaving a well-formed
    # but EMPTY `[]` on disk. So the exit code alone is not enough — count the
    # scraped matches and treat an empty slate as a failure. A genuinely empty
    # day is rare and re-runnable; a silent 0 is what hides a broken scraper.
    SCRAPED_COUNT=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_JSON'))))" 2>/dev/null || echo "-1")

    if [ "$SCRAPY_RC" -eq 0 ] && [ "$SCRAPED_COUNT" -gt 0 ]; then
        echo "[+] Scraper Finished. $SCRAPED_COUNT matches saved to $OUTPUT_JSON"
        echo "[$end_date] Status: Success | Matches: $SCRAPED_COUNT | Start: $start_date | End: $end_date | Duration: ${duration}s" >> logs/scraper_status.log
    else
        if [ "$SCRAPY_RC" -ne 0 ]; then
            echo "[-] Scraper Failed (scrapy exit code $SCRAPY_RC)."
        elif [ "$SCRAPED_COUNT" -lt 0 ]; then
            echo "[-] Scraper Failed: $OUTPUT_JSON is missing or not valid JSON."
        else
            echo "[-] Scraper Failed: 0 matches scraped."
            echo "    Either no target-league fixtures exist for $DATE, or the scrape broke."
            echo "    Check logs/pipeline_output.log — a missing Playwright browser"
            echo "    (after a playwright upgrade) is the usual cause; fix with:"
            echo "        source venv/bin/activate && playwright install chromium"
        fi
        echo "[$end_date] Status: Failed | Matches: $SCRAPED_COUNT | Start: $start_date | End: $end_date | Duration: ${duration}s" >> logs/scraper_status.log
        exit 1
    fi
fi

# 3. Run Prediction
echo ""
echo "[*] Running ML Prediction Engine..."

# Export PYTHONPATH to include project root and ml_project so imports work
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ml_project

# Check if JSON is valid (rudimentary check) or just run script
if [ ! -s "$OUTPUT_JSON" ]; then
    echo "[-] Error: Output JSON is empty. Scraper likely failed."
    exit 1
fi

python3 -c "from ml_project.predict_matches import MatchPredictor; predictor = MatchPredictor(scraper_output='$OUTPUT_JSON'); predictor.predict()"

if [ $? -eq 0 ]; then
    echo "[+] Prediction Complete."
else
    echo "[-] Prediction Failed!"
    exit 1
fi

# 4. National-Team Predictions (router): the club predictor skips international
# competitions (World Cup / Euro / Nations League); this appends their rows to
# the same predictions_<date>.csv using the eloratings model. Non-fatal — if it
# fails, club predictions remain intact.
echo ""
echo "[*] Running National-Team Prediction (eloratings model)..."
python3 scripts/national_teams/predict_nt_batch.py --matches "$OUTPUT_JSON" \
    && echo "[+] National-Team Prediction Complete." \
    || echo "[!] NT prediction step failed (non-fatal); club predictions intact."

# 5. Availability / bajas extraction (D4 Paso 1, ver docs/enriched_match_data_roadmap.md).
# Read-only respecto al modelo: solo escribe output/availability_<date>.json.
# Non-fatal — si falla (o Flashscore cambia el DOM), las predicciones ya están escritas.
echo ""
echo "[*] Extracting availability (bajas) for $DATE..."
python3 scripts/d4_injuries/extract_availability.py "$DATE" \
    && echo "[+] Availability extraction complete." \
    || echo "[!] Availability extraction failed (non-fatal); predictions intact."

# 6. Árbitro del día (D4 Paso 3, Fase B, ver docs/enriched_match_data_roadmap.md).
# Read-only respecto al modelo: solo escribe output/referees_<date>.json.
# Non-fatal — si falla (o el partido no tiene árbitro asignado aún), las predicciones ya están escritas.
echo ""
echo "[*] Extracting referee assignments for $DATE..."
python3 scripts/d4_referees/extract_referees.py "$DATE" \
    && echo "[+] Referee extraction complete." \
    || echo "[!] Referee extraction failed (non-fatal); predictions intact."

echo ""
echo "========================================"
echo "           Pipeline Finished            "
echo "Pipeline Ended: $(date "+%Y-%m-%d %H:%M:%S")"
echo "========================================"
