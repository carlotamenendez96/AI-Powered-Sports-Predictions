#!/bin/bash

# Change directory to project root (one level up from bin/)
cd "$(dirname "$0")/.." || exit

# One prediction run at a time — see bin/_lock.sh. Exits EXIT_LOCKED (4).
source bin/_lock.sh
acquire_lock predict || exit $EXIT_LOCKED

# Configuration
VENV_PATH="venv/bin/activate"
# Check for --force flag and Date Arg
FORCE_SCRAPE=false
TARGET_DATE=""
# Serve-time feature inputs (results CSVs + standings/form) are refreshed by
# default, but only when they are older than this. See step 3.
DO_REFRESH=true
FORCE_REFRESH=false
REFRESH_MAX_AGE_H=${PREDICT_REFRESH_MAX_AGE_H:-12}

for arg in "$@"; do
    if [ "$arg" == "--force" ] || [ "$arg" == "-f" ]; then
        FORCE_SCRAPE=true
    elif [ "$arg" == "--no-refresh" ]; then
        DO_REFRESH=false
    elif [ "$arg" == "--refresh" ]; then
        FORCE_REFRESH=true
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

    # The spider drops a sidecar describing what the day page held (rows before
    # filtering, rows inside target_leagues, rows kept). Remove any stale one
    # first so a previous run's file can never be read as this run's result.
    SCRAPE_STATS="logs/last_scrape_stats.json"
    rm -f "$SCRAPE_STATS"

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
    # scraped matches. But 0 matches has two very different causes, and the
    # sidecar is what tells them apart:
    #
    #   rows_on_page == 0        the day page never rendered -> BROKEN, exit 1
    #   in_target_leagues == 0   the day holds fixtures, none whitelisted
    #   kept == 0 (but > 0 above) whitelisted fixtures exist but all already
    #                            kicked off / finished (or are women's games)
    #
    # The last two are normal days with nothing to predict, not failures, and
    # they exit EXIT_NO_FIXTURES so the UI can say so instead of "Prediction
    # failed". A missing/mismatched sidecar falls back to treating 0 as broken.
    EXIT_NO_FIXTURES=3
    SCRAPED_COUNT=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_JSON'))))" 2>/dev/null || echo "-1")

    # "rows in_target kept" for this day_diff, or "" when unusable.
    SCRAPE_FACTS=$(python3 - "$SCRAPE_STATS" "$DAY_DIFF" <<'PY' 2>/dev/null || echo ""
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if int(d.get("day_diff", -999)) != int(sys.argv[2]) or not d.get("filtered"):
    sys.exit(1)          # sidecar belongs to some other crawl - do not trust it
print(d.get("rows_on_page", 0), d.get("in_target_leagues", 0), d.get("kept", 0))
PY
)

    if [ "$SCRAPY_RC" -eq 0 ] && [ "$SCRAPED_COUNT" -gt 0 ]; then
        echo "[+] Scraper Finished. $SCRAPED_COUNT matches saved to $OUTPUT_JSON"
        echo "[$end_date] Status: Success | Matches: $SCRAPED_COUNT | Start: $start_date | End: $end_date | Duration: ${duration}s" >> logs/scraper_status.log
    elif [ "$SCRAPY_RC" -eq 0 ] && [ "$SCRAPED_COUNT" -eq 0 ] && [ -n "$SCRAPE_FACTS" ] \
         && [ "$(echo "$SCRAPE_FACTS" | cut -d' ' -f1)" -gt 0 ]; then
        ROWS=$(echo "$SCRAPE_FACTS" | cut -d' ' -f1)
        IN_TARGET=$(echo "$SCRAPE_FACTS" | cut -d' ' -f2)
        echo "[=] No fixtures to predict for $DATE."
        if [ "$IN_TARGET" -eq 0 ]; then
            echo "    The day page loaded fine ($ROWS matches listed), but none are in"
            echo "    your target leagues (data_sets/target_leagues.json) — typically a"
            echo "    domestic-cup or international-break day."
        else
            echo "    The day page loaded fine ($ROWS matches listed, $IN_TARGET in your target"
            echo "    leagues), but every one has already kicked off or finished."
        fi
        echo "    The scraper is healthy; there is simply nothing to predict."
        echo "[$end_date] Status: No fixtures | Matches: 0 | On page: $ROWS | In target: $IN_TARGET | Start: $start_date | End: $end_date | Duration: ${duration}s" >> logs/scraper_status.log
        exit $EXIT_NO_FIXTURES
    else
        if [ "$SCRAPY_RC" -ne 0 ]; then
            echo "[-] Scraper Failed (scrapy exit code $SCRAPY_RC)."
        elif [ "$SCRAPED_COUNT" -lt 0 ]; then
            echo "[-] Scraper Failed: $OUTPUT_JSON is missing or not valid JSON."
        else
            echo "[-] Scraper Failed: 0 matches scraped and the day page was empty."
            if [ -z "$SCRAPE_FACTS" ]; then
                echo "    (no usable $SCRAPE_STATS — the crawl did not reach the day list)"
            fi
            echo "    Check logs/pipeline_output.log — a missing Playwright browser"
            echo "    (after a playwright upgrade) is the usual cause; fix with:"
            echo "        source venv/bin/activate && playwright install chromium"
        fi
        echo "[$end_date] Status: Failed | Matches: $SCRAPED_COUNT | Start: $start_date | End: $end_date | Duration: ${duration}s" >> logs/scraper_status.log
        exit 1
    fi
fi

# 3. Refresh serve-time feature inputs.
#
# The scraped fixture file supplies only the teams, kickoff and odds. Every
# other feature is derived at serve time from two corpora on disk:
#
#   data_sets/MatchHistory/  -> rolling form, ELO, H2H (predict_matches.get_team_stats)
#   data_sets/standings/     -> season-to-date PPG + attack/defence strength,
#                               rank gaps and draw calibration, via
#                               HeuristicAdjuster.get_team_strength
#
# Until now nothing in this pipeline refreshed either; they were updated by the
# weekly retrain or by the two UI buttons, so a daily predict could quietly
# serve week-old inputs (2026-09-26: MatchHistory same-day, standings 8 days
# stale) with nothing in predictions_<date>.csv saying so.
#
# Placement is deliberate — AFTER the scrape, so an international-break evening
# exits at "No fixtures" above instead of paying the ~13 min standings crawl for
# a slate with nothing to predict, and BEFORE the predictor, which reads both
# corpora at construction time.
#
# Coverage caveat: data_sets/standings_form_flashscore_direct_links.csv lists
# 26 leagues across 16 European countries, while MatchHistory spans 44. On a
# slate outside that set (measured on the 2026-09-27 card: Nations League,
# Liga MX, MLS) get_team_strength returns its neutral (0.0, 1.0, 1.0) fallback
# for every match and 3b buys nothing — 3a still helps, since MatchHistory does
# carry MEX/USA. Widening coverage means adding rows to that link CSV.
#
# Both steps are NON-FATAL: stale inputs degrade features, but a hard exit
# would throw away the whole slate over a Flashscore DOM change. Each prints
# the age it is working with, so the log always says which case happened.
stat_mtime() {
    # macOS/BSD first, GNU second — same portability split as the date logic.
    stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null
}

newest_mtime() {
    # Epoch seconds of the most recently written file matching <dir> <pattern>,
    # or 0 when the directory holds none.
    local dir="$1" pattern="$2" newest=0 m
    while IFS= read -r f; do
        m=$(stat_mtime "$f")
        if [ -n "$m" ] && [ "$m" -gt "$newest" ]; then newest="$m"; fi
    done < <(find "$dir" -type f -name "$pattern" 2>/dev/null)
    echo "$newest"
}

age_hours() {
    # Whole hours since <epoch>, or -1 when there is nothing on disk.
    if [ "$1" -le 0 ]; then echo "-1"; else echo $(( ($(date +%s) - $1) / 3600 )); fi
}

age_label() {
    if [ "$1" -lt 0 ]; then echo "none on disk"; else echo "${1}h old"; fi
}

echo ""
if [ "$DO_REFRESH" == "false" ]; then
    echo "[=] Step 3: skipping serve-time input refresh (--no-refresh)."
elif [ "$DAY_DIFF" -lt 0 ] && [ "$FORCE_REFRESH" != "true" ]; then
    echo "[=] Step 3: skipping serve-time input refresh — $DATE is in the past."
    echo "    standings/form describe only the CURRENT table, so refreshing them"
    echo "    cannot make a historical slate more accurate. Pass --refresh to override."
else
    echo "[*] Step 3: Refreshing serve-time feature inputs (max age ${REFRESH_MAX_AGE_H}h)..."

    # 3a. Results CSVs -> rolling form / ELO / H2H.
    RESULTS_AGE=$(age_hours "$(newest_mtime "data_sets/MatchHistory" '*.csv')")
    if [ "$FORCE_REFRESH" == "true" ] || [ "$RESULTS_AGE" -lt 0 ] || [ "$RESULTS_AGE" -ge "$REFRESH_MAX_AGE_H" ]; then
        echo "[*]   Results CSVs ($(age_label "$RESULTS_AGE")) — updating..."
        if python3 scripts/update_football_data.py; then
            echo "[+]   Results CSVs updated."
        else
            echo "[!]   Results update FAILED (non-fatal). Rolling form, ELO and H2H will"
            echo "      be computed from data that is $(age_label "$RESULTS_AGE")."
        fi
    else
        echo "[=]   Results CSVs are $(age_label "$RESULTS_AGE") (< ${REFRESH_MAX_AGE_H}h) — skipping."
    fi

    # 3b. Standings + form -> season-to-date strength in HeuristicAdjuster.
    # The wrapper already fails loudly on both a spider error and a crawl that
    # silently scrapes nothing, so its exit code is trustworthy here.
    echo ""
    STANDINGS_AGE=$(age_hours "$(newest_mtime "data_sets/standings" '*.json')")
    if [ "$FORCE_REFRESH" == "true" ] || [ "$STANDINGS_AGE" -lt 0 ] || [ "$STANDINGS_AGE" -ge "$REFRESH_MAX_AGE_H" ]; then
        echo "[*]   Standings/form ($(age_label "$STANDINGS_AGE")) — updating (crawls 26 leagues, ~13 min)..."
        if /bin/bash bin/update_leagues_data.sh; then
            echo "[+]   Standings/form updated."
        else
            echo "[!]   Standings update FAILED (non-fatal). Season-to-date PPG and"
            echo "      attack/defence strength will come from tables that are"
            echo "      $(age_label "$STANDINGS_AGE") — predictions continue, but degraded."
        fi
    else
        echo "[=]   Standings/form are $(age_label "$STANDINGS_AGE") (< ${REFRESH_MAX_AGE_H}h) — skipping."
    fi
fi

# 4. Run Prediction
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

# 5. National-Team Predictions (router): the club predictor skips international
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

# 7. Cards market (Paso 6 / Fase E) — isolated head; ON by default after gate.
# Skip with CARDS_ENABLED=0. Non-fatal; never touches 1X2 / O/U picks or bankrolls.
# 7a. Winamax cards O/U 3.5 odds (Flashscore has no market; Pamestoixima
#     Akamai-blocked headless). Plain HTTPS PRELOADED_STATE — no Playwright.
#     Patches matches_$DATE.json in place when a fixture links + market exists.
if [ "${CARDS_ENABLED:-1}" != "0" ]; then
    echo ""
    echo "[*] Enriching cards O/U 3.5 odds from Winamax for $DATE..."
    python3 -m ml_project.cards.winamax_odds "$DATE" \
        && echo "[+] Winamax cards-odds enrich complete." \
        || echo "[!] Winamax cards-odds enrich failed (non-fatal); predict continues."

    echo ""
    echo "[*] Running cards predictions for $DATE..."
    python3 -m ml_project.cards.predict_cards "$DATE" \
        && python3 -m ml_project.cards.merge_into_predictions "$DATE" \
        && echo "[+] Cards prediction + merge into predictions_$DATE.csv complete." \
        || echo "[!] Cards prediction failed (non-fatal); 1X2/O/U intact."
fi

echo ""
echo "========================================"
echo "           Pipeline Finished            "
echo "Pipeline Ended: $(date "+%Y-%m-%d %H:%M:%S")"
echo "========================================"
