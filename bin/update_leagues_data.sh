#!/bin/bash

# Change directory to project root
cd "$(dirname "$0")/.." || exit
# Activate virtual environment
source venv/bin/activate

STANDINGS_DIR="data_sets/standings"
LINKS_CSV="data_sets/standings_form_flashscore_direct_links.csv"
LINKS_TEMPLATE="data_sets/standings_form_flashscore_direct_links.template.csv"

# Seed the Flashscore URL list from the committed template when missing
# (fresh clone — the live CSV is gitignored under data_sets/*).
if [ ! -s "$LINKS_CSV" ]; then
    if [ -s "$LINKS_TEMPLATE" ]; then
        cp "$LINKS_TEMPLATE" "$LINKS_CSV"
        echo "[*] Seeded $LINKS_CSV from template."
    else
        echo "[-] Missing $LINKS_CSV (and no template at $LINKS_TEMPLATE)." >&2
        echo "    The standings spider needs this URL list to know which leagues to scrape." >&2
        exit 1
    fi
fi

# One crawl at a time (see bin/_lock.sh): concurrent crawls slow each other
# down and race on the same output files. run_predictions.sh step 3 treats a
# non-zero exit here as "standings not refreshed" and carries on.
source bin/_lock.sh
acquire_lock standings || exit $EXIT_LOCKED

# Marker for the freshness check below. Created BEFORE the crawl so any file
# the pipeline writes is strictly newer than it.
MARKER="$(mktemp -t standings_run_marker)"
trap 'rm -f "$MARKER"; release_lock' EXIT

# Run the standings spider.
# No -O because the pipeline handles the files.
#
# Time-bounded since 2026-09-27. A 40-league crawl was observed running for
# 25h without writing anything: StandingsPipeline only writes in close_spider,
# so a stalled crawl produces no output at all, and run_predictions.sh step 3
# calls this script, which meant a hang here would block the nightly
# predictions indefinitely. Two bounds, because they fail differently:
#
#   CLOSESPIDER_TIMEOUT  graceful — Scrapy stops scheduling and closes the
#                        spider, so close_spider still writes whatever was
#                        collected. Partial standings beat none.
#   watchdog             hard backstop for the case where the reactor itself
#                        is wedged and the graceful timer never fires. macOS
#                        ships no timeout(1)/gtimeout, hence the sleep+kill.
#
# The crawl log goes to a file rather than stdout: at -L INFO it is the only
# way to see WHERE a stall happened, and the 25h hang was undiagnosable
# because its output sat in a pipe buffer.
# 3600s, not 1800s: 26 leagues (78 pages) ran at ~6 pages/min and finished in
# ~13 min, but 40 leagues (120 pages) was measured at ~2.5 pages/min — the rate
# drops as the crawl goes on, so the budget has to grow faster than the league
# count. The cap is a backstop against a stall, not a target: a healthy crawl
# should finish well inside it.
CRAWL_TIMEOUT_S=${STANDINGS_CRAWL_TIMEOUT_S:-3600}
CRAWL_LOG="logs/standings_crawl.log"
mkdir -p logs

scrapy crawl standings -L INFO -s CLOSESPIDER_TIMEOUT="$CRAWL_TIMEOUT_S" > "$CRAWL_LOG" 2>&1 &
CRAWL_PID=$!
# Deadline on the WALL CLOCK, polled — not one long `sleep`. macOS suspends
# sleep(1)'s countdown while the machine sleeps, so on 2026-09-30 a
# `sleep 3900` watchdog was still sleeping after 4h and a crawl wedged in its
# graceful close ran until killed by hand. date +%s keeps counting through
# system sleep, so the kill fires on the first poll after wake-up.
DEADLINE=$(( $(date +%s) + CRAWL_TIMEOUT_S + 300 ))
(
    while kill -0 "$CRAWL_PID" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            echo "[-] Watchdog: crawl passed its deadline — SIGKILL." >&2
            pkill -9 -P "$CRAWL_PID" 2>/dev/null   # Playwright driver + Chromium
            kill -9 "$CRAWL_PID" 2>/dev/null
            break
        fi
        sleep 30
    done
) &
WATCHDOG_PID=$!

wait "$CRAWL_PID"
CRAWL_STATUS=$?
kill "$WATCHDOG_PID" 2>/dev/null
wait "$WATCHDOG_PID" 2>/dev/null

if [ $CRAWL_STATUS -ne 0 ]; then
    echo "[-] Crawl exited $CRAWL_STATUS. Last lines of $CRAWL_LOG:" >&2
    tail -15 "$CRAWL_LOG" >&2
fi

if [ $CRAWL_STATUS -ne 0 ]; then
    echo "[-] Standings spider exited $CRAWL_STATUS — standings NOT updated." >&2
    exit $CRAWL_STATUS
fi

# Exit code alone is not enough. StandingsPipeline accumulates rows in memory
# and writes them in close_spider, so a crawl that matches nothing (Flashscore
# DOM change, blocked requests, Playwright failing to render) still closes
# cleanly with an empty data_store: it writes empty `[]` JSON and exits 0.
# Before 2026-09-18 this script echoed "complete" unconditionally; even after
# the -newer check, empty writes still looked like a successful refresh.
#
# `-newer <file>` is used rather than `-newermt '-N minutes'`: the relative
# form is GNU-only and silently matches nothing on macOS/BSD find, which is
# exactly the kind of false negative this check exists to avoid.
FRESH=$(find "$STANDINGS_DIR" -type f -name '*.json' -newer "$MARKER" 2>/dev/null | wc -l | tr -d ' ')

if [ "$FRESH" -eq 0 ]; then
    echo "[-] Standings spider exited 0 but wrote no files to $STANDINGS_DIR." >&2
    echo "    The crawl scraped nothing — standings on disk are STALE." >&2
    echo "    Check the spider against Flashscore's current DOM before trusting predictions." >&2
    exit 1
fi

# Reject the empty-[] false positive (pipeline always rewrites all 9 files).
ROWS=$(python3 -c "
import json, os
p = os.path.join('$STANDINGS_DIR', 'standings_overall.json')
try:
    print(len(json.load(open(p))))
except Exception:
    print(0)
")
if [ "$ROWS" -eq 0 ]; then
    echo "[-] Standings spider wrote empty files (0 rows in standings_overall.json)." >&2
    echo "    Likely cause: missing/broken $LINKS_CSV, or Flashscore DOM change." >&2
    exit 1
fi

echo "Standings update complete. ($FRESH files refreshed, $ROWS overall rows in $STANDINGS_DIR)"
