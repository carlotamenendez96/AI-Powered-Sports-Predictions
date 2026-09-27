#!/bin/bash

# Change directory to project root
cd "$(dirname "$0")/.." || exit
# Activate virtual environment
source venv/bin/activate

STANDINGS_DIR="data_sets/standings"

# Marker for the freshness check below. Created BEFORE the crawl so any file
# the pipeline writes is strictly newer than it.
MARKER="$(mktemp -t standings_run_marker)"
trap 'rm -f "$MARKER"' EXIT

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
( sleep $((CRAWL_TIMEOUT_S + 300)); kill -9 "$CRAWL_PID" 2>/dev/null ) &
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
# cleanly with an empty data_store: it writes NO files and exits 0. Before
# 2026-09-18 this script echoed "complete" unconditionally, so that case looked
# identical to success and retrain_pipeline.sh would carry on with standings
# that could be arbitrarily stale — and standings feed inference-time team
# strength via HeuristicAdjuster.get_team_strength, so the damage lands in
# predictions rather than anywhere obvious.
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

echo "Standings update complete. ($FRESH files refreshed in $STANDINGS_DIR)"
