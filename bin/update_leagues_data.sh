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

# Run the standings spider
# No -O because the pipeline handles the files.
# Run the standings spider (Silenced)
scrapy crawl standings -L WARNING
CRAWL_STATUS=$?

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
