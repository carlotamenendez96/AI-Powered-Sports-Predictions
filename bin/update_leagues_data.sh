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
