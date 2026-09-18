#!/bin/bash
#
# bin/setup_data.sh — one-shot historical-data setup for full app rollout.
#
# Sport-flagged orchestrator. Each sport's section is idempotent (skips when
# data already on disk), so you can re-run safely.
#
# Usage:
#   ./bin/setup_data.sh                       # all sports, defaults
#   ./bin/setup_data.sh --sport football      # one sport
#   ./bin/setup_data.sh --sport nba
#   ./bin/setup_data.sh --sport euroleague
#   ./bin/setup_data.sh --sport nt            # national teams (D7)
#   ./bin/setup_data.sh 2526                  # backwards-compat: football season code (= 2025-26)
#   ./bin/setup_data.sh --sport football 2526 # explicit sport + season
#   ./bin/setup_data.sh --help                # help
#
# What each sport pulls:
#   football   — football-data.co.uk CSVs   -> data_sets/MatchHistory/   (HTTP)
#                Flashscore standings/form  -> data_sets/standings/      (Scrapy + Playwright)
#   nba        — REQUIRES manual Kaggle archive drop into data_sets/NBA/archive/
#                Then builds the canonical corpus                        (local file processing)
#                Then daily fixtures + results refresh                   (nba_api)
#   euroleague — euroleague-api raw season CSVs (E+U, 2016-17 → 2024-25) -> data_sets/Euroleague/raw/
#   nt         — eloratings.net per-country TSVs -> data_sets/national_teams/international_matches.csv
#
# Rollout order matters: football + nt are HTTP-only and fast; nba needs
# the local archive present; euroleague iterates the API game-by-game and is
# the slowest (~30-40 min for the full 9-season E+U sweep).

set -u

# --- locate project root + activate venv ---------------------------------
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT="$(pwd)"

if [ ! -f venv/bin/activate ]; then
    echo "[setup_data] FATAL: venv/ not found at $PROJECT_ROOT/venv."
    echo "             Create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source venv/bin/activate

# PYTHONPATH for ml_project imports (mirrors bin/run_*.sh wrappers).
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/ml_project"

# --- arg parsing ---------------------------------------------------------
SPORT="all"
SEASON_CODE=""

print_usage() {
    sed -n '3,28p' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sport)
            if [ $# -lt 2 ]; then
                echo "[setup_data] --sport needs a value (football|nba|euroleague|nt|all)" >&2
                exit 1
            fi
            SPORT="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            # Backwards-compat: positional arg is a football season code (e.g. "2526").
            SEASON_CODE="$1"
            shift
            ;;
    esac
done

case "$SPORT" in
    all|football|nba|euroleague|nt|national_teams) ;;
    *)
        echo "[setup_data] unknown sport: '$SPORT' (expected football|nba|euroleague|nt|all)" >&2
        exit 1
        ;;
esac

echo "[setup_data] sport=$SPORT  season_code='${SEASON_CODE:-default}'  project_root=$PROJECT_ROOT"
echo ""

# --- helpers -------------------------------------------------------------
section() { printf "\n========== %s ==========\n" "$1"; }

# --- FOOTBALL ------------------------------------------------------------
setup_football() {
    section "FOOTBALL — football-data.co.uk match history"
    if [ -n "$SEASON_CODE" ]; then
        echo "[football] Downloading season $SEASON_CODE to data_sets/MatchHistory/ …"
        python3 scripts/setup_historical_data.py "$SEASON_CODE"
    else
        echo "[football] Interactive mode — pass a season code to skip prompts (e.g. ./bin/setup_data.sh 2526)."
        python3 scripts/setup_historical_data.py
    fi

    section "FOOTBALL — Flashscore standings + form (serving-time feature inputs)"
    echo "[football] Running standings spider; outputs to data_sets/standings/ via pipeline."
    if ! ./bin/update_leagues_data.sh; then
        echo "[football] WARNING: standings update failed. Daily prediction works without it, but feature engineering"
        echo "[football]          will fall back to stale data; rerun ./bin/update_leagues_data.sh when convenient."
    fi
}

# --- NBA -----------------------------------------------------------------
setup_nba() {
    section "NBA — local archive corpus"
    local archive_dir="data_sets/NBA/archive"
    local corpus="data_sets/NBA/team_game_stats.csv"

    if [ ! -d "$archive_dir" ] || [ -z "$(ls -A "$archive_dir" 2>/dev/null)" ]; then
        echo "[nba] WARNING: $archive_dir/ is missing or empty."
        echo "[nba]          The NBA training corpus comes from a manually-placed Kaggle snapshot."
        echo "[nba]          Expected files: Games.csv, TeamStatistics(Extended).csv, PlayByPlay.parquet, ..."
        echo "[nba]          Download the snapshot, drop it into $archive_dir/, then re-run."
        echo "[nba] Skipping NBA corpus build."
    elif [ -s "$corpus" ]; then
        echo "[nba] $corpus already exists ($(wc -l <"$corpus" | tr -d ' ') lines)."
        echo "[nba] Rebuild is merge-safe; re-running process_archive.py to pick up any new archive rows."
        python3 ml_project/nba/process_archive.py
    else
        echo "[nba] Building canonical corpus from $archive_dir/ → $corpus …"
        python3 ml_project/nba/process_archive.py
    fi

    section "NBA — daily fetch (yesterday's results + tomorrow's fixtures)"
    if [ -s "$corpus" ]; then
        echo "[nba] Appending yesterday's finished games via LeagueGameLog …"
        python3 ml_project/nba/fetch_nba_daily.py append-results || \
            echo "[nba] WARNING: append-results failed (off-season or transient API issue); continuing."
        echo "[nba] Pulling tomorrow's fixtures via ScoreboardV3 …"
        python3 ml_project/nba/fetch_nba_daily.py fixtures || \
            echo "[nba] WARNING: fixtures fetch failed (off-season or transient API issue); continuing."
    else
        echo "[nba] Skipping daily fetch — no corpus yet (see warning above)."
    fi
}

# --- EUROLEAGUE ----------------------------------------------------------
setup_euroleague() {
    section "EUROLEAGUE — euroleague-api raw seasons (E + EuroCup, 2016-17 → 2024-25)"
    local raw_dir="data_sets/Euroleague/raw"
    mkdir -p "$raw_dir"
    # Pre-flight: euroleague-api dep.
    if ! python3 -c "import euroleague_api" 2>/dev/null; then
        echo "[euroleague] euroleague-api not installed; running pip install …"
        pip install euroleague-api
    fi
    echo "[euroleague] Fetcher is idempotent (skips files already on disk)."
    echo "[euroleague] Full sweep takes ~30–40 min (game_report + game_stats × 18 (comp, season) pairs)."
    echo "[euroleague] Quarter scores skipped (not used by the model; rerun probe_euroleague_api.py if ever needed)."
    PYTHONUNBUFFERED=1 python3 scripts/euroleague_probe/fetch_seasons.py --start 2017 --end 2025 --comps E,U
}

# --- NATIONAL TEAMS (D7) -------------------------------------------------
setup_national_teams() {
    section "NATIONAL TEAMS — eloratings.net match-by-match (D7 corpus)"
    local out="data_sets/national_teams/international_matches.csv"
    if [ -s "$out" ]; then
        echo "[nt] $out already exists ($(wc -l <"$out" | tr -d ' ') lines)."
        echo "[nt] Re-running build_dataset.py — cached per-country TSVs make this fast."
    else
        echo "[nt] Building international corpus from eloratings.net per-country TSVs …"
    fi
    python3 scripts/national_teams/build_dataset.py || \
        echo "[nt] WARNING: corpus build failed; football pipeline still works (NT predictions just won't run)."
}

# --- dispatch ------------------------------------------------------------
case "$SPORT" in
    all)
        setup_football
        setup_nba
        setup_euroleague
        setup_national_teams
        ;;
    football)
        setup_football
        ;;
    nba)
        setup_nba
        ;;
    euroleague)
        setup_euroleague
        ;;
    nt|national_teams)
        setup_national_teams
        ;;
esac

# Deployment-state files (bankrolls, learned team names) are gitignored and
# ship as templates; seed them so a fresh clone does not start with zero
# bankrolls and an empty resolver. Never overwrites an existing file.
python3 -m ml_project.config_bootstrap || true

echo ""
echo "========== setup_data.sh DONE =========="
echo ""
echo "Next steps after a fresh setup:"
echo "  Football : ./bin/run_predictions.sh        (tomorrow's slate)"
echo "  NBA      : ./bin/run_nba_predictions.sh    (tomorrow's slate)"
echo "  UI       : ./bin/manage_server.sh start    (http://localhost:5001)"
echo ""
