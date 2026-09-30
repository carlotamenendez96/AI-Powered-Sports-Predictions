# Single-instance lock for the bin/ wrappers. Source it, then:
#
#   acquire_lock <name> || exit $EXIT_LOCKED
#
# Why: the web UI tracks running pipelines only in memory, so a server restart
# forgets them, the button reads idle again, and a second click starts a
# duplicate. Measured 2026-09-30: four run_predictions.sh and three standings
# crawls running at once, all truncating the same logs/predict.log and racing
# on data_sets/standings/. The lock lives on disk, so it survives restarts.
#
# mkdir is the lock primitive because it is atomic on every filesystem and
# needs no flock(1), which macOS does not ship. The holder's PID is stored
# inside; a lock whose PID is gone (the holder was SIGKILLed, so its EXIT trap
# never ran) is treated as stale and taken over.
#
# The lock is released by an EXIT trap. A caller that needs its own EXIT trap
# must call release_lock from it, since bash keeps one trap per signal.

EXIT_LOCKED=4
LOCK_DIR=""

acquire_lock() {
    local dir="logs/.lock_$1"
    mkdir -p logs
    if ! mkdir "$dir" 2>/dev/null; then
        local pid
        pid=$(cat "$dir/pid" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[-] Another '$1' run is already active (PID $pid, since $(cat "$dir/started" 2>/dev/null))." >&2
            echo "    Not starting a duplicate. Stop it first, or wait for it to finish." >&2
            return 1
        fi
        echo "[*] Clearing stale '$1' lock (PID ${pid:-?} is gone)."
        rm -rf "$dir"
        mkdir "$dir" 2>/dev/null || { echo "[-] Lost the race for the '$1' lock." >&2; return 1; }
    fi
    echo $$ > "$dir/pid"
    date "+%Y-%m-%d %H:%M:%S" > "$dir/started"
    LOCK_DIR="$dir"
    trap release_lock EXIT
    return 0
}

release_lock() {
    [ -n "$LOCK_DIR" ] && rm -rf "$LOCK_DIR"
    LOCK_DIR=""
}
