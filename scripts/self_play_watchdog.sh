#!/usr/bin/env bash
# Restart a self-play run with --resume after it crashes.
#
# usage: scripts/self_play_watchdog.sh RUN_DIR -- <run_self_play.py args without --resume/--initialize>
#
# Every crash is logged to RUN_DIR/watchdog.log with its exit code and last error line. The watchdog
# stops instead of restarting when the run exits cleanly (exit 0, e.g. after SIGTERM), when
# RUN_DIR/state.json says halted, when RUN_DIR/STOP exists, or after MAX_CRASHES crashes within
# WINDOW_SECONDS, so a persistent fault fails loudly rather than looping. To stop everything: touch
# RUN_DIR/STOP, then SIGTERM the run_self_play.py process and let it drain.
set -u
RUN_DIR=$1
shift
[ "$1" = "--" ] && shift
MAX_CRASHES=${MAX_CRASHES:-3}
WINDOW_SECONDS=${WINDOW_SECONDS:-3600}
LOG="$RUN_DIR/watchdog.log"
crashes=()

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

while true; do
    if [ -e "$RUN_DIR/STOP" ]; then log "STOP file present; not starting"; exit 0; fi
    log "starting run_self_play.py --resume $*"
    .venv/bin/python scripts/run_self_play.py --resume "$@" --output "$RUN_DIR" < /dev/null
    code=$?
    if [ $code -eq 0 ]; then log "run exited cleanly (0); watchdog exiting"; exit 0; fi
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$RUN_DIR/state.json')).get('halted') else 1)"; then
        log "CRASH exit=$code and state.json is halted; investigate, not restarting"; exit 1
    fi
    now=$(date +%s)
    crashes+=("$now")
    recent=()
    for t in "${crashes[@]}"; do [ $((now - t)) -lt "$WINDOW_SECONDS" ] && recent+=("$t"); done
    crashes=("${recent[@]}")
    log "CRASH exit=$code (${#crashes[@]} in the last ${WINDOW_SECONDS}s)"
    if [ "${#crashes[@]}" -ge "$MAX_CRASHES" ]; then
        log "CRASH limit reached; not restarting"; exit 1
    fi
    sleep 60
done
