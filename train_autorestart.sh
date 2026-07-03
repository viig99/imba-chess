#!/usr/bin/env bash
# Auto-restarting wrapper around scripts/train.py.
#
# On crash (internet outage, OOM, etc.) waits RESTART_DELAY_SEC and relaunches,
# resuming from the newest artifacts/checkpoints/last_checkpoint_*.pt at that
# moment (not a fixed path, so each restart picks up the latest progress).
# Stops on: clean exit (training finished), Ctrl+C/SIGTERM, or a crash loop
# (MAX_FAST_FAILS consecutive runs dying within FAST_FAIL_SEC — a real bug,
# not a transient failure).
#
# Usage: bash scripts/train_autorestart.sh [extra train.py args...]

set -u

CHECKPOINT_DIR="artifacts/checkpoints"
RESTART_DELAY_SEC=90
# The board encoder's checkpointed backward makes a couple of ~2 GiB requests;
# without expandable segments they fail on fragmentation, not real exhaustion.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
FAST_FAIL_SEC=120
MAX_FAST_FAILS=5

interrupted=0
trap 'interrupted=1' INT TERM

fast_fails=0
while true; do
    resume_args=()
    latest=$(ls "${CHECKPOINT_DIR}"/last_checkpoint_*.pt 2>/dev/null | sort -V | tail -n 1)
    if [[ -n "${latest}" ]]; then
        resume_args=(--resume "${latest}")
        echo "[autorestart $(date '+%F %T')] launching train.py --resume ${latest}"
    else
        echo "[autorestart $(date '+%F %T')] no last_checkpoint found, launching fresh run"
    fi

    start=$(date +%s)
    python scripts/train.py --device cuda --dtype bfloat16 "${resume_args[@]}" "$@"
    code=$?
    runtime=$(( $(date +%s) - start ))

    if [[ ${interrupted} -eq 1 || ${code} -eq 130 || ${code} -eq 143 ]]; then
        echo "[autorestart $(date '+%F %T')] interrupted, exiting."
        break
    fi
    if [[ ${code} -eq 0 ]]; then
        echo "[autorestart $(date '+%F %T')] train.py finished cleanly, exiting."
        break
    fi

    if (( runtime < FAST_FAIL_SEC )); then
        fast_fails=$(( fast_fails + 1 ))
        if (( fast_fails >= MAX_FAST_FAILS )); then
            echo "[autorestart $(date '+%F %T')] ${fast_fails} consecutive failures within ${FAST_FAIL_SEC}s — likely a real bug, giving up."
            exit 1
        fi
    else
        fast_fails=0
    fi

    echo "[autorestart $(date '+%F %T')] train.py died (exit ${code} after ${runtime}s); restarting in ${RESTART_DELAY_SEC}s (fast_fails=${fast_fails}/${MAX_FAST_FAILS})"
    sleep "${RESTART_DELAY_SEC}" &
    wait $! || true
    if [[ ${interrupted} -eq 1 ]]; then
        echo "[autorestart $(date '+%F %T')] interrupted during wait, exiting."
        break
    fi
done
