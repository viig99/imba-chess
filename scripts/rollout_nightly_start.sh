#!/usr/bin/env bash
# Idempotent nightly start: safe to call more than once (e.g. a one-off
# 9:30pm `at` job tonight followed by the recurring 11pm cron entry) --
# if a session is already running, this is a silent no-op.
set -euo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
STATE_DIR="${REPO_DIR}/artifacts/rollouts/nightly"
PID_FILE="${STATE_DIR}/current.pid"
STATE_FILE="${STATE_DIR}/state.json"
CHECKPOINT="${REPO_DIR}/artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
CONFIG="${REPO_DIR}/config/imba_chess_exit_full.toml"

mkdir -p "${STATE_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

if [[ -f "${PID_FILE}" ]]; then
    old_pid=$(cat "${PID_FILE}")
    if kill -0 "${old_pid}" 2>/dev/null; then
        echo "$(date): session already running (pid ${old_pid}), skipping start" >> "${STATE_DIR}/nightly.log"
        exit 0
    fi
    echo "$(date): stale pid file (${old_pid} not running), continuing" >> "${STATE_DIR}/nightly.log"
fi

skip_games=0
if [[ -f "${STATE_FILE}" ]]; then
    skip_games=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['total_games_covered'])")
fi

session_ts=$(date +%Y%m%d_%H%M%S)
output_path="${STATE_DIR}/session_${session_ts}.parquet"

echo "$(date): starting session, skip_games=${skip_games}, output=${output_path}" >> "${STATE_DIR}/nightly.log"

nohup python scripts/generate_search_rollouts.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-path "${output_path}" \
    --skip-games "${skip_games}" \
    --flush-every-games 200 \
    >> "${STATE_DIR}/nightly.log" 2>&1 &

new_pid=$!
echo "${new_pid}" > "${PID_FILE}"
echo "${output_path}" > "${STATE_DIR}/current_output_path"
echo "$(date): launched pid ${new_pid}" >> "${STATE_DIR}/nightly.log"
