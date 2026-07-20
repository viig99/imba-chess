#!/usr/bin/env bash
# Idempotent nightly start: safe to call more than once (e.g. a one-off
# 9:30pm `at` job tonight followed by the recurring 11pm cron entry) --
# if a session is already running, this is a silent no-op.
#
# Bounded to a fixed number of nights (END_DATE, inclusive) rather than
# running forever: the recurring crontab entry itself has no "stop after N
# runs" mechanism, so the self-deactivation lives here instead of relying
# on a future session remembering to remove the crontab.
set -euo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
STATE_DIR="${REPO_DIR}/artifacts/rollouts/nightly"
PID_FILE="${STATE_DIR}/current.pid"
STATE_FILE="${STATE_DIR}/state.json"
CHECKPOINT="${REPO_DIR}/artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
CONFIG="${REPO_DIR}/config/imba_chess_exit_full.toml"
# 5 nights starting 2026-07-13: last allowed start is 2026-07-17 (its
# session runs until the 07:00 stop on 2026-07-18).
END_DATE="2026-07-17"

mkdir -p "${STATE_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

if [[ "$(date +%Y-%m-%d)" > "${END_DATE}" ]]; then
    echo "$(date): past END_DATE (${END_DATE}), not starting -- 5-night rollout-generation window is over" >> "${STATE_DIR}/nightly.log"
    exit 0
fi

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

# Unattended-night margin (2026-07-20): script default G=8 fp32 peaks 6.9GB on
# 20-game runs, but the CUDA allocator's high-water-mark creeps over long
# sessions (2026-07-15 notes) -- G=6 + expandable_segments buys OOM headroom
# for ~5% throughput, the right trade for month-long unattended operation.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
nohup python scripts/generate_search_rollouts.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-path "${output_path}" \
    --skip-games "${skip_games}" \
    --concurrent-games 6 \
    --flush-every-games 200 \
    >> "${STATE_DIR}/nightly.log" 2>&1 &

new_pid=$!
echo "${new_pid}" > "${PID_FILE}"
echo "${output_path}" > "${STATE_DIR}/current_output_path"
echo "$(date): launched pid ${new_pid}" >> "${STATE_DIR}/nightly.log"
