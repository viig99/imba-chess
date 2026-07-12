#!/usr/bin/env bash
# Idempotent nightly stop: SIGTERM the running session (if any), then read
# its progress sidecar (safe even after a hard kill -- generate_search_rollouts.py
# flushes it periodically, atomically) and fold this session's coverage into
# the cumulative state.json so tomorrow's start picks up with the right
# --skip-games instead of redoing the same games.
set -euo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
STATE_DIR="${REPO_DIR}/artifacts/rollouts/nightly"
PID_FILE="${STATE_DIR}/current.pid"
STATE_FILE="${STATE_DIR}/state.json"
OUTPUT_PATH_FILE="${STATE_DIR}/current_output_path"

cd "${REPO_DIR}"
source .venv/bin/activate

if [[ ! -f "${PID_FILE}" ]]; then
    echo "$(date): no pid file, nothing to stop" >> "${STATE_DIR}/nightly.log"
    exit 0
fi

pid=$(cat "${PID_FILE}")
if kill -0 "${pid}" 2>/dev/null; then
    echo "$(date): stopping pid ${pid}" >> "${STATE_DIR}/nightly.log"
    kill -TERM "${pid}"
    # Give it a moment to exit; it has no signal handler so this just
    # confirms the OS finished tearing it down before we read its sidecar.
    for _ in $(seq 1 10); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
    done
    kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
else
    echo "$(date): pid ${pid} already exited" >> "${STATE_DIR}/nightly.log"
fi

if [[ -f "${OUTPUT_PATH_FILE}" ]]; then
    output_path=$(cat "${OUTPUT_PATH_FILE}")
    sidecar="${output_path}.progress.json"
    if [[ -f "${sidecar}" ]]; then
        python3 - "${sidecar}" "${STATE_FILE}" <<'PYEOF'
import json
import sys

sidecar_path, state_path = sys.argv[1], sys.argv[2]
progress = json.load(open(sidecar_path))

try:
    state = json.load(open(state_path))
except FileNotFoundError:
    state = {"total_games_covered": 0, "sessions": []}

state["total_games_covered"] = progress["total_games_covered"]
state.setdefault("sessions", []).append(progress)

tmp_path = state_path + ".tmp"
with open(tmp_path, "w") as f:
    json.dump(state, f)
import os
os.replace(tmp_path, state_path)
print(f"total_games_covered now {state['total_games_covered']}")
PYEOF
        echo "$(date): folded session progress into state.json" >> "${STATE_DIR}/nightly.log"
    else
        echo "$(date): WARNING no sidecar found at ${sidecar}, state.json unchanged" >> "${STATE_DIR}/nightly.log"
    fi
fi

rm -f "${PID_FILE}"
echo "$(date): stop complete" >> "${STATE_DIR}/nightly.log"
