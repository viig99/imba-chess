#!/usr/bin/env bash
# Auto-restarting wrapper around scripts/train.py.
#
# On crash (internet outage, OOM, etc.) waits RESTART_DELAY_SEC and relaunches,
# resuming from the newest last_checkpoint_*.pt in the RUN'S checkpoint
# directory at that moment (not a fixed path, so each restart picks up the
# latest progress). Stops on: clean exit (training finished), Ctrl+C/SIGTERM,
# or a crash loop (MAX_FAST_FAILS consecutive runs dying within FAST_FAIL_SEC —
# a real bug, not a transient failure).
#
# The checkpoint directory comes from the --config TOML's [training]
# checkpoint_dir, NOT a hardcoded path. A run whose config points elsewhere
# (config/imba_chess_v4.toml writes to artifacts/checkpoints_v4) would
# otherwise find no checkpoint and silently restart from step 0 on every
# crash, which is both silent and expensive. Parsing failure is fatal here for
# the same reason.
#
# Usage:
#   ./train_autorestart.sh --config config/imba_chess_v4.toml
#   ./train_autorestart.sh --config config/imba_chess.toml --max-games 2000000
#
# Env overrides: PYTHON, CHECKPOINT_DIR, LOG_FILE, RESTART_DELAY_SEC.

set -u

PYTHON="${PYTHON:-.venv/bin/python}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-90}"
FAST_FAIL_SEC=120
MAX_FAST_FAILS=5
# The board encoder's checkpointed backward makes a couple of ~2 GiB requests;
# without expandable segments they fail on fragmentation, not real exhaustion.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON}" ]] && ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "error: interpreter '${PYTHON}' not found. Run 'uv sync' first, or set PYTHON=." >&2
    exit 1
fi

# Resolve the checkpoint directory: an explicit CHECKPOINT_DIR wins, otherwise
# read it out of the --config TOML, otherwise the repo default.
config_path=""
want_config=0
for arg in "$@"; do
    if (( want_config )); then
        config_path="${arg}"
        want_config=0
        continue
    fi
    case "${arg}" in
        --config) want_config=1 ;;
        --config=*) config_path="${arg#--config=}" ;;
    esac
done

if [[ -z "${CHECKPOINT_DIR:-}" && -n "${config_path}" ]]; then
    if [[ ! -f "${config_path}" ]]; then
        echo "error: --config '${config_path}' does not exist." >&2
        exit 1
    fi
    CHECKPOINT_DIR=$(sed -n \
        's/^[[:space:]]*checkpoint_dir[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
        "${config_path}" | head -n 1)
    if [[ -z "${CHECKPOINT_DIR}" ]]; then
        echo "error: no [training] checkpoint_dir found in '${config_path}'." >&2
        echo "       Refusing to guess: the wrong directory means every restart" >&2
        echo "       silently begins again from step 0." >&2
        exit 1
    fi
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
mkdir -p "${CHECKPOINT_DIR}"
LOG_FILE="${LOG_FILE:-${CHECKPOINT_DIR}/train.log}"

echo "[autorestart $(date '+%F %T')] python=${PYTHON} checkpoint_dir=${CHECKPOINT_DIR} log=${LOG_FILE}"

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
        echo "[autorestart $(date '+%F %T')] no last_checkpoint in ${CHECKPOINT_DIR}, launching fresh run"
    fi

    start=$(date +%s)
    "${PYTHON}" scripts/train.py --device cuda --dtype bfloat16 \
        "${resume_args[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
    code=${PIPESTATUS[0]}
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
