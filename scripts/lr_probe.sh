#!/usr/bin/env bash
# Learning-rate probe for short fine-tunes from a converged checkpoint.
#
# Motivation (2026-07-24): the 750-step continuation runs used to evaluate
# Phase 1b were resuming checkpoint_23 at the OneCycle schedule's lr of
# ~6.2e-4, and at that rate plain continuation does not improve -- train
# policy_loss ROSE 2.13 -> 2.22 over 750 steps. The runs were random-walking a
# converged model, which is the real source of the ~0.13pt "seed variance"
# that made every Phase 1b comparison unreadable. This probe asks the cheap
# question first: at what lr does continuation stop degrading?
#
# Deliberately uses NO Stockfish eval -- the readout is the training loss curve
# plus train.py's own fast-val hr@10, both free. At ~111 ms/step a 750-step run
# is ~82s of compute, so the whole sweep is minutes, not hours.
set -euo pipefail

# Overridable so the same sweep can run with the KL term off (default: the
# no-rollout base config) or on (pass BASE_CONFIG/OUT_DIR for a KL config).
REPO_DIR="/home/vigi99/CodeDir/imba-chess"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/artifacts/lr_probe}"
CHECKPOINT="${REPO_DIR}/artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
BASE_CONFIG="${BASE_CONFIG:-${REPO_DIR}/config/imba_chess_exit_seeded_rollout_norollout.toml}"
STEPS="${STEPS:-750}"
LRS=("6.2e-4" "2e-4" "5e-5" "1e-5")

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}"
exec >> "${OUT_DIR}/probe.log" 2>&1
echo "=============================================================="
echo "$(date): lr probe starting"

# The nightly rollout job holds ~6.7GB of the 8GB card. Training alongside it
# would OOM one or both, so refuse to start rather than kill the overnight run.
PID_FILE="${REPO_DIR}/artifacts/rollouts/nightly/current.pid"
if [[ -f "${PID_FILE}" ]]; then
    gen_pid="$(cat "${PID_FILE}")"
    if kill -0 "${gen_pid}" 2>/dev/null; then
        echo "$(date): ABORT -- rollout generation still running (pid ${gen_pid})."
        echo "  The 07:00 stop cron should have ended it. Investigate before rerunning."
        exit 1
    fi
fi

source .venv/bin/activate

for lr in "${LRS[@]}"; do
    tag="lr${lr}"
    cfg="${OUT_DIR}/config_${tag}.toml"
    # Path must be repo-relative for the config, but derived from OUT_DIR --
    # hardcoding "artifacts/lr_probe" here silently sent a KL-on sweep's output
    # into the KL-off sweep's directories on 2026-07-25.
    ckpt_rel="${OUT_DIR#${REPO_DIR}/}/ckpt_${tag}"
    sed "s|^checkpoint_dir = .*|checkpoint_dir = \"${ckpt_rel}\"|" \
        "${BASE_CONFIG}" > "${cfg}"
    # Fail loudly if the substitution missed -- a probe that silently wrote all
    # four runs into one checkpoint dir would corrupt every result.
    if ! grep -q "^checkpoint_dir = \"${ckpt_rel}\"" "${cfg}"; then
        echo "$(date): FATAL -- checkpoint_dir rewrite failed for ${tag}"
        exit 1
    fi

    echo "--------------------------------------------------------------"
    echo "$(date): run ${tag} (${STEPS} steps, lr=${lr})"
    python scripts/train.py \
        --config "${cfg}" \
        --resume "${CHECKPOINT}" \
        --max-steps "${STEPS}" \
        --lr-override "${lr}"
    echo "$(date): run ${tag} done"
done

echo "=============================================================="
echo "$(date): all runs complete, extracting metrics"
python scripts/lr_probe_summarize.py --probe-dir "${OUT_DIR}" | tee "${OUT_DIR}/summary.txt"
echo "$(date): lr probe finished"
