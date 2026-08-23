#!/usr/bin/env bash
# Phase B: does distilling Lichess's Stockfish evals into the value head help?
#
# Two arms differing in EXACTLY one thing: the value target on covered tokens.
#   control (beta=0) -- the one-hot game outcome, verified token-for-token
#                       identical to the target the model derives itself
#   distill (beta=1) -- the calibrated Stockfish WDL
#
# Both train on the SAME annotated-games corpus. That restriction is not free
# (annotation is self-selected by rating: 9.6% of games at 2000-2199 average
# Elo, 64.3% at 2600-2799), which is exactly why the control trains on the same
# file -- both arms see the identical stronger population, so the only
# remaining difference is the target.
#
# policy_kl_weight is forced to 0: the 2026-08-22 result settled policy-KL as a
# null, and leaving it on would confound the value result with a second change.
set -euo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
OUT_DIR="${REPO_DIR}/artifacts/value_distill"
CHECKPOINT="${REPO_DIR}/artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
BASE_CONFIG="${REPO_DIR}/config/imba_chess_exit_kl_probe_nw0.toml"
CORPUS="artifacts/corpus/train_600k_annotated.parquet"
TARGETS="artifacts/corpus/train_600k_value_targets.parquet"
LR="5e-5"
# 88,926 annotated games at ~52 games/step is ~1,710 steps for a single pass.
# Held under that: a restricted corpus repeated for many epochs would trade the
# thing being measured for overfitting.
STEPS="${STEPS:-1500}"

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}"
exec >> "${OUT_DIR}/run.log" 2>&1
echo "=============================================================="
echo "$(date): value-distillation paired run (steps=${STEPS}, lr=${LR})"

source .venv/bin/activate

declare -A ARM_BETA=( [control]="0.0" [distill]="1.0" )
for arm in control distill; do
    b="${ARM_BETA[$arm]}"
    cfg="${OUT_DIR}/config_${arm}.toml"
    ckpt_rel="artifacts/value_distill/ckpt_${arm}"
    sed -e "s|^beta = .*|beta = ${b}|" \
        -e "s|^policy_kl_weight = .*|policy_kl_weight = 0.0|" \
        -e "s|^rollout_path = .*|rollout_path = \"${TARGETS}\"|" \
        -e "s|^checkpoint_dir = .*|checkpoint_dir = \"${ckpt_rel}\"|" \
        -e "s|^train_month_shuffle_seed = .*|train_month_shuffle_seed = 42\nlocal_corpus_path = \"${CORPUS}\"|" \
        -e "s|^eval_every_steps = .*|eval_every_steps = 1000000|" \
        "${BASE_CONFIG}" > "${cfg}"
    # A silent mis-substitution makes the two arms identical and the whole
    # comparison meaningless -- fail loudly instead.
    grep -q "^beta = ${b}$" "${cfg}"                        || { echo "FATAL: beta rewrite failed (${arm})"; exit 1; }
    grep -q "^policy_kl_weight = 0.0$" "${cfg}"             || { echo "FATAL: policy_kl_weight rewrite failed"; exit 1; }
    grep -q "^local_corpus_path = \"${CORPUS}\"$" "${cfg}"  || { echo "FATAL: local_corpus_path rewrite failed"; exit 1; }
    grep -q "^rollout_path = \"${TARGETS}\"$" "${cfg}"      || { echo "FATAL: rollout_path rewrite failed"; exit 1; }
    grep -q "^num_workers = 0$" "${cfg}"                    || { echo "FATAL: num_workers must be 0 (sharding bug)"; exit 1; }

    echo "--------------------------------------------------------------"
    echo "$(date): TRAIN arm=${arm} beta=${b}"
    python scripts/train.py --config "${cfg}" --resume "${CHECKPOINT}" \
        --max-steps "${STEPS}" --lr-override "${LR}"
    echo "$(date): TRAIN arm=${arm} done"
done
echo "$(date): both arms trained"
