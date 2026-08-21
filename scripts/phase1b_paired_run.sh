#!/usr/bin/env bash
# Phase 1b paired long run: policy_kl_weight 0.1 vs 0.0, everything else identical.
#
# This is the first Phase 1b test that can actually measure anything. Every
# prior run was a silent no-op: dataloader.num_workers>0 makes
# TorchLichessIterableDataset shard the stream (num_shards = world_size *
# num_workers) at the parquet-file level, while generate_search_rollouts.py
# runs unsharded -- so the (game_id, ply) lookup missed on every token and
# has_rollout_*_target was all zeros. Measured 2026-07-25: unsharded stream
# 10.0% coverage, every 4-way shard 0.0%. Hence num_workers=0 below; it is
# load-bearing, not a perf tweak.
#
# Protocol fixed by the user 2026-07-26: all steps the corpus supports,
# lr 5e-5 (the KL-off probe's cliff: >=2e-4 degrades, <=5e-5 improves),
# policy_kl_weight 0.1, then SF2200 at 200 games/arm.
set -euo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
OUT_DIR="${REPO_DIR}/artifacts/phase1b_paired"
CHECKPOINT="${REPO_DIR}/artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
ROLLOUTS="artifacts/rollouts/nightly/session_seed42_aligned_22k.parquet"
BASE_CONFIG="${REPO_DIR}/config/imba_chess_exit_kl_probe_nw0.toml"
LR="5e-5"
# Rollout coverage spans stream positions 0..222261 at ~51.96 games/step
# (max 4277). Held slightly under so the tail of training still has coverage.
STEPS=4200
EVAL_GAMES=200

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}"
exec >> "${OUT_DIR}/paired.log" 2>&1
echo "=============================================================="
echo "$(date): Phase 1b paired run starting (steps=${STEPS}, lr=${LR})"

PID_FILE="${REPO_DIR}/artifacts/rollouts/nightly/current.pid"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "$(date): ABORT -- rollout generation is running; refusing to contend for the GPU"
    exit 1
fi

source .venv/bin/activate

declare -A ARM_WEIGHT=( [kl01]="0.1" [kl00]="0.0" )
for arm in kl01 kl00; do
    w="${ARM_WEIGHT[$arm]}"
    cfg="${OUT_DIR}/config_${arm}.toml"
    ckpt_rel="artifacts/phase1b_paired/ckpt_${arm}"
    sed -e "s|^policy_kl_weight = .*|policy_kl_weight = ${w}|" \
        -e "s|^rollout_path = .*|rollout_path = \"${ROLLOUTS}\"|" \
        -e "s|^checkpoint_dir = .*|checkpoint_dir = \"${ckpt_rel}\"|" \
        "${BASE_CONFIG}" > "${cfg}"
    # A silent mis-substitution here would make the two arms identical and the
    # whole comparison meaningless -- fail loudly instead.
    grep -q "^policy_kl_weight = ${w}$" "${cfg}" || { echo "FATAL: kl weight rewrite failed (${arm})"; exit 1; }
    grep -q "^checkpoint_dir = \"${ckpt_rel}\"$" "${cfg}" || { echo "FATAL: ckpt dir rewrite failed (${arm})"; exit 1; }
    grep -q "^num_workers = 0$" "${cfg}" || { echo "FATAL: num_workers must be 0 (sharding bug)"; exit 1; }

    echo "--------------------------------------------------------------"
    echo "$(date): TRAIN arm=${arm} policy_kl_weight=${w}"
    python scripts/train.py --config "${cfg}" --resume "${CHECKPOINT}" \
        --max-steps "${STEPS}" --lr-override "${LR}"
    echo "$(date): TRAIN arm=${arm} done"
done

echo "=============================================================="
echo "$(date): training complete -- verifying the KL arm was NOT a no-op"
python scripts/lr_probe_summarize.py --probe-dir "${OUT_DIR}" || true

for arm in kl01 kl00; do
    ck=$(ls -t "${OUT_DIR}/ckpt_${arm}"/last_checkpoint_*.pt 2>/dev/null | head -1)
    if [[ -z "${ck}" ]]; then echo "FATAL: no checkpoint for ${arm}"; exit 1; fi
    echo "--------------------------------------------------------------"
    echo "$(date): EVAL arm=${arm} (${EVAL_GAMES} games vs SF2200) ckpt=${ck}"
    python scripts/eval_vs_stockfish.py \
        --config "${OUT_DIR}/config_${arm}.toml" \
        --checkpoint "${ck}" \
        --output-json "${OUT_DIR}/eval_${arm}.json"
    echo "$(date): EVAL arm=${arm} done"
done

echo "=============================================================="
echo "$(date): Phase 1b paired run finished"
for arm in kl01 kl00; do
    python - "$arm" "${OUT_DIR}/eval_${arm}.json" <<'PYEOF'
import json, sys
arm, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
segs = d.get("segments", [])
for s in segs:
    r = s.get("results", s)
    print(f"  {arm}: label={s.get('label', '?')} score_rate={r.get('score_rate')} "
          f"W{r.get('wins')}/D{r.get('draws')}/L{r.get('losses')}")
PYEOF
done
