#!/usr/bin/env bash
# Evaluate the best checkpoint so far vs Elo-limited Stockfish, once per policy.
#
# Usage:
#   scripts/eval_best_checkpoint.sh                 # greedy + value_search_d2, 100 games vs SF1400
#   POLICIES="greedy" scripts/eval_best_checkpoint.sh
#   ELO=1600 GAMES=50 scripts/eval_best_checkpoint.sh
#   scripts/eval_best_checkpoint.sh --debug-trace-games 3   # extra args pass through to eval script
#
# Picks the best_hr10 checkpoint with the highest hr10 in its filename.
# Skips a policy if its output JSON already exists (delete it to re-run).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
OUT_DIR="${OUT_DIR:-artifacts/eval}"
ELO="${ELO:-1400}"
GAMES="${GAMES:-100}"
POLICIES="${POLICIES:-greedy value_search_d2}"

best=$(ls "${CHECKPOINT_DIR}"/best_hr10_checkpoint_*.pt 2>/dev/null | sort -t= -k2 -g | tail -n 1)
if [[ -z "${best}" ]]; then
    echo "No best_hr10_checkpoint_*.pt found in ${CHECKPOINT_DIR}" >&2
    exit 1
fi

tag=$(basename "${best}" .pt | tr '=' '-')
mkdir -p "${OUT_DIR}"
echo "Checkpoint: ${best}"
echo "Policies:   ${POLICIES}"
echo "Opponent:   SF elo=${ELO}, games=${GAMES}"

for policy in ${POLICIES}; do
    out_json="${OUT_DIR}/${tag}_sf${ELO}_${policy}.json"
    if [[ -f "${out_json}" ]]; then
        echo "== ${policy}: ${out_json} already exists, skipping =="
        continue
    fi
    echo "== ${policy} =="
    python scripts/eval_vs_stockfish.py \
        --checkpoint "${best}" \
        --no-compile \
        --model-move-policy "${policy}" \
        --ladder-elos "${ELO}" \
        --ladder-games-per-segment "${GAMES}" \
        --no-include-full-strength-segment \
        --output-json "${out_json}" \
        "$@"
    echo "== ${policy} -> ${out_json} =="
done
