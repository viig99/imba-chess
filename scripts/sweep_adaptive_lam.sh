#!/usr/bin/env bash
# One-off quick sweep: lam0 x c_visit at a cheap search budget/depth, vs a
# single fixed Stockfish level chosen to avoid ceiling/floor saturation
# (SF2000 gave score_rate=0.30 in calibration, well clear of both ends).
set -uo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
cd "${REPO_DIR}"
source .venv/bin/activate

OUT_DIR="${REPO_DIR}/artifacts/eval/adaptive_lam_sweep"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/sweep.log"
: > "${LOG}"

CHECKPOINT="artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
CONFIG="config/imba_chess_exit_alpha_low.toml"
GAMES=15

for lam0 in 0.05 0.1 0.2; do
  for c_visit in none 25 50 100; do
    tag="lam${lam0}_cv${c_visit}"
    echo "=== ${tag} starting $(date) ===" >> "${LOG}"
    c_visit_args=()
    if [[ "${c_visit}" != "none" ]]; then
      c_visit_args=(--search-c-visit "${c_visit}")
    fi
    timeout 600 python scripts/eval_vs_stockfish.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --ladder-elos "2000" --ladder-games-per-segment "${GAMES}" \
        --no-include-full-strength-segment \
        --search-budget 256 --search-max-depth 4 \
        --value-rerank-lambda "${lam0}" \
        "${c_visit_args[@]}" \
        --no-compile --no-save-games --debug-trace-games 0 \
        --output-json "${OUT_DIR}/${tag}.json" \
        >> "${LOG}" 2>&1
    echo "=== ${tag} exit=$? done $(date) ===" >> "${LOG}"
  done
done
echo "ALL_DONE" >> "${LOG}"
