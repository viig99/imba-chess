#!/usr/bin/env bash
# Confirmation re-run of the top candidates from sweep_adaptive_lam.sh's
# 15-game screen, at a larger game count to separate real signal from noise.
set -uo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
cd "${REPO_DIR}"
source .venv/bin/activate

OUT_DIR="${REPO_DIR}/artifacts/eval/adaptive_lam_sweep_confirm"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/sweep.log"
: > "${LOG}"

CHECKPOINT="artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"
CONFIG="config/imba_chess_exit_alpha_low.toml"
GAMES=45

# tag lam0 c_visit(or "none") -- best 3 candidates from the 15-game screen,
# no baseline (already have reference numbers for this checkpoint elsewhere).
CONFIGS=(
  "lam0.1_cv100 0.1 100"
  "lam0.2_cv50 0.2 50"
  "lam0.05_cv50 0.05 50"
)

for entry in "${CONFIGS[@]}"; do
  read -r tag lam0 c_visit <<< "${entry}"
  echo "=== ${tag} starting $(date) ===" >> "${LOG}"
  c_visit_args=()
  if [[ "${c_visit}" != "none" ]]; then
    c_visit_args=(--search-c-visit "${c_visit}")
  fi
  timeout 900 python scripts/eval_vs_stockfish.py \
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
echo "ALL_DONE" >> "${LOG}"
