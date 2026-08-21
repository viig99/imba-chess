#!/usr/bin/env bash
# Measure how rollout-generation cost scales with search_budget.
#
# Why this exists: a closed RL/ExIt loop needs orders of magnitude more
# search-labeled positions per hour than the production 2048-eval protocol
# can produce. Gumbel MuZero (Danihelka et al., ICLR 2022) gets a policy-
# improvement target from as few as 16-50 simulations, so the budget is the
# obvious throughput lever -- but only if cost actually falls with it, which
# is not obvious in an overhead-bound, CPU-dominated pipeline. This sweep
# measures s/game and s/labeled-position at fixed games, fixed seed, fixed G.
#
# Read the output as: does throughput scale ~linearly with budget, or does a
# fixed per-position floor dominate? The answer decides whether cheap-search
# RL is feasible on this hardware at all.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

CONFIG="${CONFIG:-config/imba_chess_exit_seeded_rollout.toml}"
CKPT="${CKPT:-artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt}"
GAMES="${GAMES:-30}"
GROUP="${GROUP:-8}"
BUDGETS="${BUDGETS:-2048 512 128 32}"
OUT_DIR="${OUT_DIR:-artifacts/budget_sweep}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"
: > "${SUMMARY}"

echo "config=${CONFIG} ckpt=${CKPT} games=${GAMES} G=${GROUP}" | tee -a "${SUMMARY}"
echo | tee -a "${SUMMARY}"

for b in ${BUDGETS}; do
  log="${OUT_DIR}/budget_${b}.log"
  out="${OUT_DIR}/budget_${b}.parquet"
  rm -f "${out}"
  echo "=== search_budget=${b} ===" | tee -a "${SUMMARY}"
  start=$(date +%s.%N)
  .venv/bin/python scripts/generate_search_rollouts.py \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --output-path "${out}" \
    --max-games "${GAMES}" \
    --search-budget "${b}" \
    --concurrent-games "${GROUP}" \
    --dtype float32 \
    --sample-seed 42 \
    --profile --profile-every-games "${GAMES}" \
    > "${log}" 2>&1
  end=$(date +%s.%N)
  wall=$(echo "${end} - ${start}" | bc)

  rows=$(.venv/bin/python - "${out}" <<'PY'
import sys, pandas as pd
try:
    print(len(pd.read_parquet(sys.argv[1], columns=["game_id"])))
except Exception:
    print(0)
PY
)
  # The profile block's own total is the authoritative steady-state number:
  # wall time additionally carries ~30s of one-off startup (model load, CUDA
  # init, flex_attention warmup) that a long run amortizes away.
  prof=$(tr '\r' '\n' < "${log}" | grep -E "search_(gpu|bookkeeping)|root_eval|timing after" | tail -4 || true)
  echo "${prof}" | sed 's/^/    /' | tee -a "${SUMMARY}"
  printf '  wall %.1fs (incl. startup) | %d labeled positions\n' "${wall}" "${rows}" \
    | tee -a "${SUMMARY}"
  echo | tee -a "${SUMMARY}"
done

# Steady-state throughput, from each run's profile total rather than wall time.
echo "=== steady-state throughput (profile total; excludes one-off startup) ===" \
  | tee -a "${SUMMARY}"
printf '%8s %9s %9s %11s %10s %12s\n' \
  budget s/game games/hr s/label labels/hr "fresh-steps/hr" | tee -a "${SUMMARY}"
for b in ${BUDGETS}; do
  log="${OUT_DIR}/budget_${b}.log"
  tr '\r' '\n' < "${log}" | grep -oE "timing after [0-9]+ games / [0-9]+ positions .*total [0-9.]+s" \
    | tail -1 \
    | sed -E 's/timing after ([0-9]+) games \/ ([0-9]+) positions.*total ([0-9.]+)s/\1 \2 \3/' \
    | awk -v b="${b}" '{
        g=$1; p=$2; t=$3;
        if (g==0 || t==0) next;
        spg=t/g; spl=t/p;
        # 51.96 games/training-step, measured (rollout-training-cost-constants)
        printf "%8s %9.2f %9.0f %11.4f %10.0f %12.1f\n", \
          b, spg, 3600/spg, spl, 3600/spl, (3600/spg)/51.96
      }' | tee -a "${SUMMARY}"
done

echo "wrote ${SUMMARY}"
