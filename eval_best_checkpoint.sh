#!/usr/bin/env bash
# Evaluate the best checkpoint so far vs Elo-limited Stockfish, once per policy.
#
# Usage:
#   ./eval_best_checkpoint.sh                        # greedy + value_search_d2, 100 games vs SF1400
#   POLICIES="greedy" ./eval_best_checkpoint.sh
#   ELO=1600 GAMES=50 ./eval_best_checkpoint.sh
#   ./eval_best_checkpoint.sh --debug-trace-games 3  # extra args pass through to eval script
#
# value_search_halving (MCTS-lite; knobs default from [eval_vs_stockfish] in the TOML):
#   POLICIES="value_search_halving" ELO=1800 ./eval_best_checkpoint.sh
#   POLICIES="value_search_halving" ELO=1800 TAG=b512 \
#       ./eval_best_checkpoint.sh --search-budget 512
#   POLICIES="value_search_halving" ELO=1800 TAG=b512d6 \
#       ./eval_best_checkpoint.sh --search-budget 512 --search-max-depth 6
#   POLICIES="value_search_halving" ELO=1800 TAG=beam \
#       ./eval_best_checkpoint.sh --halving-rounds 1   # pure-beam attribution control
#   Other halving flags: --search-top-m, --search-refutation-top-r, --search-expand-top.
#   Sweep one knob per run (100 games ~ +-0.05 score SE); knobs are recorded in the
#   output JSON under run_config.search.
#
# Picks the best_hr10 checkpoint with the highest hr10 in its filename.
# Skips a policy if its output JSON already exists (delete it to re-run).
# TAG=<label> is appended to output filenames — REQUIRED to keep sweep runs of the
# same policy from colliding with (and being skipped in favor of) an earlier run.
set -euo pipefail

# CONFIG passthrough (2026-07-19): the eval script defaults to config/imba_chess.toml;
# both tomls now carry the node-limited fp32 actor-mode protocol, but always pass
# CONFIG explicitly for non-default setups.
CONFIG="${CONFIG:-config/imba_chess.toml}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
OUT_DIR="${OUT_DIR:-artifacts/eval}"
ELO="${ELO:-1400}"
GAMES="${GAMES:-100}"
POLICIES="${POLICIES:-greedy value_search_d2}"
SUFFIX="${TAG:+_${TAG}}"

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
    out_json="${OUT_DIR}/${tag}_sf${ELO}_${policy}${SUFFIX}.json"
    if [[ -f "${out_json}" ]]; then
        echo "== ${policy}: ${out_json} already exists, skipping =="
        continue
    fi
    echo "== ${policy} =="
    python scripts/eval_vs_stockfish.py \
        --config "${CONFIG}" \
        --checkpoint "${best}" \
        --no-compile \
        --model-move-policy "${policy}" \
        --ladder-elos "${ELO}" \
        --ladder-games-per-segment "${GAMES}" \
        --no-include-full-strength-segment \
        --save-games-dir "${OUT_DIR}/games/${tag}_sf${ELO}_${policy}${SUFFIX}" \
        --output-json "${out_json}" \
        "$@"
    echo "== ${policy} -> ${out_json} =="
done
