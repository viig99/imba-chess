#!/usr/bin/env bash
# One-off overnight run: continue alpha=0.1 training from wherever the last
# session left off, bounded to fit an ~11pm-5am window, then immediately
# measure actual playing strength vs Stockfish on the result. No -e: the
# eval phase must still run even if the training phase exits non-zero
# (e.g. hits its wall-clock timeout instead of --max-steps).
set -uo pipefail

REPO_DIR="/home/vigi99/CodeDir/imba-chess"
CHECKPOINT_DIR="${REPO_DIR}/artifacts/checkpoints_exit_alpha_low"
LOG="${REPO_DIR}/artifacts/nightly_alpha01_train_eval.log"

cd "${REPO_DIR}"
source .venv/bin/activate

resume_ckpt=$(ls -t "${CHECKPOINT_DIR}"/last_checkpoint_*.pt 2>/dev/null | head -1)
echo "=== nightly alpha=0.1 training starting $(date), resuming from ${resume_ckpt} ===" > "${LOG}"

# 21600s = 6h wall-clock backstop; --max-steps 70000 is the intended clean
# stop (~70k more steps at observed ~5.8 steps/sec fits well inside 6h with
# headroom for the periodic full-val passes at each 10k-step epoch boundary).
timeout 21600 python scripts/train.py \
    --config config/imba_chess_exit_alpha_low.toml \
    --resume "${resume_ckpt}" \
    --max-steps 70000 >> "${LOG}" 2>&1
echo "=== training phase exit=$? done $(date) ===" >> "${LOG}"

final_ckpt=$(ls -t "${CHECKPOINT_DIR}"/last_checkpoint_*.pt 2>/dev/null | head -1)
echo "=== stockfish eval starting $(date) against ${final_ckpt} ===" >> "${LOG}"

mkdir -p "${REPO_DIR}/artifacts/eval"
# Reduced from the config's default ladder_games_per_segment=100 to 40 (5
# segments x 40 = 200 games) to keep this bounded overnight, while keeping
# search_budget=2048/depth=8 unchanged so the result is comparable to the
# established live-eval protocol.
timeout 7200 python scripts/eval_vs_stockfish.py \
    --config config/imba_chess_exit_alpha_low.toml \
    --checkpoint "${final_ckpt}" \
    --ladder-games-per-segment 40 \
    --output-json artifacts/eval/nightly_alpha01_vs_stockfish.json >> "${LOG}" 2>&1
echo "=== stockfish eval exit=$? done $(date) ===" >> "${LOG}"
echo "ALL_DONE" >> "${LOG}"
