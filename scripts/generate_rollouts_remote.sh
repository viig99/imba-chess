#!/usr/bin/env bash
# Launch N parallel rollout-generation shards on this machine.
#
# Meant to be run directly on a rented GPU box (e.g. inside byobu/screen so
# it survives disconnects), not via the local nightly cron -- that's
# scripts/rollout_nightly_start.sh, a separate single-process mechanism with
# its own artifacts/rollouts/nightly/ namespace. This script writes to
# artifacts/rollouts/remote/ instead, so a later `./sync_remote.sh pull`
# never collides with the local cron's own state.
#
# Usage (run from the repo root on the remote box):
#   scripts/generate_rollouts_remote.sh
#   NUM_SHARDS=20 scripts/generate_rollouts_remote.sh
#
# NUM_SHARDS default (16) was sized for a 23-core / 48GB-RAM container with
# an RTX 5090 -- this workload is RAM/CPU-bound, not GPU-bound (see
# docs/superpowers/notes/ if a throughput writeup exists, or just watch
# `free -h`). Re-tune NUM_SHARDS for a different box's specs.
#
# This does NOT autorestart crashed shards or track --skip-games across
# restarts -- watch the per-shard logs and each output file's
# .progress.json sidecar, and relaunch a single dead shard by hand with
# --skip-games set from its sidecar's total_games_covered if needed.
set -euo pipefail

CONFIG="${CONFIG:-config/imba_chess_exit_full.toml}"
CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/rollouts/remote}"
# 4, not the old 16: generate_search_rollouts.py now defaults to
# --concurrent-games 8 --dtype float32 (~7GB VRAM per process on the local
# 8GB card) -- 16 such processes would OOM a 32GB card immediately. Cross-game
# batching replaces most of what shard-parallelism was doing; re-tune
# shards x G on the 5090 per the cross-game-batched-search spec's follow-ups.
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${OUTPUT_DIR}"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  nohup .venv/bin/python scripts/generate_search_rollouts.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-path "${OUTPUT_DIR}/shard${i}of${NUM_SHARDS}.parquet" \
    --shard-id "${i}" --num-shards "${NUM_SHARDS}" \
    --flush-every-games 200 \
    > "${OUTPUT_DIR}/shard${i}.log" 2>&1 &
  echo "launched shard ${i}, pid $!"
done

echo ""
echo "${NUM_SHARDS} shards launched. Monitor with: htop / nvidia-smi -l 2 / watch -n5 free -h"
echo "Watch RAM specifically -- that was the binding constraint in local testing, not GPU."
