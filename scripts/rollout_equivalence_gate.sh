#!/usr/bin/env bash
# Rollout equivalence gate: dense-mask SDPA vs flex_attention, end to end.
#
# The inference attention path now builds a dense [S,S] mask and runs SDPA
# instead of flex_attention. Mathematically equivalent, but not bit-identical:
# fp32 kernel reduction order differs, so a near-tie score can in principle
# flip and compound across a game's search. Unit tests cover a single forward;
# this covers a whole rollout through the real search.
#
# Both arms run the SAME build. The flex arm is produced by monkeypatching
# position_evaluator.create_batch_dense_mask to delegate to
# create_batch_block_mask, so the mask path is the only variable and no
# production code carries a test-only switch.
#
# Thresholds are the ones this repo already validated for cross-game batching
# (spec 2026-07-18, Layer 2): move agreement >= 99%, p99 |dv| <= 1e-3.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

GAMES="${GAMES:-20}"
BUDGET="${BUDGET:-2048}"
# 6, not 8: config/imba_chess_exit_seeded_rollout.toml documents that 8 OOMs at
# fp32/2048-budget on this 7.66 GiB card. expandable_segments matches what
# scripts/rollout_nightly_start.sh sets for the same reason (commit f16b2cf).
CONCURRENT="${CONCURRENT:-6}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${CONFIG:-config/imba_chess_exit_seeded_rollout.toml}"
CKPT="${CKPT:-artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt}"
OUT_DIR="${OUT_DIR:-artifacts/equivalence_gate}"
mkdir -p "${OUT_DIR}"

gen() {  # gen <label> <force_flex:0|1>
  IMBA_GATE_FORCE_FLEX="$2" .venv/bin/python - "$1" <<'PY' \
    > "${OUT_DIR}/$1.log" 2>&1
import os, runpy, sys

label = sys.argv[1]
if os.environ.get("IMBA_GATE_FORCE_FLEX") == "1":
    # Route the inference path back through flex_attention's BlockMask.
    from imba_chess.eval import position_evaluator as pe
    from imba_chess.model import create_batch_block_mask

    def _as_block_mask(seq_offsets, *, total_tokens=None, device=None):
        return create_batch_block_mask(
            seq_offsets, total_tokens=total_tokens, device=device
        )

    pe.create_batch_dense_mask = _as_block_mask
    print("GATE: forced flex_attention BlockMask path")
else:
    print("GATE: dense-mask SDPA path (production)")

sys.argv = [
    "generate_search_rollouts.py",
    "--config", os.environ["GATE_CONFIG"],
    "--checkpoint", os.environ["GATE_CKPT"],
    "--output-path", os.environ["GATE_OUT_DIR"] + "/" + label + ".parquet",
    "--max-games", os.environ["GATE_GAMES"],
    "--search-budget", os.environ["GATE_BUDGET"],
    "--concurrent-games", os.environ["GATE_CONCURRENT"],
    "--dtype", "float32",
    "--sample-seed", "42",
]
runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
PY
}

export GATE_CONFIG="${CONFIG}" GATE_CKPT="${CKPT}" GATE_OUT_DIR="${OUT_DIR}"
export GATE_GAMES="${GAMES}" GATE_BUDGET="${BUDGET}" GATE_CONCURRENT="${CONCURRENT}"

echo "== arm A: flex_attention (reference) =="
gen flex 1
echo "== arm B: dense-mask SDPA (production) =="
gen sdpa 0

.venv/bin/python - "${OUT_DIR}/flex.parquet" "${OUT_DIR}/sdpa.parquet" <<'PY'
import sys
import numpy as np
import pandas as pd

a = pd.read_parquet(sys.argv[1]).sort_values(["game_id", "ply"]).reset_index(drop=True)
b = pd.read_parquet(sys.argv[2]).sort_values(["game_id", "ply"]).reset_index(drop=True)

print(f"rows: flex={len(a)} sdpa={len(b)}")
if len(a) != len(b) or not a[["game_id", "ply"]].equals(b[["game_id", "ply"]]):
    print("FAIL: (game_id, ply) index differs -- games diverged structurally")
    sys.exit(1)

move_agree = (a["best_arm_move_uci"] == b["best_arm_move_uci"]).mean()
dv = (a["best_arm_backed_value"] - b["best_arm_backed_value"]).abs()
p99, mx = float(np.percentile(dv, 99)), float(dv.max())

print(f"best-arm move agreement : {move_agree:.4%}")
print(f"|d best_arm_value| p99  : {p99:.3e}   max: {mx:.3e}")

ok = move_agree >= 0.99 and p99 <= 1e-3
print("PASS" if ok else "FAIL", "(gate: move agreement >= 99%, p99 |dv| <= 1e-3)")
sys.exit(0 if ok else 1)
PY
