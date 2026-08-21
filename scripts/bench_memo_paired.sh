#!/usr/bin/env bash
# Paired A/B/B/A benchmark for the memoized cozy-move -> vocab-id lookup.
#
# Why paired: absolute wall time is not comparable across sessions on this
# machine (~±10% drift; see docs/superpowers/specs/
# 2026-07-18-rollout-cpu-hotpath-optimization-design.md §6). A naive
# before/after across sessions measured -14% for a change that cannot touch
# GPU time at all. So run both arms inside one session, in A,B,B,A order so
# monotonic thermal drift cancels between arms, and use search_gpu as a
# drift control -- it must come out unchanged.
#
#   A = uncached reference (cozy_move_to_uci + dict lookup; the old path)
#   B = memoized _cozy_move_id_and_uci (production)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export BENCH_GAMES="${GAMES:-20}"
export BENCH_BUDGET="${BUDGET:-2048}"
export BENCH_CONCURRENT="${CONCURRENT:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
OUT="${OUT:-/tmp/memo_paired}"
mkdir -p "${OUT}"

run() {  # run <A|B> <tag>
  IMBA_BENCH_ARM="$1" BENCH_TAG="$2" .venv/bin/python - > "${OUT}/$2.log" 2>&1 <<'PY'
import os, runpy, sys

if os.environ["IMBA_BENCH_ARM"] == "A":
    # Restore the pre-memoization path exactly: a UCI string per legal move
    # (with the castling FFI check) hashed into the vocab dict.
    from imba_chess.eval import cozy_bridge, position_evaluator as pe

    def _uncached(cozy_board, move, move_vocab):
        uci = cozy_bridge.cozy_move_to_uci(cozy_board, move)
        return (move_vocab.token_to_id.get(uci), uci)

    pe._cozy_move_id_and_uci = _uncached

tag = os.environ["BENCH_TAG"]
sys.argv = [
    "generate_search_rollouts.py",
    "--config", "config/imba_chess_exit_seeded_rollout.toml",
    "--checkpoint", "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
    "--output-path", f"/tmp/memo_paired_{tag}.parquet",
    "--max-games", os.environ["BENCH_GAMES"],
    "--search-budget", os.environ["BENCH_BUDGET"],
    "--concurrent-games", os.environ["BENCH_CONCURRENT"],
    "--dtype", "float32", "--sample-seed", "42",
    "--profile", "--profile-every-games", os.environ["BENCH_GAMES"],
]
runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
PY
}

for spec in A:a1 B:b1 B:b2 A:a2; do
  arm="${spec%%:*}"; tag="${spec##*:}"
  run "${arm}" "${tag}"
  echo "arm=${arm} $(tr '\r' '\n' < "${OUT}/${tag}.log" | grep -oE 'total [0-9.]+s' | tail -1)"
done

echo
.venv/bin/python - "${OUT}" <<'PY'
import re, sys
from pathlib import Path

out = Path(sys.argv[1])


def grab(tag):
    txt = (out / f"{tag}.log").read_text(errors="replace").replace("\r", "\n")
    return (
        float(re.findall(r"total ([0-9.]+)s", txt)[-1]),
        float(re.findall(r"search_bookkeeping[^:]*: ([0-9.]+)s", txt)[-1]),
        float(re.findall(r"search_gpu[^:]*: ([0-9.]+)s", txt)[-1]),
        int(re.findall(r"\(([0-9]+) search waves", txt)[-1]),
        int(re.findall(r"([0-9]+) search evals", txt)[-1]),
    )


A = [grab("a1"), grab("a2")]
B = [grab("b1"), grab("b2")]
print("=== paired summary (A=uncached old path, B=memoized) ===")
for nm, arm in (("A uncached", A), ("B memoized", B)):
    t, b, g = (sum(x[i] for x in arm) / 2 for i in (0, 1, 2))
    print(f"  {nm}: total {t:6.2f}s  bookkeeping {b:6.2f}s  search_gpu {g:6.2f}s")

ta, tb = sum(x[0] for x in A) / 2, sum(x[0] for x in B) / 2
ba, bb = sum(x[1] for x in A) / 2, sum(x[1] for x in B) / 2
ga, gb = sum(x[2] for x in A) / 2, sum(x[2] for x in B) / 2
print(f"\n  total          {ta/tb:5.3f}x  ({ta - tb:+.2f}s)")
print(f"  bookkeeping    {ba/bb:5.3f}x  ({ba - bb:+.2f}s)")
print(f"  search_gpu     {ga/gb:5.3f}x  ({ga - gb:+.2f}s)   <- drift control, want ~1.00x")

waves = {x[3] for x in A + B}
evals = {x[4] for x in A + B}
print(f"\n  identical work: waves={waves} evals={evals}", "OK" if len(waves) == len(evals) == 1 else "MISMATCH")
PY
