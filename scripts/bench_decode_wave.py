"""THROWAWAY: is forward_decode_grouped overhead-bound or work-bound?

The budget sweep showed a flat-overhead signature in search_gpu: 44.6
evals/wave costs 20.3 ms while 1,421 evals/wave costs 42.2 ms -- 32x the work
for 2.1x the time. And cProfile attributes 10.1 ms of *Python self time* per
call to hstu_model.forward_decode_grouped (2.85s / 281 calls).

If that is real, search_gpu is mostly a per-wave fixed tax and the fix is
Python/torch restructuring (batch the per-group loop, torch.compile -- which
should work here since this path has no flex_attention -- or CUDA graphs).
If instead time scales with batch size, there is nothing to win and the
forcing-set Rust call should go first.

Method: wrap the real model method, record (batch_size, num_groups,
max_prefix) and elapsed per call over a real rollout, then report ms vs
batch size. Synthetic inputs are avoided on purpose -- prefix_kv_grouped's
padding/group_index invariants are fiddly enough that a hand-built batch
would risk measuring the wrong thing.

Run: .venv/bin/python scripts/bench_decode_wave.py
"""

from __future__ import annotations

import os
import runpy
import statistics
import sys
import time
from collections import defaultdict

import torch

from imba_chess.model.hstu_model import HSTUChessModel

RECORDS: list[tuple[int, int, int, float]] = []


def _install_probe() -> None:
    original = HSTUChessModel.forward_decode_grouped

    def timed(self, **kwargs):
        # rows in this wave
        b = int(kwargs["group_index"].numel())
        g = len(kwargs["prefix_lens_list"])
        maxp = max(kwargs["prefix_lens_list"]) if kwargs["prefix_lens_list"] else 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original(self, **kwargs)
        torch.cuda.synchronize()
        RECORDS.append((b, g, maxp, (time.perf_counter() - t0) * 1e3))
        return out

    HSTUChessModel.forward_decode_grouped = timed


def _report() -> None:
    if not RECORDS:
        print("no decode waves recorded")
        return
    total = sum(r[3] for r in RECORDS)
    print(f"\n=== {len(RECORDS)} decode waves, {total/1000:.1f}s total ===")
    print(f"{'batch rows':>12} {'waves':>7} {'mean_ms':>9} {'median_ms':>10} "
          f"{'us/row':>9} {'mean_maxP':>10}")
    buckets = defaultdict(list)
    for b, g, maxp, ms in RECORDS:
        # log2-ish buckets on wave batch size
        edge = 1
        while edge * 2 <= max(b, 1):
            edge *= 2
        buckets[edge].append((b, maxp, ms))
    for edge in sorted(buckets):
        rows = buckets[edge]
        ms = [r[2] for r in rows]
        bs = [r[0] for r in rows]
        mp = [r[1] for r in rows]
        mean_b = statistics.mean(bs)
        print(f"{edge:>6}-{edge*2-1:<5} {len(rows):>7} {statistics.mean(ms):9.2f} "
              f"{statistics.median(ms):10.2f} {1000*statistics.mean(ms)/max(mean_b,1):9.1f} "
              f"{statistics.mean(mp):10.1f}")
    small = [ms for b, g, mp, ms in RECORDS if b <= 8]
    big = [ms for b, g, mp, ms in RECORDS if b >= 128]
    if small and big:
        sb = statistics.mean([b for b, *_ in RECORDS if b <= 8])
        bb = statistics.mean([b for b, *_ in RECORDS if b >= 128])
        print(
            f"\n  small waves (<=8 rows, mean {sb:.1f}): {statistics.mean(small):.2f} ms\n"
            f"  large waves (>=128 rows, mean {bb:.1f}): {statistics.mean(big):.2f} ms\n"
            f"  work ratio {bb/sb:.1f}x  ->  time ratio {statistics.mean(big)/statistics.mean(small):.2f}x"
        )
        print("\n  VERDICT: overhead-bound (fixed per-wave tax dominates)"
              if statistics.mean(big) / statistics.mean(small) < bb / sb / 4
              else "\n  VERDICT: work-bound (time tracks batch size)")


def main() -> None:
    _install_probe()
    sys.argv = [
        "generate_search_rollouts.py",
        "--config", "config/imba_chess_exit_seeded_rollout.toml",
        "--checkpoint",
        "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
        "--output-path", "/tmp/decode_bench.parquet",
        "--max-games", os.environ.get("BENCH_GAMES", "20"),
        "--search-budget", os.environ.get("BENCH_BUDGET", "2048"),
        "--concurrent-games", os.environ.get("BENCH_CONCURRENT", "8"),
        "--dtype", "float32", "--sample-seed", "42",
    ]
    try:
        runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
    except SystemExit:
        pass
    finally:
        _report()


if __name__ == "__main__":
    main()
