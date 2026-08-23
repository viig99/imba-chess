"""THROWAWAY: split _push_children's 19.35 us/node, and test the depth hypothesis.

probe_bookkeeping_phases.py showed `_push_children` is 81.1% of the
search_bookkeeping bucket (19.35 of 23.85 us/node) -- roughly 25% of TOTAL wall
time in one function -- while the synthetic harness
(bench_search_bookkeeping_decompose.py) measured only 3.17 us/node for the same
phases, 6.1x under.

The harness reported `mean history len: 0.10`. Production runs to max_depth=8.
`search.py:661` marshals node.hash_history (a Python list that GROWS WITH DEPTH)
into Rust and marshals a fresh child_history list back, once per child per node.
That is the one cost that scales on the axis the harness under-sampled.

This probe splits _push_children into its parts AND buckets push_and_classify by
input history length, so the depth hypothesis is tested directly rather than
argued. If cost is flat in history length, the hypothesis is wrong and the
target is elsewhere.

Wrapper overhead (~0.1-0.2 us per wrapped call) inflates the smallest phases;
read the SHARES and the history-length trend, not the absolute microseconds.

Run: .venv/bin/python scripts/probe_push_children.py
"""

from __future__ import annotations

import os
import runpy
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imba_chess_native as cc  # noqa: E402

from imba_chess.eval import search  # noqa: E402
from imba_chess.eval.position_evaluator import CachedPositionEvaluator  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)
BY_HIST = defaultdict(list)  # history length -> [elapsed_us]


def _install() -> None:
    orig_push = search._push_children
    orig_classify = cc.push_and_classify
    orig_extend = CachedPositionEvaluator.extend
    orig_order = search._prior_order

    def timed_push_children(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_push(*a, **k)
        finally:
            T["push_children"] += time.perf_counter() - t0
            N["push_children"] += 1

    def timed_classify(board, move, hash_history, color_is_stm):
        hlen = len(hash_history) if hash_history is not None else 0
        t0 = time.perf_counter()
        try:
            return orig_classify(board, move, hash_history, color_is_stm)
        finally:
            dt = time.perf_counter() - t0
            T["push_and_classify"] += dt
            N["push_and_classify"] += 1
            if len(BY_HIST[hlen]) < 20000:
                BY_HIST[hlen].append(dt * 1e6)

    def timed_extend(self, handle, move_uci, move_vocab_id=None):
        t0 = time.perf_counter()
        try:
            return orig_extend(self, handle, move_uci, move_vocab_id)
        finally:
            T["extend"] += time.perf_counter() - t0
            N["extend"] += 1

    def timed_order(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_order(*a, **k)
        finally:
            T["prior_order"] += time.perf_counter() - t0
            N["prior_order"] += 1

    search._push_children = timed_push_children
    cc.push_and_classify = timed_classify
    search.cc.push_and_classify = timed_classify
    CachedPositionEvaluator.extend = timed_extend
    search._prior_order = timed_order
    os._exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))


def _report() -> None:
    nodes = N["push_children"]
    if not nodes:
        print("no nodes recorded")
        return
    total = T["push_children"]
    print("\n" + "=" * 72)
    print("_push_children sub-decomposition")
    print("=" * 72)
    print(f"nodes: {nodes:,}   children (push_and_classify calls): {N['push_and_classify']:,}"
          f"   ({N['push_and_classify'] / nodes:.2f} per node)")
    print(f"\n  {'phase':22s} {'seconds':>9s} {'us/node':>9s} {'share':>8s}   calls")
    print(f"  {'_push_children TOTAL':22s} {total:9.3f} {total / nodes * 1e6:9.2f} {'100.0%':>8s}   {nodes:,}")
    for key in ("push_and_classify", "extend", "prior_order"):
        if N[key]:
            print(f"  {key:22s} {T[key]:9.3f} {T[key] / nodes * 1e6:9.2f} "
                  f"{T[key] / total:8.1%}   {N[key]:,}")
    acc = T["push_and_classify"] + T["extend"] + T["prior_order"]
    print(f"  {'unattributed':22s} {total - acc:9.3f} {(total - acc) / nodes * 1e6:9.2f} "
          f"{(total - acc) / total:8.1%}   (TreeNode alloc, heappush, forcing set, loop)")

    print("\n" + "-" * 72)
    print("push_and_classify cost vs INPUT HISTORY LENGTH (the depth hypothesis)")
    print("-" * 72)
    print(f"  {'hist_len':>8s} {'calls':>10s} {'median us':>11s} {'mean us':>9s}")
    for hlen in sorted(BY_HIST):
        v = BY_HIST[hlen]
        if len(v) < 50:
            continue
        print(f"  {hlen:8d} {len(v):10,} {statistics.median(v):11.3f} {statistics.mean(v):9.3f}")
    keys = [h for h in sorted(BY_HIST) if len(BY_HIST[h]) >= 50]
    if len(keys) >= 2:
        lo, hi = keys[0], keys[-1]
        r = statistics.median(BY_HIST[hi]) / max(1e-9, statistics.median(BY_HIST[lo]))
        print(f"\n  hist {lo} -> {hi}: {r:.2f}x")
        print("  VERDICT: cost SCALES with history length -- marshalling is the target"
              if r > 1.5 else
              "  VERDICT: cost is FLAT in history length -- hypothesis REFUTED, look elsewhere")


def main() -> None:
    _install()
    sys.argv = [
        "generate_search_rollouts.py",
        "--config", "config/imba_chess_exit_seeded_rollout.toml",
        "--checkpoint", "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
        "--output-path", "/tmp/push_children_probe.parquet",
        "--local-corpus", "artifacts/corpus/seed42_train.parquet",
        "--max-games", os.environ.get("PROBE_GAMES", "6"),
        "--search-budget", "2048", "--concurrent-games", "8",
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
