"""Wall-clock phase split inside the current arena-backed result consumer.

Throwaway diagnostic. cProfile inflates this workload, so these accumulators
run through real rollout waves. The project's timing buckets remain
asynchronous; use `profile_torch_waves.py` for CPU/CUDA attribution.

Phases:
  kv_stack    stack per-layer K/V returned for the wave
  kv_arena    append one K/V row per node and record ancestor chains
  d2h         wave-level logits/value-logits transfer and synchronization
  movegen     legal move generation, vocab mapping, and canonical sort
              (one native cozy_bridge.project_legal_moves call per node)
  batched     value/prior tensor projection for the complete wave
  assemble    PositionEval construction

Run: `.venv/bin/python scripts/probe_project_phases.py`
"""

from __future__ import annotations

import runpy
import sys
import time
from collections import defaultdict

import torch

from imba_chess.eval import cozy_bridge
from imba_chess.eval import position_evaluator as pe

T: dict[str, float] = defaultdict(float)
N = {"nodes": 0, "waves": 0}


def instrumented_consume(self, request, out):
    t = time.perf_counter
    t0 = t()
    k_stack = torch.stack([k for k, _ in out["kv"]], dim=0)
    v_stack = torch.stack([v for _, v in out["kv"]], dim=0)
    T["kv_stack"] += t() - t0

    t0 = t()
    k_rows = k_stack.squeeze(3).permute(0, 2, 1, 3)
    v_rows = v_stack.squeeze(3).permute(0, 2, 1, 3)
    self._arena = pe._get_or_create_arena(self._arena, k_rows, v_rows)
    assigned_rows = self._arena.append(k_rows, v_rows)
    for node, own_row in zip(request.nodes, assigned_rows):
        parent_chain = [] if node.parent is None else node.parent.arena_chain
        if parent_chain is None:
            raise RuntimeError("Cannot store a child before evaluating its parent")
        node.arena_chain = parent_chain + [own_row]
    T["kv_arena"] += t() - t0

    # Logits-independent Python work runs before the first synchronization,
    # hiding queued arena writes under work the rollout owes either way.
    t0 = t()
    per_node = []
    for cozy_board in request.boards:
        ids, mvs, ucis, _forcing, _total = cozy_bridge.project_legal_moves(
            cozy_board, self._move_vocab
        )
        per_node.append((ids, mvs, ucis))
    T["movegen+map+sort"] += t() - t0

    t0 = t()
    logits = out["logits"].float().cpu()
    value_logits = out["value_logits"].float().cpu()
    T["d2h(sync wait)"] += t() - t0

    t0 = t()
    values = pe._batched_value_scalars(value_logits)
    id_lists = [ids for ids, _, _ in per_node]
    prior_rows = [[] for _ in id_lists]
    if all(id_lists):
        prior_rows = pe._batched_legal_log_priors(logits, id_lists)
    else:
        keep = [r for r, ids in enumerate(id_lists) if ids]
        if keep:
            sub = pe._batched_legal_log_priors(
                logits.index_select(0, torch.tensor(keep, dtype=torch.long)),
                [id_lists[r] for r in keep],
            )
            for r, pr in zip(keep, sub):
                prior_rows[r] = pr
    T["batched_torch"] += t() - t0

    t0 = t()
    results = [
        pe.PositionEval(
            value_stm=values[r],
            legal_moves=mvs,
            legal_ucis=ucis,
            legal_log_priors=prior_rows[r],
        )
        for r, (_ids, mvs, ucis) in enumerate(per_node)
    ]
    T["assemble"] += t() - t0
    N["nodes"] += len(per_node)
    N["waves"] += 1
    return results


pe.CachedPositionEvaluator.consume_decode_result = instrumented_consume

sys.argv = [
    "generate_search_rollouts.py",
    "--config",
    "config/imba_chess_exit_seeded_rollout.toml",
    "--checkpoint",
    "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
    "--output-path",
    "/tmp/probe_phases.parquet",
    "--max-games",
    "6",
    "--search-budget",
    "2048",
    "--concurrent-games",
    "6",
    "--dtype",
    "float32",
    "--sample-seed",
    "42",
]
try:
    runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
except SystemExit:
    pass

nodes = max(1, N["nodes"])
tot = sum(T.values())
print("\n=== consume_decode_result phase split ===")
print(
    f"nodes {nodes:,}   waves {N['waves']:,}   instrumented total {tot:.2f}s "
    f"({tot / nodes * 1e6:.1f} us/node)\n"
)
print(f"{'phase':<12}{'seconds':>9}{'us/node':>10}{'share':>8}")
print("-" * 39)
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"{k:<12}{v:>9.2f}{v / nodes * 1e6:>10.2f}{v / tot * 100:>7.1f}%")
print("-" * 39)
print(f"{'TOTAL':<12}{tot:>9.2f}{tot / nodes * 1e6:>10.2f}")
