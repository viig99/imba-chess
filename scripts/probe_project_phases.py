"""True wall-clock phase split inside consume_decode_result (no cProfile).

Throwaway diagnostic. cProfile inflates this workload ~2x (349 us/node profiled
vs 174 us/node clean), so profiler self-times cannot be compared against the
clean run's bucket shares -- that mis-sized torch.cat by 2x. This wraps the real
consume path with perf_counter accumulators instead and runs real games, so the
phase shares are directly comparable to the decode_project bucket.

Phases:
  kv_stack   torch.stack of the wave's per-layer (k, v)
  kv_path    per-node path_kv cat (the 2.05 cat/node)
  d2h        the single per-wave .float().cpu() of logits + value_logits
  movegen    list(generate_moves()) per node
  mapping    memoized (vocab id, UCI) loop per node
  sortlists  canonical UCI sort + the 3 reorder comprehensions
  gather     torch.tensor(ids) + index_select per node
  softmax    log_softmax(...).tolist() per node
  assemble   value scalar + PositionEval construction

Run: .venv/bin/python scripts/probe_project_phases.py
"""

from __future__ import annotations

import runpy
import sys
import time
from collections import defaultdict

import torch

from imba_chess.eval import position_evaluator as pe

T: dict[str, float] = defaultdict(float)
N = {"nodes": 0, "waves": 0}


def instrumented_consume(self, request, out):
    t = time.perf_counter
    t0 = t()
    k_all = torch.stack([k for k, _ in out["kv"]], dim=0)
    v_all = torch.stack([v for _, v in out["kv"]], dim=0)
    T["kv_stack"] += t() - t0

    t0 = t()
    for row, node in enumerate(request.nodes):
        own_k, own_v = k_all[:, row], v_all[:, row]
        if node.parent is None:
            node.path_kv = (own_k, own_v)
        else:
            parent_k, parent_v = node.parent.path_kv
            node.path_kv = (
                torch.cat([parent_k, own_k], dim=2),
                torch.cat([parent_v, own_v], dim=2),
            )
    T["kv_path"] += t() - t0

    t0 = t()
    logits = out["logits"].float().cpu()
    value_logits = out["value_logits"].float().cpu()
    T["d2h"] += t() - t0

    results = []
    for row, cozy_board in enumerate(request.boards):
        t0 = t()
        value_stm = pe._value_scalar_from_logits(value_logits[row])
        T["assemble"] += t() - t0

        row_logits = logits[row]
        t0 = t()
        legal_moves_all = list(cozy_board.generate_moves())
        T["movegen"] += t() - t0

        t0 = t()
        ids, mvs, ucis = [], [], []
        for move in legal_moves_all:
            mid, uci = pe._cozy_move_id_and_uci(cozy_board, move, self._move_vocab)
            if mid is not None:
                ids.append(int(mid)); mvs.append(move); ucis.append(uci)
        T["mapping"] += t() - t0

        if not ids:
            results.append(pe.PositionEval(value_stm=value_stm, legal_moves=[],
                                           legal_ucis=[], legal_log_priors=[]))
            continue

        t0 = t()
        order = sorted(range(len(mvs)), key=lambda i: ucis[i])
        mvs = [mvs[i] for i in order]
        ucis = [ucis[i] for i in order]
        ids = [ids[i] for i in order]
        T["sortlists"] += t() - t0

        t0 = t()
        idt = torch.tensor(ids, device=row_logits.device, dtype=torch.long)
        legal_logits = row_logits.index_select(0, idt)
        T["gather"] += t() - t0

        t0 = t()
        log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
        T["softmax"] += t() - t0

        t0 = t()
        results.append(pe.PositionEval(value_stm=value_stm, legal_moves=mvs,
                                       legal_ucis=ucis, legal_log_priors=log_priors))
        T["assemble"] += t() - t0
        N["nodes"] += 1

    N["waves"] += 1
    return results


pe.CachedPositionEvaluator.consume_decode_result = instrumented_consume

sys.argv = [
    "generate_search_rollouts.py",
    "--config", "config/imba_chess_exit_seeded_rollout.toml",
    "--checkpoint", "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
    "--output-path", "/tmp/probe_phases.parquet",
    "--max-games", "6", "--search-budget", "2048", "--concurrent-games", "6",
    "--dtype", "float32", "--sample-seed", "42",
]
try:
    runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
except SystemExit:
    pass

nodes = max(1, N["nodes"])
tot = sum(T.values())
print(f"\n=== consume_decode_result phase split ===")
print(f"nodes {nodes:,}   waves {N['waves']:,}   instrumented total {tot:.2f}s "
      f"({tot / nodes * 1e6:.1f} us/node)\n")
print(f"{'phase':<12}{'seconds':>9}{'us/node':>10}{'share':>8}")
print("-" * 39)
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"{k:<12}{v:>9.2f}{v / nodes * 1e6:>10.2f}{v / tot * 100:>7.1f}%")
print("-" * 39)
print(f"{'TOTAL':<12}{tot:>9.2f}{tot / nodes * 1e6:>10.2f}")
