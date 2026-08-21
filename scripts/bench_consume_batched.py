"""Old per-node vs new batched projection, at real wave shapes, no GPU.

The end-to-end run showed no change (74.5s -> 74.3s, decode_project 23.1 ->
23.5s on identical work), which is within this machine's noise. Deterministic
microbenchmark to decide keep-or-revert.

Hypothesis under test: batching removes ~7 us/node of torch dispatch but adds
an O(B*M) Python list -> torch.tensor conversion (~80 ns/element), so the two
cancel.

Real wave shape at budget 2048 / concurrent 8: B ~= 1321 nodes, vocab 1970,
~31 mapped legal moves per node.

Run: .venv/bin/python scripts/bench_consume_batched.py
"""

from __future__ import annotations

import random
import statistics
import time

import torch

from imba_chess.eval.position_evaluator import (
    _batched_legal_log_priors,
    _batched_value_scalars,
    _value_scalar_from_logits,
)

B, V = 1321, 1970


def make_wave(seed: int = 0):
    rng = random.Random(seed)
    id_lists = []
    for _ in range(B):
        n = max(1, int(rng.gauss(31, 7)))
        id_lists.append(rng.sample(range(V), min(n, V)))
    torch.manual_seed(seed)
    return id_lists, torch.randn(B, V), torch.randn(B, 3)


def old_path(logits, value_logits, id_lists):
    out = []
    for row, ids in enumerate(id_lists):
        v = _value_scalar_from_logits(value_logits[row])
        t = torch.tensor(ids, device=logits.device, dtype=torch.long)
        legal = logits[row].index_select(0, t)
        lp = torch.log_softmax(legal.float(), dim=0).tolist()
        out.append((v, lp))
    return out


def new_path(logits, value_logits, id_lists):
    vs = _batched_value_scalars(value_logits)
    lps = _batched_legal_log_priors(logits, id_lists)
    return list(zip(vs, lps))


def main() -> None:
    id_lists, logits, value_logits = make_wave()
    widths = [len(x) for x in id_lists]
    print(f"B={B}  vocab={V}  mapped moves: mean {statistics.mean(widths):.1f} "
          f"max {max(widths)} (padded width)")
    print(f"padding waste: {(max(widths) * B) / sum(widths):.2f}x elements gathered\n")

    a = old_path(logits, value_logits, id_lists)
    b = new_path(logits, value_logits, id_lists)
    assert len(a) == len(b)
    worst_v = max(abs(x[0] - y[0]) for x, y in zip(a, b))
    worst_p = max(max((abs(p - q) for p, q in zip(x[1], y[1])), default=0.0)
                  for x, y in zip(a, b))
    print(f"numeric agreement: max |dvalue| {worst_v:.3e}   max |dlog_prior| {worst_p:.3e}")

    for fn in (old_path, new_path):  # warm
        fn(logits, value_logits, id_lists)

    reps = 12
    res = {}
    for name, fn in (("old (per-node)", old_path), ("new (batched)", new_path)):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(logits, value_logits, id_lists)
            ts.append(time.perf_counter() - t0)
        res[name] = statistics.median(ts)

    print(f"\n{'variant':<18}{'ms/wave':>10}{'us/node':>10}")
    print("-" * 38)
    for k, v in res.items():
        print(f"{k:<18}{v * 1e3:>10.2f}{v / B * 1e6:>10.2f}")
    o, n = res["old (per-node)"], res["new (batched)"]
    print("-" * 38)
    print(f"speedup: {o / n:.3f}x   ({(o - n) / B * 1e6:+.2f} us/node)")
    print(f"\nat 428,016 nodes/20-game run: {(o - n) * 428016 / B:+.2f} s")


if __name__ == "__main__":
    main()
