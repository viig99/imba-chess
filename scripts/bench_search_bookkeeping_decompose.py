"""Decompose search_bookkeeping, the largest remaining Python bucket.

15.9s / 31.3% of a 20-game rollout is `_halving_stepwise`'s own CPU time
between EvalRequest yields. "Largest bucket" is not a plan -- the projection
port only worked because it was aimed at a phase that had been measured first,
so split this the same way before porting anything.

Phases timed separately, on real tree-node states threaded through the real
`_cozy_push` (so hash_history lengths and zeroing resets are realistic, not
synthetic):

  forcing     _forcing_index_set_tree: is_capture_cozy + gives_check per legal
              move, on opponent-to-move nodes only
  push        _cozy_push: board copy + play + repetition_hash threading
  terminal    terminal_value_native per created child
  order       _prior_order: the per-expansion sort
  heap        heappush/heappop with the itertools.count tiebreaker

Run: .venv/bin/python scripts/bench_search_bookkeeping_decompose.py
"""

from __future__ import annotations

import heapq
import itertools
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_project_decompose import build_boards  # noqa: E402

from imba_chess.data.move_vocab import MoveVocab  # noqa: E402
from imba_chess.eval import cozy_bridge  # noqa: E402
from imba_chess.eval.search import _cozy_push, _prior_order  # noqa: E402

EXPAND_TOP = 3
REFUTATION_TOP_R = 2


def build_nodes(vocab):
    """(board, legal_moves, log_priors, hash_history) per node.

    Histories are threaded through _cozy_push along each opening line rather
    than fabricated, so the window terminal_value_native sums over has the
    length distribution the real search sees.
    """
    nodes = []
    for board in build_boards():
        ids, moves, ucis, forcing, _total = cozy_bridge.project_legal_moves(
            board, vocab
        )
        if not moves:
            continue
        # A plausible prior shape: decreasing, so _prior_order does real work.
        priors = [-0.1 * i for i in range(len(moves))]
        history: tuple[int, ...] = ()
        nodes.append((board, moves, priors, history, forcing))
        # Descend one edge to get non-empty histories in the corpus too.
        child, child_history = _cozy_push(board, moves[0], history)
        c_ids, c_moves, _c_ucis, c_forcing, _c_total = (
            cozy_bridge.project_legal_moves(child, vocab)
        )
        if c_moves:
            nodes.append(
                (
                    child,
                    c_moves,
                    [-0.1 * i for i in range(len(c_moves))],
                    child_history,
                    c_forcing,
                )
            )
    return nodes


def main() -> None:
    vocab = MoveVocab.build_static()
    nodes = build_nodes(vocab)
    picks_per_node = EXPAND_TOP + REFUTATION_TOP_R
    print(
        f"nodes: {len(nodes)}   "
        f"mean legal moves/node: {statistics.mean(len(n[1]) for n in nodes):.1f}   "
        f"mean history len: {statistics.mean(len(n[3]) for n in nodes):.2f}"
    )

    # Pre-stage children so `terminal` is timed without `push` inside it.
    children = []
    for board, moves, _priors, history, _forcing in nodes:
        for move in moves[:picks_per_node]:
            children.append(_cozy_push(board, move, history))
    print(f"children: {len(children)} ({picks_per_node} per node)")

    def phase_forcing():
        # Post-fusion: the flags arrive with the projection, so all the search
        # does is turn them into an index set. The old separate scan
        # (is_capture_cozy + gives_check per move) is gone.
        for _board, _moves, _priors, _history, forcing in nodes:
            {idx for idx, flag in enumerate(forcing) if flag}

    def phase_push():
        for board, moves, _priors, history, _forcing in nodes:
            for move in moves[:picks_per_node]:
                _cozy_push(board, move, history)

    def phase_terminal():
        for child, child_history in children:
            cozy_bridge.terminal_value_native(
                child, color_is_stm=True, hash_history=child_history
            )

    def phase_order():
        for _board, _moves, priors, _history, _forcing in nodes:
            _prior_order(priors)

    def phase_heap():
        counter = itertools.count()
        frontier: list = []
        for _child, _history in children:
            heapq.heappush(frontier, (-1.0, next(counter), None))
        while frontier:
            heapq.heappop(frontier)

    phases = [
        ("forcing", phase_forcing),
        ("push", phase_push),
        ("terminal", phase_terminal),
        ("order", phase_order),
        ("heap", phase_heap),
    ]
    for _name, fn in phases:  # warm
        fn()
        fn()

    reps = 30
    results = {}
    for name, fn in phases:
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        results[name] = statistics.median(samples) / len(nodes) * 1e6

    total = sum(results.values())
    print(f"\n{'phase':<12}{'us/node':>10}{'share':>10}")
    print("-" * 32)
    for name, _fn in phases:
        print(f"{name:<12}{results[name]:>10.2f}{results[name] / total * 100:>9.1f}%")
    print("-" * 32)
    print(f"{'measured':<12}{total:>10.2f}")
    print(
        "\nforcing runs on opponent-to-move nodes only, so weight its share by "
        "that fraction before sizing a fix."
    )
    print(
        "Note: this covers per-node/per-child work only. Arm scoring, backup, "
        "and the generator's own control flow are outside it."
    )


if __name__ == "__main__":
    main()
