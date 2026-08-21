"""Speed gate: native MoveProjector vs the Python projection path.

The cutover is only worth its risk if the Rust side is decisively faster, so
this measures the exact work the retired Python projection did per evaluated
search node -- move generation, vocab mapping, output-list construction, and the
canonical UCI sort -- against `cozy_bridge.project_legal_moves`, which does all four
in one FFI call. Tensor projection and log_softmax are outside both sides.

Gate (see the native design spec): native <= 5.56 us/node, i.e. at least 2x the
11.12 us/node Python baseline measured by scripts/bench_project_decompose.py.
Exits non-zero if outputs diverge or the gate misses.

Same 80 opening positions as bench_project_decompose.py, imported rather than
restated so the two benchmarks cannot drift apart. Python/native order
alternates per repetition so neither side systematically owns the warm cache.

The Python arm is `bench_project_decompose.python_project`, a reconstruction of
the retired path kept in that script alongside the decomposition it feeds: the
cutover left no Python projection in the tree, and one shared reconstruction
beats two private copies drifting apart. It does the same work the pre-cutover
`_legal_moves_ids_ucis` did -- movegen, a memoized per-move (id, UCI) lookup
that skips the board-dependent castles, and the joint sort -- so the comparison
stays honest against the 11.12 us/node baseline of record.

Run: .venv/bin/python scripts/bench_native_move_projector.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_project_decompose import (  # noqa: E402
    build_boards,
    python_forcing,
    python_project,
)

from imba_chess.data.move_vocab import MoveVocab  # noqa: E402
from imba_chess.eval import cozy_bridge  # noqa: E402

GATE_US_PER_NODE = 5.56
PYTHON_BASELINE_US_PER_NODE = 11.12
PYTHON_FORCING_US_PER_NODE = 19.77
# Fraction of expansions that take the refutation-floor branch, i.e. that used
# to run the forcing scan at all. Measured over a real 2-game rollout:
# 20,248 of 45,055 _push_children calls.
OPPONENT_NODE_FRACTION = 0.449
# Native projection WITHOUT forcing, as measured before the two were fused.
# The "before" state for sizing the fusion is native projection plus a PYTHON
# forcing scan -- the projection was already in Rust by then, so comparing
# against the Python projection would credit this change with a win the
# earlier cutover already banked.
NATIVE_PROJECTION_ONLY_US = 3.24
# Turning the flags into an index set, all that is left of the scan.
FUSED_SET_BUILD_US = 0.41
REPS = 60


def move_tokens(vocab: MoveVocab) -> dict[str, int]:
    specials = {
        vocab.config.pad_token,
        vocab.config.start_token,
        vocab.config.unk_token,
    }
    return {
        token: token_id
        for token, token_id in vocab.token_to_id.items()
        if token not in specials
    }


def move_fields(move) -> tuple[int, int, str | None, str]:
    promotion = None if move.promotion is None else str(move.promotion)
    return int(move.from_square), int(move.to_square), promotion, str(move)


def check_exact(boards, vocab) -> bool:
    """Refuse to report a speedup for a projector that computes the wrong
    thing. Runs before timing and is not itself timed."""
    for board in boards:
        want_ids, want_moves, want_ucis, want_total = python_project(board, vocab)
        ids, moves, ucis, forcing, total = cozy_bridge.project_legal_moves(
            board, vocab
        )
        if (ids, ucis, total) != (want_ids, want_ucis, want_total):
            return False
        want_forcing = python_forcing(board, want_moves)
        if {i for i, flag in enumerate(forcing) if flag} != want_forcing:
            return False
        if [move_fields(m) for m in moves] != [move_fields(m) for m in want_moves]:
            return False
    return True


def main() -> int:
    vocab = MoveVocab.build_static()
    boards = build_boards()
    nodes = len(boards)

    legal_counts = [len(list(board.generate_moves())) for board in boards]
    print(
        f"positions: {nodes}   "
        f"mean legal moves/node: {statistics.mean(legal_counts):.1f}   "
        f"vocab: {len(vocab.token_to_id)}"
    )

    exact = check_exact(boards, vocab)
    print(f"exact outputs: {'PASS' if exact else 'FAIL'}")
    if not exact:
        print("native projection diverges from the Python oracle; not timing.")
        return 1

    def run_python() -> None:
        # Both arms do projection AND forcing: the native side fuses them into
        # one call, so timing it against projection alone would flatter it.
        for board in boards:
            _ids, moves, _ucis, _total = python_project(board, vocab)
            python_forcing(board, moves)

    def run_native() -> None:
        for board in boards:
            cozy_bridge.project_legal_moves(board, vocab)

    # Warm both, including the Python arm's move memo, which was warm in
    # production before the cutover, and the bridge's per-vocab projector
    # cache, which is warm in production now.
    for _ in range(3):
        run_python()
        run_native()

    python_us: list[float] = []
    native_us: list[float] = []
    for rep in range(REPS):
        order = (
            [("python", run_python), ("native", run_native)]
            if rep % 2 == 0
            else [("native", run_native), ("python", run_python)]
        )
        for name, fn in order:
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) / nodes * 1e6
            (python_us if name == "python" else native_us).append(elapsed)

    py_median = statistics.median(python_us)
    nat_median = statistics.median(native_us)
    speedup = py_median / nat_median

    print(f"\nreps: {REPS} per side, alternating order")
    print(f"{'path':<10}{'median us/node':>16}{'min':>10}{'max':>10}")
    print("-" * 46)
    for name, samples in (("python", python_us), ("native", native_us)):
        print(
            f"{name:<10}{statistics.median(samples):>16.2f}"
            f"{min(samples):>10.2f}{max(samples):>10.2f}"
        )
    print("-" * 46)
    print(
        f"baselines of record: projection {PYTHON_BASELINE_US_PER_NODE:.2f} + "
        f"forcing {PYTHON_FORCING_US_PER_NODE:.2f} us/node "
        f"(bench_project_decompose.py, bench_search_bookkeeping_decompose.py)"
    )
    print(f"speedup: {speedup:.2f}x")

    # Per-call cost is not the production win. Forcing used to run only on
    # opponent-to-move nodes, while the fused call computes it on every node,
    # so weight it -- and measure from the state this change actually started
    # from, which already had projection in Rust.
    before = NATIVE_PROJECTION_ONLY_US + (
        PYTHON_FORCING_US_PER_NODE * OPPONENT_NODE_FRACTION
    )
    after = nat_median + (FUSED_SET_BUILD_US * OPPONENT_NODE_FRACTION)
    print(
        f"\nincremental production win, weighted "
        f"({OPPONENT_NODE_FRACTION:.0%} of nodes ran the forcing scan):"
    )
    print(
        f"  before: native projection {NATIVE_PROJECTION_ONLY_US:.2f} + python "
        f"forcing {PYTHON_FORCING_US_PER_NODE:.2f}x{OPPONENT_NODE_FRACTION:.3f} "
        f"= {before:.2f} us/node"
    )
    print(
        f"  after:  fused native {nat_median:.2f} + set build "
        f"{FUSED_SET_BUILD_US:.2f}x{OPPONENT_NODE_FRACTION:.3f} "
        f"= {after:.2f} us/node"
    )
    print(f"  saved:  {before - after:.2f} us/node ({before / after:.2f}x)")
    print(
        "  The 8.5x above is the per-call ratio against a fully-Python arm; "
        "this is what the change is worth in production."
    )

    passed = nat_median <= GATE_US_PER_NODE
    print(
        f"gate: native median {nat_median:.2f} <= {GATE_US_PER_NODE:.2f} us/node "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        print(
            "Gate missed. Stop before production cutover: do not execute "
            "Tasks 4-5, and record the measured failure in the design spec."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
