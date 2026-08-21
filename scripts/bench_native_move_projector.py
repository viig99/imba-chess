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

The Python arm is a local reimplementation, not the production path: after the
cutover there is no Python projection left in the tree. It is the same work the
pre-cutover `_legal_moves_ids_ucis` did -- movegen, a memoized per-move
(id, UCI) lookup that must skip the board-dependent castles, and the joint
sort -- so the comparison stays honest against the 11.12 us/node baseline of
record.

Run: .venv/bin/python scripts/bench_native_move_projector.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import imba_chess_native as native_cc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_project_decompose import build_boards  # noqa: E402

from imba_chess.data.move_vocab import MoveVocab  # noqa: E402
from imba_chess.eval import cozy_bridge  # noqa: E402

# ── Python arm ─────────────────────────────────────────────────────────────

_CASTLE_RAW_TO_UCI = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}


def _move_id_and_uci(board, move, vocab, memo):
    """Reconstruction of the retired production fast path, memo and all.

    Timing the native projector against a naive unmemoized loop would flatter
    it, so keep the memo -- and keep castling out of it, since the same Move
    means different things on different boards.
    """
    hit = memo.get(move)
    if hit is not None:
        return hit
    raw = str(move)
    if raw in _CASTLE_RAW_TO_UCI:
        if board.piece_on(move.from_square) == native_cc.Piece.King:
            raw = _CASTLE_RAW_TO_UCI[raw]
        return (vocab.token_to_id.get(raw), raw)
    result = (vocab.token_to_id.get(raw), raw)
    memo[move] = result
    return result


def python_project(board, vocab, memo):
    legal_moves = list(board.generate_moves())
    ids, moves, ucis = [], [], []
    for move in legal_moves:
        move_id, uci = _move_id_and_uci(board, move, vocab, memo)
        if move_id is not None:
            ids.append(int(move_id))
            moves.append(move)
            ucis.append(uci)
    if not ids:
        return [], [], [], len(legal_moves)
    order = sorted(range(len(moves)), key=lambda i: ucis[i])
    return (
        [ids[i] for i in order],
        [moves[i] for i in order],
        [ucis[i] for i in order],
        len(legal_moves),
    )


GATE_US_PER_NODE = 5.56
PYTHON_BASELINE_US_PER_NODE = 11.12
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


def check_exact(boards, vocab, memo) -> bool:
    """Refuse to report a speedup for a projector that computes the wrong
    thing. Runs before timing and is not itself timed."""
    for board in boards:
        want_ids, want_moves, want_ucis, want_total = python_project(
            board, vocab, memo
        )
        ids, moves, ucis, total = cozy_bridge.project_legal_moves(board, vocab)
        if (ids, ucis, total) != (want_ids, want_ucis, want_total):
            return False
        if [move_fields(m) for m in moves] != [move_fields(m) for m in want_moves]:
            return False
    return True


def main() -> int:
    vocab = MoveVocab.build_static()
    boards = build_boards()
    nodes = len(boards)
    memo: dict = {}

    legal_counts = [len(list(board.generate_moves())) for board in boards]
    print(
        f"positions: {nodes}   "
        f"mean legal moves/node: {statistics.mean(legal_counts):.1f}   "
        f"vocab: {len(vocab.token_to_id)}"
    )

    exact = check_exact(boards, vocab, memo)
    print(f"exact outputs: {'PASS' if exact else 'FAIL'}")
    if not exact:
        print("native projection diverges from the Python oracle; not timing.")
        return 1

    def run_python() -> None:
        for board in boards:
            python_project(board, vocab, memo)

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
        f"python baseline of record: {PYTHON_BASELINE_US_PER_NODE:.2f} us/node "
        f"(bench_project_decompose.py: movegen + mapping + sort)"
    )
    print(f"speedup: {speedup:.2f}x")

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
