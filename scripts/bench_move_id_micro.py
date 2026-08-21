"""GPU-free interleaved microbenchmark for the cozy move -> (vocab id, UCI) lookup.

Companion to scripts/bench_encode_cozy_micro.py; same methodology and the same
reason for it (see that file's docstring: end-to-end paired runs on this laptop
cannot resolve sub-1.2x CPU effects, so measure the function in isolation and
alternate arms rep by rep).

  A (old) = cozy_move_to_uci(board, move) + move_vocab.token_to_id.get(uci)
  B (new) = _cozy_move_id_and_uci(...), memoized per vocab on the cozy Move

Run: .venv/bin/python scripts/bench_move_id_micro.py
"""

from __future__ import annotations

import statistics
import time

import chess

from imba_chess.eval import cozy_bridge
from imba_chess.eval.position_evaluator import _cozy_move_id_and_uci
from imba_chess.data.move_vocab import MoveVocab


def old_move_id_and_uci(cozy_board, move, move_vocab):
    """Pre-optimization pair: build a fresh UCI string, then hash it."""
    uci = cozy_bridge.cozy_move_to_uci(cozy_board, move)
    return (move_vocab.token_to_id.get(uci), uci)


def build_positions() -> list:
    lines = [
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7",
        "d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 Ne4 Bxe7 Qxe7 Rc1 c6 Bd3 Nxc3",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be7 f3 O-O Qd2 Nbd7 g4 b5",
        "Nf3 d5 g3 c6 Bg2 Nf6 O-O Bf5 d3 e6 Nbd2 Be7 Qe1 O-O e4 dxe4 dxe4 Bg6 e5 Nfd7",
    ]
    out = []
    for line in lines:
        b = chess.Board()
        for tok in line.split():
            b.push_san(tok)
            cb = cozy_bridge.board_to_cozy(b.copy())
            out.append((cb, list(cb.generate_moves())))
    return out


def main() -> None:
    vocab = MoveVocab.build_static()
    positions = build_positions()
    n_moves = sum(len(mv) for _, mv in positions)
    print(f"positions: {len(positions)}   legal moves total: {n_moves}")

    for cb, moves in positions:
        for mv in moves:
            new, old = _cozy_move_id_and_uci(cb, mv, vocab), old_move_id_and_uci(cb, mv, vocab)
            assert new == old, f"MISMATCH move={mv} new={new} old={old}"
    print("correctness: new == old on every legal move of every position OK")

    reps, inner = 40, 8

    def timed(fn) -> float:
        t0 = time.perf_counter()
        for _ in range(inner):
            for cb, moves in positions:
                for mv in moves:
                    fn(cb, mv, vocab)
        return time.perf_counter() - t0

    timed(_cozy_move_id_and_uci)  # warm memo + code caches
    timed(old_move_id_and_uci)

    ratios, news, olds = [], [], []
    for r in range(reps):
        if r % 2 == 0:
            t_new, t_old = timed(_cozy_move_id_and_uci), timed(old_move_id_and_uci)
        else:
            t_old, t_new = timed(old_move_id_and_uci), timed(_cozy_move_id_and_uci)
        news.append(t_new)
        olds.append(t_old)
        ratios.append(t_old / t_new)

    calls = inner * n_moves
    print(f"\ncalls per timing: {calls}   reps: {reps}")
    for name, xs in (("old", olds), ("new", news)):
        print(f"  {name}: median {statistics.median(xs) * 1e3:7.2f} ms  "
              f"({statistics.median(xs) / calls * 1e9:6.1f} ns/call)")

    print(f"\n  speedup (median of per-rep ratios): {statistics.median(ratios):.4f}x")
    print(f"  per-rep ratio range: {min(ratios):.4f}x .. {max(ratios):.4f}x")
    separated = max(news) < min(olds)
    print(f"  slowest new ({max(news) * 1e3:.2f} ms) < fastest old "
          f"({min(olds) * 1e3:.2f} ms)? {'YES -- fully separated' if separated else 'no -- overlap'}")

    saved_ns = (statistics.median(olds) - statistics.median(news)) / calls * 1e9
    print(f"\n  saved per call: {saved_ns:.1f} ns")
    print(f"  at 10.5M calls/20-game rollout: {saved_ns * 10.5e6 / 1e9:.2f} s saved")


if __name__ == "__main__":
    main()
