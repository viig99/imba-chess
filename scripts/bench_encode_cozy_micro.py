"""GPU-free interleaved microbenchmark for BoardStateEncoder.encode_cozy.

End-to-end rollout timing on this laptop cannot see changes below ~1.2x: paired
A/B/B/A runs repeatedly failed their own drift control (search_gpu moved
1.22-1.27x on provably identical GPU work, waves/evals byte-identical) because
the desktop -- a YouTube live stream in Firefox, visible in the profile as
13.45s of _ssl._SSLSocket.read, plus compositing at 81% GPU util with zero
compute processes -- shares this machine's GPU and thermal envelope, producing
a superlinear ramp that A/B/B/A cancels only when drift is linear.

So measure the function itself: no GPU, no model, alternate old/new rep by rep
and report the MEDIAN of per-rep ratios plus the full range. If the slowest
new-arm rep still beats the fastest old-arm rep, the win is real regardless of
drift.

Run: .venv/bin/python scripts/bench_encode_cozy_micro.py
"""

from __future__ import annotations

import statistics
import time

import chess

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.eval import cozy_bridge


def encode_cozy_old(self, board):
    """Verbatim pre-optimization body (git HEAD:src/imba_chess/data/board_state.py)."""
    import cozy_chess as cc

    from imba_chess.data.board_state import BoardState, _bucket

    cfg = self.config
    ids = [0] * 64
    white = int(board.colors(cc.Color.White))
    for offset, piece in (
        (0, cc.Piece.Pawn),
        (1, cc.Piece.Knight),
        (2, cc.Piece.Bishop),
        (3, cc.Piece.Rook),
        (4, cc.Piece.Queen),
        (5, cc.Piece.King),
    ):
        bb = int(board.pieces(piece))
        for square in chess.scan_forward(bb & white):
            ids[square] = offset + 1
        for square in chess.scan_forward(bb & ~white):
            ids[square] = offset + 7
    rights_white = board.castle_rights(cc.Color.White)
    rights_black = board.castle_rights(cc.Color.Black)
    castle_id = (
        (1 if rights_white.short is not None else 0)
        | (2 if rights_white.long is not None else 0)
        | (4 if rights_black.short is not None else 0)
        | (8 if rights_black.long is not None else 0)
    )
    return BoardState(
        piece_ids=ids,
        turn_id=int(board.side_to_move() == cc.Color.Black),
        castle_id=castle_id,
        ep_file_id=self._ep_file_id_cozy(board),
        halfmove_bucket_id=_bucket(
            board.halfmove_clock, cfg.halfmove_max, cfg.halfmove_bucket_size
        ),
        fullmove_bucket_id=_bucket(
            board.fullmove_number, cfg.fullmove_max, cfg.fullmove_bucket_size
        ),
    )


def build_boards() -> list:
    """Realistic search-node positions: walk several openings, keep every ply."""
    lines = [
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7",
        "d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 Ne4 Bxe7 Qxe7 Rc1 c6 Bd3 Nxc3",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be7 f3 O-O Qd2 Nbd7 g4 b5",
        "Nf3 d5 g3 c6 Bg2 Nf6 O-O Bf5 d3 e6 Nbd2 Be7 Qe1 O-O e4 dxe4 dxe4 Bg6 e5 Nfd7",
    ]
    boards = []
    for line in lines:
        b = chess.Board()
        for tok in line.split():
            b.push_san(tok)
            boards.append(cozy_bridge.board_to_cozy(b.copy()))
    return boards


def main() -> None:
    enc = BoardStateEncoder()
    boards = build_boards()
    print(f"positions: {len(boards)}")

    for b in boards:
        new, old = enc.encode_cozy(b), encode_cozy_old(enc, b)
        assert new == old, f"MISMATCH\n new={new}\n old={old}"
    print("correctness: new == old on every position OK")

    reps, inner = 40, 15

    def timed(fn) -> float:
        t0 = time.perf_counter()
        for _ in range(inner):
            for b in boards:
                fn(b)
        return time.perf_counter() - t0

    old_fn = lambda b: encode_cozy_old(enc, b)  # noqa: E731
    timed(enc.encode_cozy)  # warm lazy cozy import, const table, code caches
    timed(old_fn)

    ratios, news, olds = [], [], []
    for r in range(reps):
        if r % 2 == 0:  # alternate order to cancel within-rep bias
            t_new, t_old = timed(enc.encode_cozy), timed(old_fn)
        else:
            t_old, t_new = timed(old_fn), timed(enc.encode_cozy)
        news.append(t_new)
        olds.append(t_old)
        ratios.append(t_old / t_new)

    calls = inner * len(boards)
    print(f"\ncalls per timing: {calls}   reps: {reps}")
    for name, xs in (("old", olds), ("new", news)):
        print(f"  {name}: median {statistics.median(xs) * 1e3:7.2f} ms  "
              f"({statistics.median(xs) / calls * 1e6:5.2f} us/call)")

    print(f"\n  speedup (median of per-rep ratios): {statistics.median(ratios):.4f}x")
    print(f"  per-rep ratio range: {min(ratios):.4f}x .. {max(ratios):.4f}x")
    print(f"  ratio stdev: {statistics.stdev(ratios):.4f}")
    # Strongest available claim: worst new rep still beats best old rep.
    separated = max(news) < min(olds)
    print(f"  slowest new ({max(news) * 1e3:.2f} ms) < fastest old "
          f"({min(olds) * 1e3:.2f} ms)? {'YES -- fully separated' if separated else 'no -- distributions overlap'}")

    saved_us = (statistics.median(olds) - statistics.median(news)) / calls * 1e6
    print(f"\n  saved per call: {saved_us:.3f} us")
    print(f"  at 428,016 calls/20-game rollout: {saved_us * 428016 / 1e6:.2f} s saved")


if __name__ == "__main__":
    main()
