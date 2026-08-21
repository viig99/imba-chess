"""Decompose the per-node cost of _project_legal_logits_cozy + log_softmax.

decode_project is the largest bucket (30.9%, 54 us/node at budget 2048), but
"largest bucket" is not a design: encode_cozy was a real 1.346x that moved
end-to-end by 1.3%. So split the 54 us into phases first, on real boards and
real-shaped logits, and only then decide what to batch.

Phases timed separately:
  movegen   list(generate_moves())
  mapping   the memoized per-move (vocab id, UCI) loop
  sort      sorted(range(n), key=lambda i: ucis[i]) + the 3 reorder comprehensions
  totensor  torch.tensor(ids) + logits.index_select
  softmax   torch.log_softmax(...).tolist()

Run: .venv/bin/python scripts/bench_project_decompose.py
"""

from __future__ import annotations

import statistics
import time
from weakref import WeakKeyDictionary

import chess
import imba_chess_native as cc
import torch

from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval import cozy_bridge

# ── Retired production fast path, reconstructed ────────────────────────────
# The memoized per-move (vocab id, UCI) lookup this script decomposes was
# deleted in the native projection cutover. It is rebuilt here rather than
# dropped so the 11.12 us/node baseline of record stays reproducible, and so
# scripts/bench_native_move_projector.py has one honest Python arm to time
# against instead of a second private copy. Keep the memo -- and keep castling
# out of it, since the same Move means different things on different boards.

_CASTLE_RAW_TO_UCI = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}

_MOVE_ID_MEMO: "WeakKeyDictionary[MoveVocab, dict]" = WeakKeyDictionary()


def _cozy_move_id_and_uci(cozy_board, move, move_vocab):
    memo = _MOVE_ID_MEMO.get(move_vocab)
    if memo is None:
        memo = {}
        _MOVE_ID_MEMO[move_vocab] = memo
    hit = memo.get(move)
    if hit is not None:
        return hit
    raw = str(move)
    if raw in _CASTLE_RAW_TO_UCI:
        if cozy_board.piece_on(move.from_square) == cc.Piece.King:
            raw = _CASTLE_RAW_TO_UCI[raw]
        return (move_vocab.token_to_id.get(raw), raw)
    result = (move_vocab.token_to_id.get(raw), raw)
    memo[move] = result
    return result


def python_project(board, vocab):
    """Whole retired projection: movegen, mapping, list build, canonical sort."""
    legal_moves = list(board.generate_moves())
    ids, moves, ucis = [], [], []
    for move in legal_moves:
        move_id, uci = _cozy_move_id_and_uci(board, move, vocab)
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


def build_boards() -> list:
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
            out.append(cozy_bridge.board_to_cozy(b.copy()))
    return out


def main() -> None:
    vocab = MoveVocab.build_static()
    boards = build_boards()
    V = len(vocab.token_to_id)
    torch.manual_seed(0)
    # One CPU logits row per node, exactly as consume_decode_result produces
    # after its single per-wave .float().cpu().
    logits = torch.randn(len(boards), V)
    print(f"boards: {len(boards)}   vocab: {V}")

    # Pre-stage per-phase inputs so each phase is timed in isolation.
    movesets = [list(b.generate_moves()) for b in boards]
    mapped = []
    for b, mv in zip(boards, movesets):
        ids, mvs, ucis = [], [], []
        for m in mv:
            mid, uci = _cozy_move_id_and_uci(b, m, vocab)
            if mid is not None:
                ids.append(int(mid))
                mvs.append(m)
                ucis.append(uci)
        mapped.append((ids, mvs, ucis))
    print(
        f"mean legal moves/node: {statistics.mean(len(m) for m in movesets):.1f}   "
        f"mean mapped: {statistics.mean(len(m[0]) for m in mapped):.1f}"
    )

    def phase_movegen():
        for b in boards:
            list(b.generate_moves())

    def phase_mapping():
        for b, mv in zip(boards, movesets):
            ids, mvs, ucis = [], [], []
            for m in mv:
                mid, uci = _cozy_move_id_and_uci(b, m, vocab)
                if mid is not None:
                    ids.append(int(mid))
                    mvs.append(m)
                    ucis.append(uci)

    def phase_sort():
        for ids, mvs, ucis in mapped:
            order = sorted(range(len(mvs)), key=lambda i: ucis[i])
            [mvs[i] for i in order]
            [ucis[i] for i in order]
            [ids[i] for i in order]

    def phase_totensor():
        for row, (ids, _, _) in enumerate(mapped):
            t = torch.tensor(ids, dtype=torch.long)
            logits[row].index_select(0, t)

    def phase_softmax():
        for row, (ids, _, _) in enumerate(mapped):
            t = torch.tensor(ids, dtype=torch.long)
            ll = logits[row].index_select(0, t)
            torch.log_softmax(ll.float(), dim=0).tolist()

    phases = [
        ("movegen", phase_movegen),
        ("mapping", phase_mapping),
        ("sort", phase_sort),
        ("totensor", phase_totensor),
        ("softmax(incl totensor)", phase_softmax),
    ]
    for _, fn in phases:  # warm
        fn()
        fn()

    reps = 30
    print(f"\n{'phase':<24}{'us/node':>10}{'share of measured':>20}")
    print("-" * 54)
    results = {}
    for name, fn in phases:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        results[name] = statistics.median(ts) / len(boards) * 1e6

    # softmax phase includes totensor; report the marginal cost.
    results["softmax(marginal)"] = (
        results["softmax(incl totensor)"] - results["totensor"]
    )
    order = ["movegen", "mapping", "sort", "totensor", "softmax(marginal)"]
    tot = sum(results[k] for k in order)
    for k in order:
        print(f"{k:<24}{results[k]:>10.2f}{results[k] / tot * 100:>19.1f}%")
    print("-" * 54)
    print(f"{'measured total':<24}{tot:>10.2f}")
    print(
        "\nThis total covers legal projection only; arena K/V bookkeeping and "
        "PositionEval construction are outside the benchmark."
    )
    torch_share = results["totensor"] + results["softmax(marginal)"]
    print(
        f"\nper-node torch dispatch (batchable across a 1,321-node wave): "
        f"{torch_share:.2f} us/node = {torch_share / tot * 100:.1f}% of measured"
    )
    print(
        f"lambda sort (removable):                                     "
        f"{results['sort']:.2f} us/node = {results['sort'] / tot * 100:.1f}% of measured"
    )


if __name__ == "__main__":
    main()
