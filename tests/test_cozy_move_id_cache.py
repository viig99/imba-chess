"""Cached cozy-move -> (vocab id, UCI) lookup on the rollout hot path.

Why this cache exists: profiling rollout generation
(docs/superpowers/notes/2026-08-20-rl-throughput-bottleneck.md) found
`_project_legal_logits_cozy` is the largest addressable Python cost -- it runs
once per evaluated search node (428k calls in a 20-game run) and for every
legal move builds a UCI *string* via `str(move)` plus up to two FFI calls for
the castling check, then hashes that string into the vocab dict. That is 10.5M
string allocations, 20.6M `piece_on` calls and 19.5M dict lookups per run, to
answer a question that is almost entirely a function of the move alone.

cozy `Move` is hashable with value semantics, so the answer can be memoized --
EXCEPT for castling, which is board-dependent and is what these tests pin.
"""

import pytest

torch = pytest.importorskip("torch")
cc = pytest.importorskip("cozy_chess")

from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval import cozy_bridge
from imba_chess.eval.position_evaluator import (
    _cozy_move_id_and_uci,
    _project_legal_logits_cozy,
)

# cozy encodes castling as king-takes-own-rook, so the literal move string is
# ambiguous: on one board `e1h1` is O-O (-> "e1g1"), on another it is a rook
# sliding e1->h1 (-> "e1h1"). Both are the SAME (from, to, promotion) Move.
KING_CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
ROOK_SLIDE_FEN = "7k/8/8/8/8/8/8/K3R3 w - - 0 1"


def _vocab_for(*ucis: str) -> MoveVocab:
    token_to_id = {"<pad>": 0, "<start>": 1}
    for i, u in enumerate(ucis):
        token_to_id[u] = 2 + i
    return MoveVocab(token_to_id=token_to_id)


def _move(board, raw: str):
    for m in board.generate_moves():
        if str(m) == raw:
            return m
    raise AssertionError(f"{raw} not legal in {board.fen()}")


def test_same_move_key_maps_differently_on_castling_vs_sliding_board():
    """The regression this cache must not introduce.

    A cache keyed on the Move alone conflates these two: identical
    (from, to, promotion), different correct answers. Querying the castling
    board FIRST would poison a naive cache for the sliding board.
    """
    king_board = cc.Board.from_fen(KING_CASTLE_FEN)
    rook_board = cc.Board.from_fen(ROOK_SLIDE_FEN)
    km = _move(king_board, "e1h1")
    rm = _move(rook_board, "e1h1")
    assert km == rm and hash(km) == hash(rm), "premise: same Move key"

    vocab = _vocab_for("e1g1", "e1h1")

    # Castling board first -- this is the order that poisons a naive cache.
    assert _cozy_move_id_and_uci(king_board, km, vocab)[1] == "e1g1"
    assert _cozy_move_id_and_uci(rook_board, rm, vocab)[1] == "e1h1"
    # And again, to prove neither answer got cached over the other.
    assert _cozy_move_id_and_uci(king_board, km, vocab)[1] == "e1g1"
    assert _cozy_move_id_and_uci(rook_board, rm, vocab)[1] == "e1h1"


def test_matches_uncached_reference_over_random_playouts():
    """Exhaustive agreement with cozy_move_to_uci + dict lookup."""
    import copy
    import random

    rng = random.Random(42)
    boards = []
    board = cc.Board()
    while len(boards) < 300:
        moves = list(board.generate_moves())
        if not moves or board.status() != cc.GameStatus.Ongoing:
            board = cc.Board()
            continue
        boards.append(board)
        nxt = copy.copy(board)
        nxt.play(rng.choice(moves))
        board = nxt
    boards.append(cc.Board.from_fen(KING_CASTLE_FEN))
    boards.append(cc.Board.from_fen(ROOK_SLIDE_FEN))

    # A vocab covering every UCI the reference path produces.
    ucis = {
        cozy_bridge.cozy_move_to_uci(b, m)
        for b in boards
        for m in b.generate_moves()
    }
    vocab = _vocab_for(*sorted(ucis))

    checked = 0
    for b in boards:
        for m in b.generate_moves():
            ref_uci = cozy_bridge.cozy_move_to_uci(b, m)
            got_id, got_uci = _cozy_move_id_and_uci(b, m, vocab)
            assert got_uci == ref_uci
            assert got_id == vocab.token_to_id.get(ref_uci)
            checked += 1
    assert checked > 5000, f"weak coverage: only {checked} moves"


def test_unmapped_move_reports_none_id_not_a_crash():
    """A move absent from the vocab must yield id None, like dict.get did."""
    board = cc.Board()
    m = _move(board, "e2e4")
    vocab = _vocab_for("a2a3")  # deliberately does not contain e2e4

    move_id, uci = _cozy_move_id_and_uci(board, m, vocab)
    assert uci == "e2e4"
    assert move_id is None


def test_projection_still_agrees_with_reference_after_caching():
    """End-to-end: _project_legal_logits_cozy output is unchanged."""
    board = cc.Board.from_fen(KING_CASTLE_FEN)
    ucis = sorted(
        cozy_bridge.cozy_move_to_uci(board, m) for m in board.generate_moves()
    )
    vocab = _vocab_for(*ucis)
    logits = torch.zeros(len(vocab.token_to_id))

    _, moves, got_ucis, total, mapped = _project_legal_logits_cozy(
        logits=logits, cozy_board=board, move_vocab=vocab
    )

    assert total == mapped == len(ucis)
    assert sorted(got_ucis) == ucis
    assert "e1g1" in got_ucis, "castling must project to standard UCI"
    assert len(moves) == len(got_ucis)
