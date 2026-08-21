"""Per-vocabulary native projector cache on the rollout hot path.

Projection runs once per evaluated search node (428k calls in a 20-game run),
so the ~1,970-entry projector behind it must be built once per vocabulary and
reused, never rebuilt per node -- and it must be keyed weakly, so a cache
outlives neither its vocabulary nor the process's memory budget.

The invariant these tests exist for predates the Rust projector: cozy encodes
castling as king-takes-own-rook, so the literal move string is ambiguous. On
one board `e1h1` is O-O (-> "e1g1"); on another it is a rook sliding e1->h1.
Both are the SAME (from, to, promotion) Move, which is why the answer can
never be memoized on the move alone.
"""

import gc
import weakref

import pytest

pytest.importorskip("torch")
import imba_chess_native as native_cc  # noqa: E402

from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig  # noqa: E402
from imba_chess.eval import cozy_bridge  # noqa: E402

KING_CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
ROOK_SLIDE_FEN = "7k/8/8/8/8/8/8/K3R3 w - - 0 1"


def _vocab_for(*ucis: str) -> MoveVocab:
    token_to_id = {"<pad>": 0, "<start>": 1}
    for i, u in enumerate(ucis):
        token_to_id[u] = 2 + i
    return MoveVocab(token_to_id=token_to_id)


def _native_move(board, raw: str):
    for move in board.generate_moves():
        if str(move) == raw:
            return move
    raise AssertionError(f"{raw} not legal in {board.fen()}")


def test_project_legal_moves_filters_special_tokens_and_caches_per_vocab():
    board = native_cc.Board()
    first = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=True))
    second = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=False))

    first_result = cozy_bridge.project_legal_moves(board, first)
    second_result = cozy_bridge.project_legal_moves(board, second)

    assert first_result[0] == [first.token_to_id["e2e4"]]
    assert second_result[0] == [second.token_to_id["e2e4"]]
    assert first_result[2] == second_result[2] == ["e2e4"]


def test_project_legal_moves_reuses_one_projector_per_vocab():
    """Rebuilding a ~1,970-entry projector per node would undo the speedup."""
    board = native_cc.Board()
    vocab = MoveVocab.build_static()

    cozy_bridge.project_legal_moves(board, vocab)
    cached = cozy_bridge._PROJECTORS[vocab]
    cozy_bridge.project_legal_moves(board, vocab)

    assert cozy_bridge._PROJECTORS[vocab] is cached


def test_projector_cache_does_not_keep_its_vocab_alive():
    """Weak keying, as the memo it replaces had: the cache must die with its
    owner rather than pinning every vocab ever projected."""
    vocab = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=False))
    cozy_bridge.project_legal_moves(native_cc.Board(), vocab)
    assert len(cozy_bridge._PROJECTORS) >= 1
    witness = weakref.ref(vocab)

    del vocab
    gc.collect()

    assert witness() is None, "the projector cache pinned its vocab alive"


def test_projection_is_board_dependent_for_an_ambiguous_move_key():
    """The invariant the old memo existed to protect, restated for the
    projector: `e1h1` is O-O on one board and a rook slide on another, and
    both are the same (from, to, promotion) Move. Querying the castling board
    first must not poison the sliding board."""
    king_board = native_cc.Board.from_fen(KING_CASTLE_FEN)
    rook_board = native_cc.Board.from_fen(ROOK_SLIDE_FEN)
    assert _native_move(king_board, "e1h1") == _native_move(rook_board, "e1h1")

    vocab = _vocab_for("e1g1", "e1h1")

    assert "e1g1" in cozy_bridge.project_legal_moves(king_board, vocab)[2]
    assert "e1h1" in cozy_bridge.project_legal_moves(rook_board, vocab)[2]
    assert "e1g1" in cozy_bridge.project_legal_moves(king_board, vocab)[2]
    assert "e1h1" in cozy_bridge.project_legal_moves(rook_board, vocab)[2]


def test_project_legal_moves_reports_unmapped_moves_by_omission():
    board = native_cc.Board()
    vocab = _vocab_for("a2a3")

    ids, moves, ucis, total = cozy_bridge.project_legal_moves(board, vocab)

    assert ucis == ["a2a3"]
    assert ids == [vocab.token_to_id["a2a3"]]
    assert len(moves) == 1
    assert total == 20
