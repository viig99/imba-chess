"""Exact differential: native MoveProjector vs the Python projection path.

`_legal_moves_ids_ucis` is the sole source of the vocab-mapping + UCI-sort
discipline the search relies on, so the Rust twin has to reproduce it move for
move, id for id, and -- because callers gather logits positionally -- in the
same order. Compare independently observable state (ids, UCI strings, move
fields) rather than object equality: the two bindings wrap incompatible Rust
Move types.

Temporary shape: once Task 4 retires cozy-chess-py, the old-binding side of
this file collapses onto the owned binding.
"""

from __future__ import annotations

import chess
import cozy_chess as old_cc
import imba_chess_native as native_cc
import pytest

from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.eval.position_evaluator import _legal_moves_ids_ucis
from tests.test_cozy_bridge import EDGE_FENS, _random_boards


def _move_fields(move) -> tuple[int, int, str | None, str]:
    promotion = None if move.promotion is None else str(move.promotion)
    return int(move.from_square), int(move.to_square), promotion, str(move)


def _move_tokens(vocab: MoveVocab) -> dict[str, int]:
    """The vocab's move entries only -- the projector has no notion of the
    pad/start/unk specials, which are unparseable as UCI by construction."""
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


# Built once and shared across cases on purpose: a projector is immutable and
# reused across every evaluated node in production, so exercise it that way.
_STATIC_VOCAB = MoveVocab.build_static()
_STATIC_PROJECTOR = native_cc.MoveProjector(_move_tokens(_STATIC_VOCAB))

# A vocabulary that maps only a slice of the label space, with ids that do not
# match the static vocab's. Catches a projector that reaches for global state
# or silently falls back to the full label set.
_RESTRICTED_UCIS = sorted(_move_tokens(_STATIC_VOCAB))[::7]
_RESTRICTED_VOCAB = MoveVocab.build(
    _RESTRICTED_UCIS, config=MoveVocabConfig(include_unk=False)
)
_RESTRICTED_PROJECTOR = native_cc.MoveProjector(_move_tokens(_RESTRICTED_VOCAB))

_PROJECTOR_RANDOM_BOARDS = _random_boards(200, seed=911)
_PROJECTOR_RANDOM_POSITIONS = _PROJECTOR_RANDOM_BOARDS[
    :: max(1, len(_PROJECTOR_RANDOM_BOARDS) // 200)
][:200]
_POSITIONS = [chess.Board(fen) for fen in EDGE_FENS] + _PROJECTOR_RANDOM_POSITIONS


def _assert_matches_oracle(
    board: chess.Board,
    vocab: MoveVocab,
    projector: native_cc.MoveProjector,
) -> None:
    fen = board.fen(en_passant="fen")
    old_board = old_cc.Board.from_fen(fen)
    new_board = native_cc.Board.from_fen(fen)

    expected_ids, expected_moves, expected_ucis, expected_total = (
        _legal_moves_ids_ucis(old_board, vocab)
    )
    ids, moves, ucis, total = projector.project(new_board)

    assert total == expected_total
    assert ids == expected_ids
    assert ucis == expected_ucis
    assert [_move_fields(move) for move in moves] == [
        _move_fields(move) for move in expected_moves
    ]


@pytest.mark.parametrize("board", _POSITIONS)
def test_native_projector_matches_python_oracle(board: chess.Board) -> None:
    _assert_matches_oracle(board, _STATIC_VOCAB, _STATIC_PROJECTOR)


@pytest.mark.parametrize("board", _POSITIONS)
def test_native_projector_matches_oracle_on_a_restricted_vocabulary(
    board: chess.Board,
) -> None:
    _assert_matches_oracle(board, _RESTRICTED_VOCAB, _RESTRICTED_PROJECTOR)


def test_restricted_vocabulary_actually_drops_moves() -> None:
    """Guards the case above from passing vacuously."""
    dropped = 0
    for board in _POSITIONS:
        fen = board.fen(en_passant="fen")
        full = len(_STATIC_PROJECTOR.project(native_cc.Board.from_fen(fen))[0])
        part = len(_RESTRICTED_PROJECTOR.project(native_cc.Board.from_fen(fen))[0])
        assert part <= full
        dropped += full - part
    assert dropped > 0


def test_two_projectors_stay_independent_when_interleaved() -> None:
    for board in _POSITIONS[:40]:
        _assert_matches_oracle(board, _STATIC_VOCAB, _STATIC_PROJECTOR)
        _assert_matches_oracle(board, _RESTRICTED_VOCAB, _RESTRICTED_PROJECTOR)
        _assert_matches_oracle(board, _STATIC_VOCAB, _STATIC_PROJECTOR)


def test_projection_covers_every_legal_move_of_a_full_vocabulary() -> None:
    """The static vocab is a superset of legal standard-chess moves, so nothing
    may be dropped -- a silent drop would starve the search of a candidate."""
    for board in _POSITIONS:
        new_board = native_cc.Board.from_fen(board.fen(en_passant="fen"))
        ids, _, _, total = _STATIC_PROJECTOR.project(new_board)
        assert len(ids) == total
