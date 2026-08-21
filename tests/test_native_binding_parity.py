"""Cross-binding differential: owned imba_chess_native vs the pinned
third-party cozy-chess-py wrapper it is replacing.

Both extensions wrap incompatible Rust Board/Move types, so this compares
independently observable state (FEN, hash, status, move fields) rather than
Python object identity/equality across modules. Temporary: deleted in Task 4
once cozy-chess-py is no longer a dependency (see the native design spec's
Stage 1 -> Stage C cutover)."""

from __future__ import annotations

import chess
import cozy_chess as old_cc
import imba_chess_native as new_cc
import pytest

from tests.test_cozy_bridge import EDGE_FENS, _random_boards


def _move_fields(move) -> tuple[int, int, str | None, str]:
    promotion = None if move.promotion is None else str(move.promotion)
    return int(move.from_square), int(move.to_square), promotion, str(move)

_RANDOM_BOARDS = _random_boards(200, seed=731)
_RANDOM_POSITIONS = _RANDOM_BOARDS[
    :: max(1, len(_RANDOM_BOARDS) // 200)
][:200]


@pytest.mark.parametrize(
    "board",
    [chess.Board(fen) for fen in EDGE_FENS] + _RANDOM_POSITIONS,
)
def test_native_binding_matches_pinned_binding(board: chess.Board) -> None:
    old = old_cc.Board.from_fen(board.fen())
    new = new_cc.Board.from_fen(board.fen())

    assert new.fen() == old.fen()
    assert new.hash() == old.hash()
    assert str(new.status()) == str(old.status())
    assert [_move_fields(move) for move in new.generate_moves()] == [
        _move_fields(move) for move in old.generate_moves()
    ]
