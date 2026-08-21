"""Exact differential for native terminal detection and edge pushing.

`push_and_classify` folds `search._cozy_push` and
`cozy_bridge.terminal_value_native` into one crossing. Both fed search labels
that are gated on bit-identical rollout output, so the Python originals stay
here as independent oracles: a game result that is silently wrong does not
raise, it just changes which line the search believes in.

Coverage aims at the branches that are rare in random play and therefore easy
to get wrong: insufficient material, the ep-normalized repetition hash, the
third-occurrence claim, and python-chess's two one-ply-early claims.
"""

from __future__ import annotations

import copy
import chess
import imba_chess_native as native_cc
import pytest

from imba_chess.eval import cozy_bridge
from tests.test_cozy_bridge import EDGE_FENS, _random_boards

# Positions where the interesting branches actually fire.
TERMINAL_FENS = [
    "7k/8/8/8/8/8/8/K5R1 w - - 0 1",            # lone rook: sufficient material
    "7k/8/8/8/8/8/8/K7 w - - 0 1",              # bare kings: insufficient
    "7k/8/8/8/8/8/8/KB6 w - - 0 1",             # king+bishop: insufficient
    "7k/8/8/8/8/8/8/KN6 w - - 0 1",             # king+knight: insufficient
    "6bk/8/8/8/8/8/8/KB6 w - - 0 1",            # bishops on one square colour
    "7k/8/5K2/8/8/8/8/R7 w - - 0 1",            # mate in one available
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",           # stalemate: Drawn
    "8/8/8/8/8/5k2/6q1/7K w - - 0 1",           # checkmated: Won
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",        # ep flag WITH a legal capturer
    "4k3/8/8/3pP3/8/8/8/6K1 w - d6 0 1",        # ep flag, no legal capturer
    "8/8/8/8/8/6k1/8/6K1 w - - 98 60",          # just under the fifty-move edge
    "8/8/8/8/8/6k1/8/6K1 w - - 99 60",          # the one-ply-early claim
]


_RANDOM = _random_boards(120, seed=4242)
_POSITIONS = (
    [chess.Board(fen) for fen in EDGE_FENS + TERMINAL_FENS]
    + _RANDOM[:: max(1, len(_RANDOM) // 120)][:120]
)


def _cozy_push(cozy_board, cozy_move, hash_history):
    """The retired `search._cozy_push`, kept here as the oracle for the push
    half of `push_and_classify`.

    Copy-and-play one tree edge, threading the repetition hash_history.
    child_history resets to empty on a zeroing move (capture or pawn move --
    child.halfmove_clock == 0) and otherwise carries the parent's history
    forward with the parent's own repetition_hash appended.
    """
    child = copy.copy(cozy_board)
    child.play(cozy_move)
    if child.halfmove_clock == 0:
        child_history: list[int] = []
    else:
        child_history = list(hash_history) + [cozy_bridge.repetition_hash(cozy_board)]
    return child, child_history


def _native(board: chess.Board):
    return native_cc.Board.from_fen(board.fen(en_passant="fen"))


@pytest.mark.parametrize("board", _POSITIONS)
@pytest.mark.parametrize("color_is_stm", [True, False])
def test_push_and_classify_matches_python_push_plus_terminal(
    board: chess.Board, color_is_stm: bool
) -> None:
    native_board = _native(board)
    history: tuple[int, ...] = ()
    for move in native_board.generate_moves():
        want_child, want_history = _cozy_push(native_board, move, history)
        want_value = cozy_bridge.terminal_value_native(
            want_child, color_is_stm=color_is_stm, hash_history=want_history
        )

        child, child_history, value = native_cc.push_and_classify(
            native_board, move, list(history), color_is_stm
        )

        assert child.fen() == want_child.fen()
        assert list(child_history) == list(want_history)
        assert value == want_value


@pytest.mark.parametrize("board", _POSITIONS)
def test_repetition_hash_matches_python(board: chess.Board) -> None:
    native_board = _native(board)
    assert native_cc.repetition_hash_of(native_board) == cozy_bridge.repetition_hash(
        native_board
    )


@pytest.mark.parametrize("board", _POSITIONS)
@pytest.mark.parametrize("color_is_stm", [True, False])
def test_terminal_value_of_matches_python(
    board: chess.Board, color_is_stm: bool
) -> None:
    native_board = _native(board)
    assert native_cc.terminal_value_of(
        native_board, color_is_stm, []
    ) == cozy_bridge.terminal_value_native(
        native_board, color_is_stm=color_is_stm, hash_history=()
    )


def test_threefold_repetition_claim_matches_python_over_a_real_shuffle() -> None:
    """Walks knights out and back so the same position recurs, driving the
    history window, the third-occurrence claim, and the one-ply-early claim
    through both implementations in lockstep."""
    board = native_cc.Board()
    history: tuple[int, ...] = ()
    line = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3
    checked = 0
    for uci in line:
        move = next(m for m in board.generate_moves() if str(m) == uci)
        want_child, want_history = _cozy_push(board, move, history)
        for color_is_stm in (True, False):
            want = cozy_bridge.terminal_value_native(
                want_child, color_is_stm=color_is_stm, hash_history=want_history
            )
            _c, _h, got = native_cc.push_and_classify(
                board, move, list(history), color_is_stm
            )
            assert got == want
            checked += 1
        board, history = want_child, want_history
    assert checked == 2 * len(line)
    # The line really does repeat: otherwise this test proves nothing.
    assert len(history) >= 7


def test_history_resets_on_a_zeroing_move() -> None:
    board = native_cc.Board()
    history = [1, 2, 3]
    pawn = next(m for m in board.generate_moves() if str(m) == "e2e4")
    _child, child_history, _value = native_cc.push_and_classify(
        board, pawn, list(history), True
    )
    assert list(child_history) == [], "a pawn move zeroes the clock"

    knight = next(m for m in board.generate_moves() if str(m) == "g1f3")
    _child2, hist2, _v2 = native_cc.push_and_classify(board, knight, list(history), True)
    assert list(hist2) == history + [cozy_bridge.repetition_hash(board)]


@pytest.mark.parametrize("fen", TERMINAL_FENS)
def test_terminal_fixtures_are_legal_positions(fen: str) -> None:
    """A fixture that is an illegal position fails the differential above for
    the wrong reason -- it looks like an implementation bug. Fail here instead,
    where the message says what is actually wrong."""
    native_cc.Board.from_fen(fen)
