"""Permanent differential harness: python-chess is the oracle for every
cozy-backed primitive used by search.py. Covers perft-suite edge positions
(castling, en-passant pins/discoveries, promotions) plus seeded random games.
"""

import random

import chess
import pytest

from imba_chess.eval.cozy_bridge import (
    board_to_cozy,
    gives_check,
    py_move_to_cozy,
)
from tests.test_cozy_bridge import EDGE_FENS, _random_boards

# Hand-built (position, move, expected gives_check) cases the random sweep is
# unlikely to hit. All five verified against python-chess 1.11.2 on 2026-07-18.
CURATED_CASES = [
    # En passant capture giving direct check (pawn lands on d3, attacks Ke2).
    ("k7/8/8/8/3Pp3/8/4K3/8 b - d3 0 1", "e4d3", True),
    # En passant capture giving DISCOVERED check (vacating e4 opens Re8-Ke1).
    ("k3r3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3", True),
    # Castling that gives check (rook lands f1, black king on f-file).
    ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", True),
    # Knight under-promotion with check (Ne8 attacks Kg7).
    ("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1", "e7e8n", True),
    # Quiet discovered check (Nd5 vacates the a1-h8 diagonal onto Kh8).
    ("7k/8/8/8/8/2N5/8/B3K3 w - - 0 1", "c3d5", True),
]


def _all_boards() -> list[chess.Board]:
    return [chess.Board(f) for f in EDGE_FENS] + _random_boards(200, seed=1234)


def test_gives_check_matches_python_chess_everywhere():
    checked = 0
    for board in _all_boards():
        cozy = board_to_cozy(board)
        for move in board.legal_moves:
            assert gives_check(cozy, py_move_to_cozy(board, move)) == board.gives_check(
                move
            ), (board.fen(), move.uci())
            checked += 1
    assert checked > 50_000


@pytest.mark.parametrize("fen,uci,expected", CURATED_CASES)
def test_gives_check_curated_edge_cases(fen, uci, expected):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, "test fixture is broken: move not legal"
    assert board.gives_check(move) == expected, "test fixture is broken: oracle disagrees"
    assert gives_check(board_to_cozy(board), py_move_to_cozy(board, move)) == expected


def test_legal_move_sets_match_python_chess_everywhere():
    from imba_chess.eval.cozy_bridge import cozy_move_to_uci

    for board in _all_boards():
        cozy = board_to_cozy(board)
        assert sorted(m.uci() for m in board.legal_moves) == sorted(
            cozy_move_to_uci(cozy, m) for m in cozy.generate_moves()
        ), board.fen()


def test_terminal_value_fast_matches_terminal_value_for_color():
    from imba_chess.eval.cozy_bridge import terminal_value_fast
    from imba_chess.eval.search import terminal_value_for_color

    terminal_seen = 0
    # Random games REPLAYED so boards carry real move stacks -- repetition and
    # 50-move claims need history, bare FENs can't exercise them.
    rng = random.Random(99)
    for g in range(400):
        board = chess.Board()
        # Shuffle-heavy move choice to actually reach repetitions/50-move claims.
        for _ in range(200):
            moves = list(board.legal_moves)
            if not moves:
                break
            quiet = [m for m in moves if not board.is_capture(m) and m.promotion is None]
            move = rng.choice(quiet if (quiet and rng.random() < 0.8) else moves)
            board.push(move)
            expected = terminal_value_for_color(board, color=chess.WHITE)
            got = terminal_value_fast(board_to_cozy(board), board, chess.WHITE)
            assert got == expected, (board.fen(), expected, got)
            if expected is not None:
                terminal_seen += 1
                break
    assert terminal_seen >= 30  # sweep must actually hit terminal states
