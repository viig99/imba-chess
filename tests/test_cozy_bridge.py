import random

import chess
import cozy_chess as cc
import pytest

from imba_chess.eval.cozy_bridge import (
    board_to_cozy,
    cozy_move_to_uci,
    py_move_to_cozy,
)

# Perft-suite positions: kiwipete, ep-pin, promotion-heavy, castling-rich.
EDGE_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def _random_boards(n_games: int = 50, seed: int = 7) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    for g in range(n_games):
        board = chess.Board()
        for _ in range(rng.randrange(10, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            boards.append(board.copy())
            if board.is_game_over():
                break
    return boards


@pytest.mark.parametrize("fen", EDGE_FENS)
def test_board_to_cozy_matches_fen_roundtrip(fen):
    board = chess.Board(fen)
    assert board_to_cozy(board).fen() == cc.Board.from_fen(fen).fen()


def test_board_to_cozy_matches_fen_roundtrip_on_random_games():
    for board in _random_boards():
        # en_passant="fen": raw/unconditional ep serialization, matching both
        # cozy's own native fen() convention and board_to_cozy's conversion
        # (which preserves ep_square regardless of a legal/pseudo-legal
        # capturer existing -- see board_to_cozy's docstring comment).
        assert (
            board_to_cozy(board).fen()
            == cc.Board.from_fen(board.fen(en_passant="fen")).fen()
        )


def test_move_translation_roundtrips_all_legal_moves():
    for board in [chess.Board(f) for f in EDGE_FENS] + _random_boards(20, seed=11):
        cozy = board_to_cozy(board)
        # py -> cozy: every python-chess legal move maps to a cozy-legal move
        for move in board.legal_moves:
            assert cozy.is_legal(py_move_to_cozy(board, move)), (board.fen(), move)
        # cozy -> uci: the translated set equals python-chess's uci set
        py_ucis = sorted(m.uci() for m in board.legal_moves)
        cc_ucis = sorted(cozy_move_to_uci(cozy, m) for m in cozy.generate_moves())
        assert py_ucis == cc_ucis, board.fen()


def test_castling_translation_both_directions():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    cozy = board_to_cozy(board)
    kingside = chess.Move.from_uci("e1g1")
    assert str(py_move_to_cozy(board, kingside)) == "e1h1"
    queenside = chess.Move.from_uci("e1c1")
    assert str(py_move_to_cozy(board, queenside)) == "e1a1"
    ucis = {cozy_move_to_uci(cozy, m) for m in cozy.generate_moves()}
    assert "e1g1" in ucis and "e1c1" in ucis
    assert "e1h1" not in ucis and "e1a1" not in ucis
