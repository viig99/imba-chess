"""Shared, deterministic chess test inputs; no collected-test imports."""

import json
import random
from pathlib import Path
import chess

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
            boards.append(board.copy(stack=False))
            if board.is_game_over():
                break
    return boards



def native_positions(name):
    data = json.loads((Path(__file__).parent / "fixtures/native_positions.json").read_text())
    return [chess.Board(fen) for fen in data[name]["fens"]]
