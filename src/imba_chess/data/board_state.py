from __future__ import annotations

import chess

from .models import BoardState, BoardTokenConfig

PIECE_TO_ID = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}


def _bucket(value: int, max_value: int, bucket_size: int) -> int:
    if value < 0:
        value = 0
    elif value > max_value:
        value = max_value
    return value // bucket_size


def _castle_id(board: chess.Board) -> int:
    rights = board.castling_rights
    mask = 0
    if rights & chess.BB_H1:
        mask |= 1
    if rights & chess.BB_A1:
        mask |= 2
    if rights & chess.BB_H8:
        mask |= 4
    if rights & chess.BB_A8:
        mask |= 8
    return mask


def _ep_file_id(board: chess.Board, mode: str) -> int:
    ep_square = board.ep_square
    if ep_square is None:
        return 0

    if mode == "legal":
        if not board.has_legal_en_passant():
            return 0
    elif mode == "xfen":
        if not board.has_pseudo_legal_en_passant():
            return 0
    elif mode != "fen":
        raise ValueError(f"Unsupported en_passant mode: {mode}")

    return chess.square_file(ep_square) + 1


def _piece_ids(board: chess.Board) -> list[int]:
    piece_ids = [0] * 64
    for square, piece in board.piece_map().items():
        piece_ids[square] = PIECE_TO_ID[piece.symbol()]
    return piece_ids


class BoardStateEncoder:
    def __init__(self, config: BoardTokenConfig | None = None) -> None:
        self.config = config or BoardTokenConfig()

    def encode(self, board: chess.Board) -> BoardState:
        return BoardState(
            piece_ids=_piece_ids(board),
            turn_id=0 if board.turn == chess.WHITE else 1,
            castle_id=_castle_id(board),
            ep_file_id=_ep_file_id(board, self.config.en_passant),
            halfmove_bucket_id=_bucket(
                board.halfmove_clock,
                self.config.halfmove_max,
                self.config.halfmove_bucket_size,
            ),
            fullmove_bucket_id=_bucket(
                board.fullmove_number,
                self.config.fullmove_max,
                self.config.fullmove_bucket_size,
            ),
        )

