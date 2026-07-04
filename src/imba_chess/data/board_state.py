from __future__ import annotations

import chess

from .models import BoardState, BoardTokenConfig


def _bucket(value: int, max_value: int, bucket_size: int) -> int:
    # clamp then bucketize
    if value < 0:
        value = 0
    elif value > max_value:
        value = max_value
    return value // bucket_size


def _castle_id(board: chess.Board) -> int:
    # Standard chess assumption (not Chess960): castling_rights is a bitboard of rook squares.
    rights = board.castling_rights
    return (
        (1 if (rights & chess.BB_H1) else 0)
        | (2 if (rights & chess.BB_A1) else 0)
        | (4 if (rights & chess.BB_H8) else 0)
        | (8 if (rights & chess.BB_A8) else 0)
    )


def _piece_ids(board: chess.Board) -> list[int]:
    # Scan per-piece-type bitboards directly; ~3x faster than piece_map(),
    # which allocates a dict and a Piece object per occupied square.
    ids = [0] * 64
    white = board.occupied_co[chess.WHITE]
    for offset, bb in (
        (0, board.pawns),
        (1, board.knights),
        (2, board.bishops),
        (3, board.rooks),
        (4, board.queens),
        (5, board.kings),
    ):
        for square in chess.scan_forward(bb & white):
            ids[square] = offset + 1
        for square in chess.scan_forward(bb & ~white):
            ids[square] = offset + 7
    return ids


class BoardStateEncoder:
    def __init__(self, config: BoardTokenConfig | None = None) -> None:
        self.config = config or BoardTokenConfig()

        cfg = self.config
        if cfg.halfmove_bucket_size <= 0:
            raise ValueError("halfmove_bucket_size must be > 0")
        if cfg.fullmove_bucket_size <= 0:
            raise ValueError("fullmove_bucket_size must be > 0")

        mode = cfg.en_passant
        if mode == "fen":
            self._ep_ok = None
        elif mode == "legal":
            self._ep_ok = chess.Board.has_legal_en_passant
        elif mode == "xfen":
            self._ep_ok = chess.Board.has_pseudo_legal_en_passant
        else:
            raise ValueError(f"Unsupported en_passant mode: {mode}")

    def _ep_file_id(self, board: chess.Board) -> int:
        ep_square = board.ep_square
        if ep_square is None:
            return 0
        if self._ep_ok is not None and not self._ep_ok(board):
            return 0
        # chess.square_file(ep_square) == ep_square & 7
        return (ep_square & 7) + 1

    def encode(self, board: chess.Board) -> BoardState:
        cfg = self.config
        return BoardState(
            piece_ids=_piece_ids(board),
            turn_id=int(not board.turn),  # white(True)->0, black(False)->1
            castle_id=_castle_id(board),
            ep_file_id=self._ep_file_id(board),
            halfmove_bucket_id=_bucket(
                board.halfmove_clock, cfg.halfmove_max, cfg.halfmove_bucket_size
            ),
            fullmove_bucket_id=_bucket(
                board.fullmove_number, cfg.fullmove_max, cfg.fullmove_bucket_size
            ),
        )
