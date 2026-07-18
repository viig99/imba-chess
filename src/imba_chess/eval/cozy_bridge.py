"""python-chess <-> cozy-chess interop for search hot paths.

cozy-chess (Rust, via cozy-chess-py) is an internal acceleration detail of
search.py; python-chess remains the interface currency and the correctness
oracle (tests/test_cozy_differential.py). Convention differences owned here:

- Castling: python-chess UCI moves the king two files (e1g1); cozy-chess
  represents castling as king-takes-own-rook (e1h1).
- cozy Board.status() covers checkmate/stalemate only; draw claims and
  insufficient material remain the caller's job.
"""

from __future__ import annotations

import copy

import chess
import cozy_chess as cc

_PIECES = (
    (cc.Piece.Pawn, "pawns"),
    (cc.Piece.Knight, "knights"),
    (cc.Piece.Bishop, "bishops"),
    (cc.Piece.Rook, "rooks"),
    (cc.Piece.Queen, "queens"),
    (cc.Piece.King, "kings"),
)


def board_to_cozy(board: chess.Board) -> cc.Board:
    """Convert via raw bitboard ints (~6us; python-chess .fen() alone is ~16us)."""
    builder = cc.BoardBuilder.empty()
    occ_white = board.occupied_co[chess.WHITE]
    for piece, attr in _PIECES:
        bitboard = getattr(board, attr)
        for color, mask in (
            (cc.Color.White, bitboard & occ_white),
            (cc.Color.Black, bitboard & ~occ_white & bitboard),
        ):
            if mask:
                for square in cc.BitBoard(mask):
                    builder.set_piece(square, piece, color)
    if board.turn == chess.BLACK:
        builder.set_side_to_move(cc.Color.Black)
    rights = board.castling_rights
    for color, kingside_bb, queenside_bb in (
        (cc.Color.White, chess.BB_H1, chess.BB_A1),
        (cc.Color.Black, chess.BB_H8, chess.BB_A8),
    ):
        short = cc.File.H if rights & kingside_bb else None
        long = cc.File.A if rights & queenside_bb else None
        if short is not None or long is not None:
            builder.set_castle_rights(color, short=short, long=long)
    if board.ep_square is not None and board.has_legal_en_passant():
        for square in cc.BitBoard(1 << board.ep_square):
            builder.set_en_passant(square)
    builder.set_halfmove_clock(board.halfmove_clock)
    builder.set_fullmove_number(board.fullmove_number)
    return builder.build()


def py_move_to_cozy(board: chess.Board, move: chess.Move) -> cc.Move:
    uci = move.uci()
    if board.is_castling(move):
        rook_file = (
            "h"
            if chess.square_file(move.to_square) > chess.square_file(move.from_square)
            else "a"
        )
        uci = uci[0] + uci[1] + rook_file + uci[1]
    return cc.Move.from_str(uci)


def cozy_move_to_uci(cozy_board: cc.Board, move: cc.Move) -> str:
    uci = str(move)
    if (
        cozy_board.piece_on(move.from_square) == cc.Piece.King
        and cozy_board.color_on(move.to_square) == cozy_board.side_to_move()
    ):
        # King "capturing" its own rook = castling; emit standard two-file UCI.
        new_file = "g" if uci[2] > uci[0] else "c"
        return uci[0] + uci[1] + new_file + uci[3]
    return uci
