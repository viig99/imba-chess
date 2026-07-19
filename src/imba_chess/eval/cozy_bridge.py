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
            (cc.Color.Black, bitboard & ~occ_white),
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
    if board.ep_square is not None:
        # Raw/unconditional, matching cozy's own native double-push semantics
        # (see cozy Board.fen()/en_passant()) -- NOT gated on a legal capturer
        # existing. Mode-specific ("legal"/"xfen") filtering is the caller's
        # job (see BoardStateEncoder._ep_file_id_cozy); gating here would
        # silently discard information those modes need to reconstruct.
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


def gives_check(cozy_board: cc.Board, cozy_move: cc.Move) -> bool:
    """Does this legal move give check? Simulate-in-Rust (~240ns vs ~3us
    for python-chess's Python-level push/check/pop)."""
    after = copy.copy(cozy_board)
    after.play(cozy_move)
    return bool(after.checkers())


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


def _no_heavy_pieces(cozy_board: cc.Board) -> bool:
    return not (
        int(cozy_board.pieces(cc.Piece.Pawn))
        | int(cozy_board.pieces(cc.Piece.Rook))
        | int(cozy_board.pieces(cc.Piece.Queen))
    )


def terminal_value_fast(
    cozy_board: cc.Board, board: chess.Board, color: chess.Color
) -> float | None:
    """Drop-in for search.terminal_value_for_color, cozy fast path.

    cozy status() decides checkmate/stalemate (~50ns vs ~4.3us). python-chess
    stays the oracle on the rare paths: insufficient material (only reachable
    when no pawn/rook/queen exists -- cheap cozy pre-filter) and draw claims
    (only reachable at halfmove_clock >= 7, the pre-existing guard).
    """
    status = cozy_board.status()
    if status == cc.GameStatus.Won:
        # cozy 'Won' == side to move is checkmated; winner is the other side.
        winner = not board.turn
        return 1.0 if winner == color else -1.0
    if status == cc.GameStatus.Drawn:
        return 0.0  # stalemate
    if _no_heavy_pieces(cozy_board) and board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 7:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            return 1.0 if outcome.winner == color else -1.0
    return None
