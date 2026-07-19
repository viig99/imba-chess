"""python-chess <-> cozy-chess interop for search hot paths.

cozy-chess (Rust, via cozy-chess-py) is an internal acceleration detail
shared by search.py, position_evaluator.py, and board_state.py; python-chess
remains the interface currency and the correctness oracle
(tests/test_cozy_differential.py). Convention differences owned here:

- Castling: python-chess UCI moves the king two files (e1g1); cozy-chess
  represents castling as king-takes-own-rook (e1h1).
- cozy Board.status() covers checkmate/stalemate only; draw claims and
  insufficient material remain the caller's job.
"""

from __future__ import annotations

import copy
from typing import Sequence

import chess
import cozy_chess as cc

_FILE_CHARS = "abcdefgh"

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


def is_capture_cozy(cozy_board: cc.Board, move: cc.Move) -> bool:
    """Does this LEGAL cozy move capture a piece? Mirrors python-chess's
    Board.is_capture() (normal capture + en passant) without needing a
    python-chess board.

    Castling is excluded despite the destination square being occupied:
    cozy represents castling as king-takes-own-rook (see module docstring),
    which must not count as a capture. The en-passant case relies on
    legality: a pawn can only move diagonally to an empty square via en
    passant, so no explicit ep-square check is needed for moves drawn from
    a legal-move generator.
    """
    moving_piece = cozy_board.piece_on(move.from_square)
    if cozy_board.piece_on(move.to_square) is None:
        return moving_piece == cc.Piece.Pawn and (
            int(move.from_square) % 8 != int(move.to_square) % 8
        )
    if (
        moving_piece == cc.Piece.King
        and cozy_board.color_on(move.to_square) == cozy_board.side_to_move()
    ):
        return False  # castling: king "captures" its own rook
    return True


def _no_heavy_pieces(cozy_board: cc.Board) -> bool:
    return not (
        int(cozy_board.pieces(cc.Piece.Pawn))
        | int(cozy_board.pieces(cc.Piece.Rook))
        | int(cozy_board.pieces(cc.Piece.Queen))
    )


def _ep_adjacent_capturers_cozy(cozy_board: cc.Board) -> list[cc.Move]:
    """Pseudo-legal ep capture moves (0-2) for the board's ep flag, if any.

    Shared plumbing for ep-legality probes: used here by
    ep_has_legal_capturer/repetition_hash, and by
    BoardStateEncoder._ep_file_id_cozy (src/imba_chess/data/board_state.py)
    for its "legal"/"xfen" en-passant token modes -- both need to know which
    (<=2) adjacent enemy pawns could pseudo-legally capture the ep square.
    """
    ep_file = cozy_board.en_passant()
    if ep_file is None:
        return []
    file_char = str(ep_file)
    file_idx = _FILE_CHARS.index(file_char)
    stm = cozy_board.side_to_move()
    to_rank = "6" if stm == cc.Color.White else "3"
    from_rank = "5" if stm == cc.Color.White else "4"
    pawns = int(cozy_board.colors(stm) & cozy_board.pieces(cc.Piece.Pawn))
    moves = []
    for adj in (file_idx - 1, file_idx + 1):
        if not 0 <= adj <= 7:
            continue
        from_sq_index = (int(from_rank) - 1) * 8 + adj
        if not (pawns >> from_sq_index) & 1:
            continue
        moves.append(
            cc.Move.from_str(f"{_FILE_CHARS[adj]}{from_rank}{file_char}{to_rank}")
        )
    return moves


def ep_has_legal_capturer(cozy_board: cc.Board) -> bool:
    """True iff the board's ep flag (if any) has an actual LEGAL capturing
    pawn move -- mirrors python-chess's Board.has_legal_en_passant(), the
    gate its transposition key (and therefore threefold-repetition counting)
    uses to decide whether ep is "real". False when there is no ep flag."""
    return any(cozy_board.is_legal(mv) for mv in _ep_adjacent_capturers_cozy(cozy_board))


def repetition_hash(cozy_board: cc.Board) -> int:
    """Canonical repetition/transposition hash.

    cozy's Board.hash() folds the ep flag into the hash unconditionally, even
    when no legal capturer exists (a "phantom" ep flag set by any double
    pawn push, per cozy's native FEN-style semantics). python-chess's
    transposition key -- what threefold-repetition claims actually key off
    of -- excludes ep unless Board.has_legal_en_passant() is true. Without
    this normalization, the same position reached once via a capturer-less
    double push and once via any other path would hash differently and
    threefold repetitions would be undercounted (see Task 2 handoff:
    board_to_cozy preserves raw/unconditional ep, and cozy's own play() sets
    it the same unconditional way on double pushes).
    """
    if cozy_board.en_passant() is not None and not ep_has_legal_capturer(cozy_board):
        return cozy_board.hash_without_ep()
    return cozy_board.hash()


def _has_insufficient_material_for(cozy_board: cc.Board, color: cc.Color) -> bool:
    pawns = int(cozy_board.pieces(cc.Piece.Pawn))
    rooks = int(cozy_board.pieces(cc.Piece.Rook))
    queens = int(cozy_board.pieces(cc.Piece.Queen))
    knights = int(cozy_board.pieces(cc.Piece.Knight))
    bishops = int(cozy_board.pieces(cc.Piece.Bishop))
    kings = int(cozy_board.pieces(cc.Piece.King))
    occ = int(cozy_board.colors(color))
    other = cc.Color.Black if color == cc.Color.White else cc.Color.White
    occ_other = int(cozy_board.colors(other))

    if occ & (pawns | rooks | queens):
        return False

    # Knights are only insufficient material if:
    # (1) We do not have any other pieces, including more than one knight.
    # (2) The opponent does not have pawns, knights, bishops or rooks.
    #     These would allow selfmate.
    if occ & knights:
        return bin(occ).count("1") <= 2 and not (occ_other & ~kings & ~queens)

    # Bishops are only insufficient material if:
    # (1) We do not have any other pieces, including bishops of the
    #     opposite color.
    # (2) The opponent does not have bishops of the opposite color,
    #     pawns or knights. These would allow selfmate.
    if occ & bishops:
        same_color = not (bishops & chess.BB_DARK_SQUARES) or not (
            bishops & chess.BB_LIGHT_SQUARES
        )
        return same_color and not pawns and not knights

    return True


def insufficient_material(cozy_board: cc.Board) -> bool:
    """Exact python-chess Board.is_insufficient_material() semantics:
    neither side has a winning-material possibility, transcribed from
    chess.Board.has_insufficient_material() (chess/__init__.py) onto cozy
    bitboards -- including the "all bishops on the board share a square
    color" case, which is evaluated over BOTH colors' bishops, not just the
    color being tested."""
    return _has_insufficient_material_for(
        cozy_board, cc.Color.White
    ) and _has_insufficient_material_for(cozy_board, cc.Color.Black)


def terminal_value_native(
    cozy_board: cc.Board, *, color_is_stm: bool, hash_history: Sequence[int]
) -> float | None:
    """Drop-in for search.terminal_value_for_color, cozy-native (no
    python-chess board involved). `hash_history` holds repetition_hash()
    values of PRIOR positions since the last irreversible move (most-recent-
    last; excludes the current position at `cozy_board`).

    "Irreversible move" here means "zeroing" (capture or pawn move, i.e.
    halfmove_clock resets to 0) -- narrower than python-chess's
    `is_irreversible`, which also resets on castling-rights loss and
    ep-cession. That's intentional, not an oversight: castling rights and a
    capturable ep square are both part of what `repetition_hash` hashes, so
    two positions on opposite sides of one of those *additional* boundaries
    already hash differently and can never falsely match in the window sum
    below. Reset-on-zeroing-only is therefore sufficient; callers (e.g. a
    future search-tree history) must not narrow the reset condition further,
    but MAY keep this exact contract without also tracking castling/ep
    transitions separately.
    """
    status = cozy_board.status()
    if status == cc.GameStatus.Won:
        # cozy 'Won' == side to move is checkmated.
        return -1.0 if color_is_stm else 1.0
    if status == cc.GameStatus.Drawn:
        # Covers BOTH stalemate AND cozy's own halfmove_clock>=100 auto-draw
        # (cozy's status() implements the raw fifty-move cutoff internally,
        # checkmate/stalemate always taking precedence -- verified against
        # the oracle; halfmove_clock is clamped at 100 by cozy's play()).
        # This subsumes the plain (non-early) fifty-move claim.
        return 0.0
    if _no_heavy_pieces(cozy_board) and insufficient_material(cozy_board):
        return 0.0

    halfmove = cozy_board.halfmove_clock
    if halfmove < 7:
        # A repetition/50-move claim needs >= 7 reversible plies of history
        # for the third occurrence (or the one-ply-early claims below), so
        # the O(history) scan below can be skipped entirely.
        return None

    current = repetition_hash(cozy_board)
    window = hash_history[-halfmove:] if halfmove < len(hash_history) else hash_history
    if sum(1 for h in window if h == current) >= 2:
        return 0.0  # third occurrence reached

    # python-chess also allows claiming one reversible ply early:
    # - repetition: any legal move that REACHES the third occurrence
    #   (can_claim_threefold_repetition).
    # - fifty-move: at halfmove_clock == 99, any legal non-zeroing move
    #   (which necessarily reaches halfmove 100) whose resulting position
    #   still has a legal move of its own (can_claim_fifty_moves ->
    #   is_fifty_moves, which itself requires a legal move to exist -- NOT
    #   reliably readable off child.status(), since status() reports Drawn
    #   at halfmove==100 regardless of whether further moves exist).
    claim_fifty_early = halfmove == 99
    target = [*window, current]
    for move in cozy_board.generate_moves():
        child = copy.copy(cozy_board)
        child.play(move)
        if child.halfmove_clock == 0:
            continue  # irreversible move: cannot repeat, resets the clock
        if claim_fifty_early and child.generate_moves():
            return 0.0
        child_hash = repetition_hash(child)
        if sum(1 for h in target if h == child_hash) >= 2:
            return 0.0
    return None
