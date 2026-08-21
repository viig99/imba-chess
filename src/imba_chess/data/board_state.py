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


_CC_FILES = None  # built lazily: {cc.File.X: 0-based index}


def _cc_file_index(file_obj) -> int:
    global _CC_FILES
    if _CC_FILES is None:
        import imba_chess_native as cc

        _CC_FILES = {
            f: i
            for i, f in enumerate(
                (cc.File.A, cc.File.B, cc.File.C, cc.File.D, cc.File.E, cc.File.F, cc.File.G, cc.File.H)
            )
        }
    return _CC_FILES[file_obj]


_COZY_ENCODE_CONSTS: tuple | None = None


def _cozy_encode_consts() -> tuple:
    """(White, Black, ((white_id, black_id, Piece), ...)) resolved once.

    `encode_cozy` runs once per evaluated search node -- 428k times in a
    20-game rollout -- so its original per-call `import imba_chess_native as cc` plus
    ~14 enum attribute lookups were pure repeated overhead.

    cozy stays a LAZY import on purpose: this module is on the training data
    path, which must not require cozy-chess to be installed.
    """
    global _COZY_ENCODE_CONSTS
    if _COZY_ENCODE_CONSTS is None:
        import imba_chess_native as cc

        _COZY_ENCODE_CONSTS = (
            cc.Color.White,
            cc.Color.Black,
            (
                (1, 7, cc.Piece.Pawn),
                (2, 8, cc.Piece.Knight),
                (3, 9, cc.Piece.Bishop),
                (4, 10, cc.Piece.Rook),
                (5, 11, cc.Piece.Queen),
                (6, 12, cc.Piece.King),
            ),
        )
    return _COZY_ENCODE_CONSTS


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

    def _ep_file_id_cozy(self, board) -> int:
        ep_file = board.en_passant()
        if ep_file is None:
            return 0
        file_idx = _cc_file_index(ep_file)
        if self._ep_ok is None:  # "fen" mode: report as-is
            return file_idx + 1
        # cozy reports the file after ANY double push (FEN-style). "legal" and
        # "xfen" modes require an actual capturer; probe the <=2 candidate
        # en-passant captures (shared with cozy_bridge.repetition_hash, which
        # needs the same "is this ep flag legally capturable?" probe). cozy
        # only generates fully LEGAL moves, and its is_legal() is exact (ep
        # pins included) — which matches "legal" mode. For "xfen"
        # (pseudo-legal capturer exists), a legal capture implies a
        # pseudo-legal one; the reverse gap (pinned capturer) is the ep-pin
        # case — handled by just checking pawn adjacency for xfen.
        from imba_chess.eval.cozy_bridge import _ep_adjacent_capturers_cozy

        candidates = _ep_adjacent_capturers_cozy(board)
        if self._ep_ok is chess.Board.has_legal_en_passant:
            return file_idx + 1 if any(board.is_legal(mv) for mv in candidates) else 0
        return file_idx + 1 if candidates else 0  # xfen: pseudo-legal capturer exists

    def encode_cozy(self, board) -> BoardState:
        cfg = self.config
        white_color, black_color, piece_table = _cozy_encode_consts()
        ids = [0] * 64
        white = int(board.colors(white_color))
        not_white = ~white
        for white_id, black_id, piece in piece_table:
            bb = int(board.pieces(piece))
            # chess.scan_forward inlined: it is a generator, so the original
            # cost 12 generator objects plus ~32 __next__ dispatches per call,
            # and this runs once per evaluated search node (428k per 20-game
            # rollout). Same bit-twiddle, no generator machinery.
            w = bb & white
            while w:
                lsb = w & -w
                ids[lsb.bit_length() - 1] = white_id
                w ^= lsb
            b = bb & not_white
            while b:
                lsb = b & -b
                ids[lsb.bit_length() - 1] = black_id
                b ^= lsb
        rights_white = board.castle_rights(white_color)
        rights_black = board.castle_rights(black_color)
        castle_id = (
            (1 if rights_white.short is not None else 0)
            | (2 if rights_white.long is not None else 0)
            | (4 if rights_black.short is not None else 0)
            | (8 if rights_black.long is not None else 0)
        )
        return BoardState(
            piece_ids=ids,
            turn_id=int(board.side_to_move() == black_color),
            castle_id=castle_id,
            ep_file_id=self._ep_file_id_cozy(board),
            halfmove_bucket_id=_bucket(board.halfmove_clock, cfg.halfmove_max, cfg.halfmove_bucket_size),
            fullmove_bucket_id=_bucket(board.fullmove_number, cfg.fullmove_max, cfg.fullmove_bucket_size),
        )
