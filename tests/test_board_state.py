import chess
import pytest

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.models import BoardTokenConfig


def test_start_position_state_fields():
    board = chess.Board()
    state = BoardStateEncoder().encode(board)

    assert len(state.piece_ids) == 64
    assert state.turn_id == 0
    assert state.castle_id == 15
    assert state.ep_file_id == 0
    assert state.halfmove_bucket_id == 0
    assert state.fullmove_bucket_id == 0

    # a1 rook, e1 king, e8 king, a8 rook
    assert state.piece_ids[chess.A1] == 4
    assert state.piece_ids[chess.E1] == 6
    assert state.piece_ids[chess.E8] == 12
    assert state.piece_ids[chess.A8] == 10


def test_piece_ids_from_sparse_board_fen():
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    state = BoardStateEncoder().encode(board)

    assert state.piece_ids[chess.A1] == 6
    assert state.piece_ids[chess.H1] == 12
    assert sum(1 for value in state.piece_ids if value != 0) == 2


def test_castle_id_mask_for_custom_rights():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    state = BoardStateEncoder().encode(board)
    # K (1) + q (8) => 9
    assert state.castle_id == 9


def test_ep_file_id_modes():
    board = chess.Board()
    board.push_san("e4")  # ep square e3 is set in FEN semantics

    legal_state = BoardStateEncoder(BoardTokenConfig(en_passant="legal")).encode(board)
    fen_state = BoardStateEncoder(BoardTokenConfig(en_passant="fen")).encode(board)

    assert legal_state.ep_file_id == 0
    assert fen_state.ep_file_id == 5  # file e => 5 (1-based)


def test_bucket_clamping():
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 150 500")
    config = BoardTokenConfig(
        halfmove_max=100,
        halfmove_bucket_size=2,
        fullmove_max=200,
        fullmove_bucket_size=2,
    )
    state = BoardStateEncoder(config).encode(board)

    assert state.halfmove_bucket_id == 50
    assert state.fullmove_bucket_id == 100


def test_invalid_en_passant_mode_raises():
    # Mode validation now happens eagerly at construction, not on encode().
    with pytest.raises(ValueError):
        BoardStateEncoder(BoardTokenConfig(en_passant="bad"))  # type: ignore[arg-type]
