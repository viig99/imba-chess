"""Exact differential for the native board-state encoder.

`encode_board_state` replaces BoardStateEncoder.encode_cozy, which ran once per
evaluated search node and made ~10 FFI crossings before bit-twiddling 64
squares in Python. The output is model input, so a wrong id does not raise --
it silently feeds the network a different position.

All three en-passant modes are covered: they disagree with each other by
design, so testing one would hide a mode mix-up.
"""

from __future__ import annotations

import chess
import imba_chess_native as native_cc
import pytest

from imba_chess.data.board_state import BoardStateEncoder, BoardTokenConfig
from tests.test_cozy_bridge import EDGE_FENS, _random_boards

EP_MODES = {"fen": 0, "legal": 1, "xfen": 2}

# Positions where the ep modes actually diverge, plus castling/clock edges.
EXTRA_FENS = [
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",          # ep flag, legal capturer
    "4k3/8/8/3pP3/8/8/8/6K1 w - d6 0 1",          # ep flag, capturer elsewhere
    "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",          # black to move, ep
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",       # all castling rights
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",         # mixed rights
    "8/8/8/8/8/6k1/8/6K1 w - - 99 200",           # clock clamping
    "8/8/8/8/8/6k1/8/6K1 w - - 0 1",              # clocks at zero
]

_RANDOM = _random_boards(150, seed=31337)
_POSITIONS = (
    [chess.Board(fen) for fen in EDGE_FENS + EXTRA_FENS]
    + _RANDOM[:: max(1, len(_RANDOM) // 150)][:150]
)


def _native(board: chess.Board):
    return native_cc.Board.from_fen(board.fen(en_passant="fen"))


@pytest.mark.parametrize("board", _POSITIONS)
@pytest.mark.parametrize("ep_mode", sorted(EP_MODES))
def test_native_encode_matches_python(board: chess.Board, ep_mode: str) -> None:
    cfg = BoardTokenConfig(en_passant=ep_mode)
    encoder = BoardStateEncoder(cfg)
    native_board = _native(board)

    want = encoder._encode_cozy_python(native_board)
    piece_ids, turn_id, castle_id, ep_file_id, halfmove, fullmove = (
        native_cc.encode_board_state(
            native_board,
            EP_MODES[ep_mode],
            cfg.halfmove_max,
            cfg.halfmove_bucket_size,
            cfg.fullmove_max,
            cfg.fullmove_bucket_size,
        )
    )

    assert list(piece_ids) == list(want.piece_ids)
    assert turn_id == want.turn_id
    assert castle_id == want.castle_id
    assert ep_file_id == want.ep_file_id
    assert halfmove == want.halfmove_bucket_id
    assert fullmove == want.fullmove_bucket_id


def test_ep_modes_actually_disagree_somewhere() -> None:
    """Guards the parametrization above: if all three modes agreed everywhere,
    the per-mode cases would be proving nothing about mode handling."""
    disagreements = 0
    for board in _POSITIONS:
        native_board = _native(board)
        ids = {
            native_cc.encode_board_state(native_board, code, 100, 10, 200, 10)[3]
            for code in EP_MODES.values()
        }
        if len(ids) > 1:
            disagreements += 1
    assert disagreements > 0, "no fixture distinguishes the en-passant modes"


def test_piece_ids_use_the_expected_scheme() -> None:
    board = native_cc.Board()
    piece_ids, turn_id, castle_id, ep_file_id, _hm, _fm = (
        native_cc.encode_board_state(board, 0, 100, 10, 200, 10)
    )
    # a1 rook = white rook = 4; e1 king = 6; a8 rook = black rook = 10.
    assert piece_ids[0] == 4 and piece_ids[4] == 6
    assert piece_ids[56] == 10 and piece_ids[60] == 12
    assert piece_ids[24] == 0, "d4 is empty at the start"
    assert turn_id == 0 and castle_id == 0b1111 and ep_file_id == 0


def test_rejects_a_bad_ep_mode_and_bucket_size() -> None:
    board = native_cc.Board()
    with pytest.raises(ValueError, match="ep_mode"):
        native_cc.encode_board_state(board, 9, 100, 10, 200, 10)
    with pytest.raises(ValueError, match="bucket sizes"):
        native_cc.encode_board_state(board, 0, 100, 0, 200, 10)
