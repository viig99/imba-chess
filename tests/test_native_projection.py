"""Exact differential for the production legal-move projection path.

`cozy_bridge.project_legal_moves` is the only implementation of the
vocab-mapping + UCI-sort discipline the search relies on, and it is Rust all
the way down. This file keeps an independent Python oracle so that path is
still checked against something other than itself: the oracle re-derives
castling normalization, vocabulary lookup, and the canonical sort from the
native Board's own primitives, without going through `MoveProjector`.

Because callers gather logits positionally, order is part of correctness --
compare ids, UCIs, move identity, and the total legal count, not just set
membership.
"""

from __future__ import annotations

import chess
import imba_chess_native as native_cc
import pytest

from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.eval import cozy_bridge
from tests.test_cozy_bridge import EDGE_FENS, _random_boards


def _python_project(board, move_vocab: MoveVocab):
    """Independent oracle: what the projector must produce, in plain Python.

    Deliberately not a refactor of the production path -- it recomputes the
    castled king's destination file from square geometry rather than any
    shared helper, so a wrong rule in Rust cannot be mirrored here.
    """
    triples = []
    legal_moves = list(board.generate_moves())
    for move in legal_moves:
        uci = str(move)
        is_castle = (
            board.piece_on(move.from_square) == native_cc.Piece.King
            and board.color_on(move.to_square) == board.side_to_move()
        )
        if is_castle:
            target_file = (
                "g" if int(move.to_square) % 8 > int(move.from_square) % 8 else "c"
            )
            uci = uci[:2] + target_file + uci[3:]
        move_id = move_vocab.token_to_id.get(uci)
        if move_id is not None:
            triples.append((uci, int(move_id), move))
    triples.sort(key=lambda item: item[0])
    return (
        [item[1] for item in triples],
        [item[2] for item in triples],
        [item[0] for item in triples],
        len(legal_moves),
    )


def _move_fields(move) -> tuple[int, int, str | None, str]:
    promotion = None if move.promotion is None else str(move.promotion)
    return int(move.from_square), int(move.to_square), promotion, str(move)


def _move_tokens(vocab: MoveVocab) -> dict[str, int]:
    specials = {
        vocab.config.pad_token,
        vocab.config.start_token,
        vocab.config.unk_token,
    }
    return {
        token: token_id
        for token, token_id in vocab.token_to_id.items()
        if token not in specials
    }


_STATIC_VOCAB = MoveVocab.build_static()

# A vocabulary mapping only a slice of the label space, with ids that do not
# match the static vocab's. Catches a projector that reaches for global state
# or silently falls back to the full label set.
_RESTRICTED_UCIS = sorted(_move_tokens(_STATIC_VOCAB))[::7]
_RESTRICTED_VOCAB = MoveVocab.build(
    _RESTRICTED_UCIS, config=MoveVocabConfig(include_unk=False)
)

_PROJECTOR_RANDOM_BOARDS = _random_boards(200, seed=911)
_PROJECTOR_RANDOM_POSITIONS = _PROJECTOR_RANDOM_BOARDS[
    :: max(1, len(_PROJECTOR_RANDOM_BOARDS) // 200)
][:200]
_POSITIONS = [chess.Board(fen) for fen in EDGE_FENS] + _PROJECTOR_RANDOM_POSITIONS


def _native_board(board: chess.Board):
    return native_cc.Board.from_fen(board.fen(en_passant="fen"))


def _assert_matches_oracle(board: chess.Board, vocab: MoveVocab) -> None:
    native_board = _native_board(board)

    expected_ids, expected_moves, expected_ucis, expected_total = _python_project(
        native_board, vocab
    )
    ids, moves, ucis, total = cozy_bridge.project_legal_moves(native_board, vocab)

    assert total == expected_total
    assert ids == expected_ids
    assert ucis == expected_ucis
    assert [_move_fields(move) for move in moves] == [
        _move_fields(move) for move in expected_moves
    ]


@pytest.mark.parametrize("board", _POSITIONS)
def test_native_projection_matches_python_oracle(board: chess.Board) -> None:
    _assert_matches_oracle(board, _STATIC_VOCAB)


@pytest.mark.parametrize("board", _POSITIONS)
def test_native_projection_matches_oracle_on_a_restricted_vocabulary(
    board: chess.Board,
) -> None:
    _assert_matches_oracle(board, _RESTRICTED_VOCAB)


def test_restricted_vocabulary_actually_drops_moves() -> None:
    """Guards the case above from passing vacuously."""
    dropped = 0
    for board in _POSITIONS:
        native_board = _native_board(board)
        full = len(cozy_bridge.project_legal_moves(native_board, _STATIC_VOCAB)[0])
        part = len(cozy_bridge.project_legal_moves(native_board, _RESTRICTED_VOCAB)[0])
        assert part <= full
        dropped += full - part
    assert dropped > 0


def test_two_vocabularies_stay_independent_when_interleaved() -> None:
    for board in _POSITIONS[:40]:
        _assert_matches_oracle(board, _STATIC_VOCAB)
        _assert_matches_oracle(board, _RESTRICTED_VOCAB)
        _assert_matches_oracle(board, _STATIC_VOCAB)


def test_projection_covers_every_legal_move_of_a_full_vocabulary() -> None:
    """The static vocab is a superset of legal standard-chess moves, so nothing
    may be dropped -- a silent drop would starve the search of a candidate."""
    for board in _POSITIONS:
        ids, _, _, total = cozy_bridge.project_legal_moves(
            _native_board(board), _STATIC_VOCAB
        )
        assert len(ids) == total


def test_castling_projects_to_the_king_destination_but_stays_playable() -> None:
    """The one convention the oracle and the projector must both encode: the
    vocabulary sees e1g1, the caller gets back the raw king-takes-rook move."""
    board = native_cc.Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    _, moves, ucis, _ = cozy_bridge.project_legal_moves(board, _STATIC_VOCAB)

    raw_by_uci = {uci: move for uci, move in zip(ucis, moves)}
    assert str(raw_by_uci["e1g1"]) == "e1h1"
    assert str(raw_by_uci["e1c1"]) == "e1a1"
    for uci in ("e1g1", "e1c1"):
        copy_board = board.__copy__()
        copy_board.play(raw_by_uci[uci])
