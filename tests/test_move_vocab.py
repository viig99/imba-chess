from pathlib import Path

import pytest

from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab


def test_move_vocab_build_encode_decode(tmp_path: Path):
    vocab = MoveVocab.build(["e2e4", "e7e5", "e2e4"])

    assert len(vocab) >= 5  # specials + 2 moves
    e2e4_id = vocab.encode("e2e4")
    assert vocab.decode(e2e4_id) == "e2e4"

    unk_id = vocab.encode("a2a3")
    assert unk_id == vocab.unk_id

    path = tmp_path / "move_vocab.json"
    vocab.save(path)
    loaded = MoveVocab.load(path)
    assert loaded.encode("e7e5") == vocab.encode("e7e5")


def test_move_vocab_static_has_expected_size_and_promotions():
    vocab = MoveVocab.build_static()

    # 1792 reachable from->to pairs + 176 promotions + pad/start specials
    assert len(vocab) == 1970
    assert vocab.encode("e2e4") >= 0
    assert vocab.encode("a7a8q") >= 0
    assert vocab.encode("h2h1n") >= 0
    # Castling and knight moves are inside the reachable set.
    assert vocab.encode("e1g1") >= 0
    assert vocab.encode("g1f3") >= 0
    # Geometrically impossible pairs are excluded.
    with pytest.raises(KeyError):
        vocab.encode("a1h2")


def test_move_vocab_static_covers_all_legal_moves_in_sampled_positions():
    import chess

    vocab = MoveVocab.build_static()
    fens = [
        chess.STARTING_FEN,
        # Castling both sides available, en passant square set.
        "r3k2r/pppq1ppp/2npbn2/1B2p3/3PP3/2N1BN2/PPP1QPPP/R3K2R w KQkq - 0 1",
        # Promotions with captures available.
        "rnbq1bnr/ppPppk1p/8/8/8/8/PP1PPpPP/RNBQKBNR w KQ - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            assert vocab.encode(move.uci()) >= 0, f"missing {move.uci()} for {fen}"


def test_move_vocab_static_without_unk_raises_for_unknown():
    vocab = MoveVocab.build_static()
    with pytest.raises(KeyError):
        vocab.encode("zzzz")


def test_load_or_create_static_move_vocab_creates_and_loads(tmp_path: Path):
    path = tmp_path / "static_vocab.json"
    vocab = load_or_create_static_move_vocab(path=path)

    assert path.exists()
    assert len(vocab) == 1970

    loaded = load_or_create_static_move_vocab(path=path)
    assert len(loaded) == 1970
