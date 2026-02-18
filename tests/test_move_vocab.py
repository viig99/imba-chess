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

    # 4208 static UCI tokens + pad/start specials
    assert len(vocab) == 4210
    assert vocab.encode("e2e4") >= 0
    assert vocab.encode("a7a8q") >= 0
    assert vocab.encode("h2h1n") >= 0


def test_move_vocab_static_without_unk_raises_for_unknown():
    vocab = MoveVocab.build_static()
    with pytest.raises(KeyError):
        vocab.encode("zzzz")


def test_load_or_create_static_move_vocab_creates_and_loads(tmp_path: Path):
    path = tmp_path / "static_vocab.json"
    vocab = load_or_create_static_move_vocab(path=path)

    assert path.exists()
    assert len(vocab) == 4210

    loaded = load_or_create_static_move_vocab(path=path)
    assert len(loaded) == 4210
