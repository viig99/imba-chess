from pathlib import Path

from imba_chess.data.move_vocab import MoveVocab


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

