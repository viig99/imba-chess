import pytest

torch = pytest.importorskip("torch")

from imba_chess.data.dataloader import build_event_dataloader
from imba_chess.data.move_vocab import MoveVocab


class DummyLichessDataset:
    def __init__(self, games):
        self.games = games

    def as_torch_iterable(self, rank=None, world_size=None):
        return iter(self.games)


def _game(game_id: str, first_move: str, second_move: str):
    return {
        "game_id": game_id,
        "plays": [
            {
                "move_uci": first_move,
                "state": {
                    "piece_ids": [0] * 64,
                    "turn_id": 0,
                    "castle_id": 15,
                    "ep_file_id": 0,
                    "halfmove_bucket_id": 0,
                    "fullmove_bucket_id": 0,
                },
            },
            {
                "move_uci": second_move,
                "state": {
                    "piece_ids": [1] * 64,
                    "turn_id": 1,
                    "castle_id": 15,
                    "ep_file_id": 0,
                    "halfmove_bucket_id": 0,
                    "fullmove_bucket_id": 0,
                },
            },
        ],
    }


def test_build_event_dataloader_returns_tensor_dict():
    games = [_game("g1", "e2e4", "e7e5"), _game("g2", "d2d4", "d7d5")]
    vocab = MoveVocab.build_from_games(games)
    dataset = DummyLichessDataset(games)

    loader = build_event_dataloader(
        lichess_dataset=dataset,
        move_vocab=vocab,
        max_tokens_per_batch=1024,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert batch["seq_lens"].tolist() == [3, 3]
    assert batch["seq_offsets"].tolist() == [0, 3, 6]
    assert batch["seq_token_id"].shape == (6,)
    assert batch["piece_ids"].shape == (6, 64)
    assert batch["target_move_id"].dtype == torch.long
    assert batch["game_id"] == ["g1", "g2"]


def test_build_event_dataloader_auto_creates_vocab(tmp_path):
    games = [_game("g1", "e2e4", "e7e5")]
    dataset = DummyLichessDataset(games)
    vocab_path = tmp_path / "auto_vocab.json"

    loader = build_event_dataloader(
        lichess_dataset=dataset,
        max_tokens_per_batch=1024,
        num_workers=0,
        move_vocab_path=vocab_path,
    )

    batch = next(iter(loader))
    assert vocab_path.exists()
    assert batch["seq_token_id"].shape == (3,)


def test_build_event_dataloader_packs_by_max_tokens():
    games = [
        _game("g1", "e2e4", "e7e5"),
        _game("g2", "d2d4", "d7d5"),
        _game("g3", "c2c4", "e7e6"),
    ]
    vocab = MoveVocab.build_from_games(games)
    dataset = DummyLichessDataset(games)

    loader = build_event_dataloader(
        lichess_dataset=dataset,
        move_vocab=vocab,
        max_tokens_per_batch=6,  # each game is 3 tokens (BOS + 2 plies)
        num_workers=0,
    )

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["game_id"] == ["g1", "g2"]
    assert batches[1]["game_id"] == ["g3"]
