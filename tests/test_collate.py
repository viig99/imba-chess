import pytest

torch = pytest.importorskip("torch")

from imba_chess.data.collate import collate_jagged_batch


def test_collate_jagged_batch_shapes_and_offsets():
    batch = [
        {
            "game_id": "g1",
            "seq_token_id": [1, 0, 0],
            "piece_ids": [[0] * 64, [1] * 64, [2] * 64],
            "turn_id": [0, 0, 1],
            "castle_id": [0, 15, 15],
            "ep_file_id": [0, 0, 0],
            "halfmove_bucket_id": [0, 0, 0],
            "fullmove_bucket_id": [0, 0, 1],
            "prev_move_id": [1, 1, 4],
            "target_move_id": [-100, 4, 8],
        },
        {
            "game_id": "g2",
            "seq_token_id": [1, 0],
            "piece_ids": [[0] * 64, [3] * 64],
            "turn_id": [0, 0],
            "castle_id": [0, 15],
            "ep_file_id": [0, 0],
            "halfmove_bucket_id": [0, 0],
            "fullmove_bucket_id": [0, 0],
            "prev_move_id": [1, 1],
            "target_move_id": [-100, 3],
        },
    ]

    out = collate_jagged_batch(batch)
    assert out["num_games"] == 2
    assert out["total_tokens"] == 5
    assert out["seq_lens"].tolist() == [3, 2]
    assert out["seq_offsets"].tolist() == [0, 3, 5]
    assert out["piece_ids"].shape == (5, 64)
    assert out["seq_token_id"].shape == (5,)
    assert out["target_move_id"][0].item() == -100
    assert out["game_id"] == ["g1", "g2"]
    assert out["turn_id"].dtype == torch.long

