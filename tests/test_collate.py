import pytest

torch = pytest.importorskip("torch")

from imba_chess.data.collate import collate_batch


def test_collate_batch_shapes_and_masks():
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
            "target_move_id": [0, 4, 8],
            "attention_mask": [1, 1, 1],
            "loss_mask": [0, 1, 1],
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
            "target_move_id": [0, 3],
            "attention_mask": [1, 1],
            "loss_mask": [0, 1],
        },
    ]

    out = collate_batch(batch)

    assert out["seq_token_id"].shape == (2, 3)
    assert out["piece_ids"].shape == (2, 3, 64)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["loss_mask"].tolist() == [[0, 1, 1], [0, 1, 0]]
    assert out["game_id"] == ["g1", "g2"]
    assert out["seq_token_id"].dtype == torch.long

