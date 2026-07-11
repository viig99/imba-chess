import pytest

torch = pytest.importorskip("torch")

from imba_chess.data.collate import collate_jagged_batch


def test_collate_jagged_batch_shapes_and_offsets():
    batch = [
        {
            "game_id": "g1",
            "game_result_white": 1,
            "seq_token_id": [1, 0, 0],
            "piece_ids": [[0] * 64, [1] * 64, [2] * 64],
            "turn_id": [0, 0, 1],
            "castle_id": [0, 15, 15],
            "ep_file_id": [0, 0, 0],
            "halfmove_bucket_id": [0, 0, 0],
            "fullmove_bucket_id": [0, 0, 1],
            "prev_move_id": [1, 1, 4],
            "target_move_id": [-100, 4, 8],
            "played_by_elo": [0, 2200, 2280],
        },
        {
            "game_id": "g2",
            "game_result_white": -1,
            "seq_token_id": [1, 0],
            "piece_ids": [[0] * 64, [3] * 64],
            "turn_id": [0, 0],
            "castle_id": [0, 15],
            "ep_file_id": [0, 0],
            "halfmove_bucket_id": [0, 0],
            "fullmove_bucket_id": [0, 0],
            "prev_move_id": [1, 1],
            "target_move_id": [-100, 3],
            "played_by_elo": [0, 2320],
        },
    ]

    out = collate_jagged_batch(batch)
    assert out["num_games"] == 2
    assert out["game_result_white"].tolist() == [1, -1]
    assert out["total_tokens"] == 5
    assert out["seq_lens"].tolist() == [3, 2]
    assert out["seq_offsets"].tolist() == [0, 3, 5]
    assert out["piece_ids"].shape == (5, 64)
    assert out["seq_token_id"].shape == (5,)
    assert out["target_move_id"][0].item() == -100
    assert out["game_id"] == ["g1", "g2"]
    assert out["turn_id"].dtype == torch.long


def test_collate_jagged_batch_raises_on_mismatched_scalar_lengths():
    batch = [
        {
            "game_id": "g_bad",
            "game_result_white": 1,
            "seq_token_id": [1, 0, 0],
            "piece_ids": [[0] * 64, [1] * 64, [2] * 64],
            "turn_id": [0, 0, 1],
            "castle_id": [0, 15, 15],
            "ep_file_id": [0, 0, 0],
            "halfmove_bucket_id": [0, 0, 0],
            "fullmove_bucket_id": [0, 0, 1],
            "prev_move_id": [1, 1, 4],
            "target_move_id": [-100, 4, 8],
            "played_by_elo": [0, 2200],  # wrong length
        }
    ]

    with pytest.raises(ValueError, match="played_by_elo length"):
        collate_jagged_batch(batch)


def _sample_with_rollout(game_id, seq_len, value_target_soft, has_rollout):
    return {
        "game_id": game_id,
        "game_result_white": 1,
        "seq_token_id": [1] + [0] * (seq_len - 1),
        "piece_ids": [[0] * 64 for _ in range(seq_len)],
        "turn_id": [0] * seq_len,
        "castle_id": [15] * seq_len,
        "ep_file_id": [0] * seq_len,
        "halfmove_bucket_id": [0] * seq_len,
        "fullmove_bucket_id": [0] * seq_len,
        "prev_move_id": [1] * seq_len,
        "target_move_id": [-100] + [4] * (seq_len - 1),
        "played_by_elo": [0] * seq_len,
        "value_target_soft": value_target_soft,
        "has_rollout_value_target": has_rollout,
    }


def test_collate_includes_rollout_fields_when_present_on_every_sample():
    batch = [
        _sample_with_rollout("g1", 2, [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [0, 1]),
        _sample_with_rollout("g2", 2, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0, 0]),
    ]
    out = collate_jagged_batch(batch)
    assert out["value_target_soft"].shape == (4, 3)
    assert out["has_rollout_value_target"].shape == (4,)
    assert out["has_rollout_value_target"].dtype == torch.bool
    assert out["has_rollout_value_target"].tolist() == [False, True, False, False]
    assert out["value_target_soft"][1].tolist() == pytest.approx([0.2, 0.3, 0.5])


def test_collate_raises_on_mixed_rollout_key_presence():
    batch = [
        _sample_with_rollout("g1", 2, [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [0, 1]),
        {
            "game_id": "g2",
            "game_result_white": -1,
            "seq_token_id": [1, 0],
            "piece_ids": [[0] * 64, [1] * 64],
            "turn_id": [0, 0],
            "castle_id": [0, 15],
            "ep_file_id": [0, 0],
            "halfmove_bucket_id": [0, 0],
            "fullmove_bucket_id": [0, 0],
            "prev_move_id": [1, 1],
            "target_move_id": [-100, 3],
            "played_by_elo": [0, 2320],
        },
    ]
    with pytest.raises(ValueError, match="Mixed presence"):
        collate_jagged_batch(batch)


def test_collate_without_rollout_fields_unchanged():
    # Existing test_collate_jagged_batch_shapes_and_offsets already covers
    # this; this test only asserts the new keys are absent.
    batch = [
        _sample_with_rollout("g1", 2, [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [0, 1])
    ]
    del batch[0]["value_target_soft"]
    del batch[0]["has_rollout_value_target"]
    out = collate_jagged_batch(batch)
    assert "value_target_soft" not in out
    assert "has_rollout_value_target" not in out
