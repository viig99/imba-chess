import pytest

torch = pytest.importorskip("torch")

from imba_chess.data.collate import collate_jagged_batch


def _sample(game_id, seq_len, *, result=1, value_target=None, has_value_target=None):
    return {
        "game_id": game_id,
        "game_result_white": result,
        "seq_token_id": [1] + [0] * (seq_len - 1),
        "piece_ids": [[i] * 64 for i in range(seq_len)],
        "turn_id": [i % 2 for i in range(seq_len)],
        "castle_id": [0] + [15] * (seq_len - 1),
        "ep_file_id": [0] * seq_len,
        "halfmove_bucket_id": [0] * seq_len,
        "fullmove_bucket_id": [0] * seq_len,
        "prev_move_id": [1] * seq_len,
        "target_move_id": [-100] + [4 + i for i in range(seq_len - 1)],
        "played_by_elo": [0] + [2200] * (seq_len - 1),
        "value_target": value_target or [[0.0, 0.0, 0.0]] * seq_len,
        "has_value_target": has_value_target or [0] * seq_len,
    }


def test_collate_jagged_batch_shapes_and_offsets():
    out = collate_jagged_batch([_sample("g1", 3), _sample("g2", 2, result=-1)])
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
    assert out["value_target"].shape == (5, 3)
    assert out["value_target"].dtype == torch.float32
    assert out["has_value_target"].shape == (5,)
    assert out["has_value_target"].dtype == torch.bool


def test_collate_flattens_value_targets_in_token_order():
    out = collate_jagged_batch(
        [
            _sample("g1", 2, value_target=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.7]], has_value_target=[0, 1]),
            _sample("g2", 2),
        ]
    )
    assert out["has_value_target"].tolist() == [False, True, False, False]
    assert out["value_target"][1].tolist() == pytest.approx([0.3, 0.0, 0.7])


@pytest.mark.parametrize("key", ["turn_id", "value_target", "has_value_target"])
def test_collate_jagged_batch_raises_on_mismatched_lengths(key):
    sample = _sample("g1", 3)
    sample[key] = sample[key][:-1]
    with pytest.raises(ValueError, match=key):
        collate_jagged_batch([sample])


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        collate_jagged_batch([])
