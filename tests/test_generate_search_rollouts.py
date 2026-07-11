from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_search_rollouts.py"
    spec = importlib.util.spec_from_file_location("generate_search_rollouts_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load generate_search_rollouts.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_ply_indices_is_deterministic_and_bounded():
    module = _load_script_module()

    first = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g1")
    second = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g1")
    assert first == second
    assert all(0 <= idx < 40 for idx in first)
    assert len(first) >= 1


def test_sample_ply_indices_differs_by_game_id():
    module = _load_script_module()

    a = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g1")
    b = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g2")
    assert a != b or len(a) <= 1  # near-certain to differ with 40 plies / every_n=8


def test_sample_ply_indices_empty_game():
    module = _load_script_module()
    assert module._sample_ply_indices(0, every_n=8, seed=42, game_id="g1") == []


def test_pad_or_truncate_arms_pads_short_lists():
    module = _load_script_module()
    rows = [
        {"move_uci": "e2e4", "backed_value": 0.3, "evals_spent": 100, "policy_log_prob": -0.2},
    ]
    padded = module._pad_or_truncate_arms(rows, top_m=3)
    assert len(padded) == 3
    assert padded[0]["move_uci"] == "e2e4"
    assert padded[1]["move_uci"] == ""
    assert padded[1]["backed_value"] == 0.0
    assert padded[1]["evals_spent"] == 0


def test_pad_or_truncate_arms_truncates_long_lists():
    module = _load_script_module()
    rows = [
        {"move_uci": f"m{i}", "backed_value": float(i), "evals_spent": i, "policy_log_prob": -float(i)}
        for i in range(5)
    ]
    truncated = module._pad_or_truncate_arms(rows, top_m=3)
    assert len(truncated) == 3
    assert [r["move_uci"] for r in truncated] == ["m0", "m1", "m2"]


def test_pad_or_truncate_arms_maps_none_backed_value_to_zero():
    module = _load_script_module()
    rows = [
        {"move_uci": "e2e4", "backed_value": None, "evals_spent": 0, "policy_log_prob": -0.1},
    ]
    padded = module._pad_or_truncate_arms(rows, top_m=1)
    assert padded[0]["backed_value"] == 0.0


def test_process_game_end_to_end_with_tiny_model(tmp_path):
    import torch as torch_module

    pytest.importorskip("torch")
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.move_vocab import MoveVocab
    from imba_chess.model import HSTUChessModel, build_hstu_chess_config
    from imba_chess.config import ModelConfig

    module = _load_script_module()

    move_vocab = MoveVocab.build_static()
    model_cfg = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=16,
            attention_dim=16,
            num_heads=2,
            num_layers=1,
            max_position_embeddings=64,
            enable_value_head=True,
        ),
        move_vocab_size=len(move_vocab),
    )
    model = HSTUChessModel(model_cfg)
    model.eval()

    game = {
        "game_id": "https://lichess.org/smoketest",
        "result": "1-0",
        "plays": [
            {"move_uci": "e2e4"},
            {"move_uci": "e7e5"},
            {"move_uci": "g1f3"},
            {"move_uci": "b8c6"},
        ],
    }

    rows = module._process_game(
        game,
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch_module.device("cpu"),
        dtype=torch_module.float32,
        halving_config=module.HalvingConfig(budget=32, top_m=4, max_depth=2),
        every_n_plies=1,
        sample_seed=42,
        checkpoint_path="dummy.pt",
    )

    assert len(rows) >= 1
    for row in rows:
        assert row.game_id == "https://lichess.org/smoketest"
        assert 0 <= row.ply < 4
        assert len(row.arm_move_uci) == 4
        assert len(row.root_wdl_unsearched) == 3
        assert abs(sum(row.root_wdl_unsearched) - 1.0) < 1e-4
