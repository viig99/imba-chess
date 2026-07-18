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


def test_arm_rows_to_dicts_keeps_every_row_no_padding_or_truncation():
    module = _load_script_module()
    rows = [
        {"move_uci": f"m{i}", "backed_value": float(i), "evals_spent": i, "policy_log_prob": -float(i)}
        for i in range(5)
    ]
    projected = module._arm_rows_to_dicts(rows)
    assert [r["move_uci"] for r in projected] == ["m0", "m1", "m2", "m3", "m4"]


def test_arm_rows_to_dicts_keeps_forcing_floor_arms_beyond_top_m():
    # select_value_search_halving can return more rows than top_m when the
    # root-level forcing floor appends captures/checks/promotions that
    # weren't already in the top-m-by-prior cut. Those must survive into
    # the stored rollout row, not get truncated away.
    module = _load_script_module()
    rows = [
        {"move_uci": "e2e4", "backed_value": 0.1, "evals_spent": 50, "policy_log_prob": -0.2},
        {"move_uci": "d2d4", "backed_value": 0.2, "evals_spent": 50, "policy_log_prob": -0.3},
        {"move_uci": "f3g5", "backed_value": -0.4, "evals_spent": 10, "policy_log_prob": -4.0},
    ]
    projected = module._arm_rows_to_dicts(rows)
    assert len(projected) == 3
    assert projected[-1]["move_uci"] == "f3g5"


def test_arm_rows_to_dicts_maps_none_backed_value_to_zero():
    module = _load_script_module()
    rows = [
        {"move_uci": "e2e4", "backed_value": None, "evals_spent": 0, "policy_log_prob": -0.1},
    ]
    projected = module._arm_rows_to_dicts(rows)
    assert projected[0]["backed_value"] == 0.0


def _run_games_through_scheduler(module, games, *, model, move_vocab, device, dtype, halving_config, concurrent_games):
    """Drive N game coroutines through the real BatchScheduler + merged
    executors this script now uses at runtime (--concurrent-games always
    routes through the scheduler, even at the default of 1) -- the only way
    to exercise _process_game post-coroutine-refactor, and doubling as a
    CPU/tiny-model check that the merged executors don't crash or corrupt
    shapes when concurrent_games > 1."""
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.eval.batch_scheduler import BatchScheduler

    board_state_encoder = BoardStateEncoder()
    done: dict[str, list] = {}
    errors: list[tuple[str, BaseException]] = []

    def game_factory():
        for game in games:
            gen = module._process_game(
                game,
                model=model,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                device=device,
                dtype=dtype,
                halving_config=halving_config,
                every_n_plies=1,
                sample_seed=42,
                checkpoint_path="dummy.pt",
            )
            yield game["game_id"], gen

    scheduler = BatchScheduler(
        game_factory=game_factory(),
        executors={
            "root_eval": module._make_root_eval_executor(
                model=model, device=device, dtype=dtype, stats=None
            ),
            "decode_wave": module._make_decode_wave_executor(
                model=model, device=device, dtype=dtype, stats=None
            ),
        },
        concurrent_games=concurrent_games,
        on_game_done=lambda game_id, rows: done.__setitem__(game_id, rows or []),
        on_game_error=lambda game_id, exc: errors.append((game_id, exc)),
    )
    scheduler.run()
    assert errors == []
    return done


def test_process_game_end_to_end_with_tiny_model(tmp_path):
    import torch as torch_module

    pytest.importorskip("torch")
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

    halving_config = module.HalvingConfig(budget=32, top_m=4, max_depth=2)
    done = _run_games_through_scheduler(
        module,
        [game],
        model=model,
        move_vocab=move_vocab,
        device=torch_module.device("cpu"),
        dtype=torch_module.float32,
        halving_config=halving_config,
        concurrent_games=1,
    )
    rows = done["https://lichess.org/smoketest"]

    assert len(rows) >= 1
    for row in rows:
        assert row.game_id == "https://lichess.org/smoketest"
        assert 0 <= row.ply < 4
        # >= top_m, not ==: the root-level forcing floor in
        # select_value_search_halving can append extra capture/check/promo
        # arms beyond the top-m-by-prior cut, and those must not be dropped.
        assert len(row.arm_move_uci) >= 4
        assert len(row.arm_backed_value) == len(row.arm_move_uci)
        assert len(row.arm_evals_spent) == len(row.arm_move_uci)
        assert len(row.arm_log_prior) == len(row.arm_move_uci)
        assert len(row.root_wdl_unsearched) == 3
        assert abs(sum(row.root_wdl_unsearched) - 1.0) < 1e-4
        assert row.search_refutation_top_r == module.HalvingConfig().refutation_top_r
        assert row.search_expand_top == module.HalvingConfig().expand_top
        assert row.search_lam == module.HalvingConfig().lam


def test_concurrent_games_matches_sequential_with_tiny_model():
    """--concurrent-games > 1 must produce the same rows as concurrent_games=1
    (modulo tiny floating-point differences from batched-matmul reduction
    order): this is the CPU-only sanity check for the merged root_eval /
    grouped decode_wave executors that Task 4 owns (the GPU byte-identical
    gate is a later task's job -- this just guards against a gross
    correctness bug in the merge/split logic using a real tiny model)."""
    import torch as torch_module

    pytest.importorskip("torch")
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

    games = [
        {
            "game_id": "https://lichess.org/g1",
            "result": "1-0",
            "plays": [
                {"move_uci": "e2e4"},
                {"move_uci": "e7e5"},
                {"move_uci": "g1f3"},
                {"move_uci": "b8c6"},
            ],
        },
        {
            "game_id": "https://lichess.org/g2",
            "result": "0-1",
            "plays": [
                {"move_uci": "d2d4"},
                {"move_uci": "d7d5"},
                {"move_uci": "c2c4"},
                {"move_uci": "e7e6"},
                {"move_uci": "b1c3"},
            ],
        },
    ]
    halving_config = module.HalvingConfig(budget=24, top_m=4, max_depth=2)
    kwargs = dict(
        model=model,
        move_vocab=move_vocab,
        device=torch_module.device("cpu"),
        dtype=torch_module.float32,
        halving_config=halving_config,
    )

    sequential = _run_games_through_scheduler(module, games, concurrent_games=1, **kwargs)
    merged = _run_games_through_scheduler(module, games, concurrent_games=2, **kwargs)

    assert set(sequential) == set(merged) == {"https://lichess.org/g1", "https://lichess.org/g2"}
    for game_id in sequential:
        seq_rows, merged_rows = sequential[game_id], merged[game_id]
        assert len(seq_rows) == len(merged_rows)
        for seq_row, merged_row in zip(seq_rows, merged_rows):
            assert seq_row.ply == merged_row.ply
            assert seq_row.best_arm_move_uci == merged_row.best_arm_move_uci
            assert seq_row.arm_move_uci == merged_row.arm_move_uci
            assert seq_row.best_arm_backed_value == pytest.approx(
                merged_row.best_arm_backed_value, abs=1e-4
            )
            assert seq_row.root_wdl_unsearched == pytest.approx(
                merged_row.root_wdl_unsearched, abs=1e-4
            )


def test_merge_decode_requests_fabricated_suffix_rows_match_request_device():
    """Regression test for a --concurrent-games > 1 GPU crash: when one
    game's decode-wave request has no suffix at all (suffix_kv=None, e.g.
    every node in its wave is root-adjacent) and another game's request in
    the same tick has a real suffix, _merge_decode_requests fabricates
    zero-filled suffix_positions/suffix_mask rows for the suffix-less game
    so all rows can be concatenated into one tensor. Those fabricated rows
    must land on the SAME device as the request's own real tensors (read off
    prefix_kv, which is always model-device-resident kv_caches) -- not
    whatever torch's default device happens to be. On a real run this
    crashed as "RuntimeError: Expected all tensors to be on the same device,
    but got ... cpu, different from ... cuda:0" inside torch.cat, because
    the fabricated rows used torch.zeros(...) with no device= argument.

    The actual bug needs a CPU+CUDA mix, which can't run here. This test
    reproduces the identical *class* of bug without any GPU: it temporarily
    makes torch's default device "meta" (a real, distinct-from-cpu device
    that needs no hardware) while every tensor in the fake requests is
    built with an *explicit* device="cpu" -- so if _merge_decode_requests
    ever again creates a fabricated tensor without an explicit device=, it
    silently lands on "meta" instead of "cpu", and the same torch.cat
    device-mismatch RuntimeError fires here that fired on CUDA.
    """
    pytest.importorskip("torch")
    import torch

    from imba_chess.eval.position_evaluator import _DecodeRequest

    module = _load_script_module()

    device = torch.device("cpu")
    num_layers, heads, dqk, dv = 1, 2, 4, 4

    def make_prefix(prefix_len: int):
        return [
            (
                torch.zeros(heads, prefix_len, dqk, device=device),
                torch.zeros(heads, prefix_len, dv, device=device),
            )
            for _ in range(num_layers)
        ]

    def make_new_token_batch(wave_size: int):
        return {
            "piece_ids": torch.zeros(wave_size, 64, dtype=torch.long, device=device),
            "seq_token_id": torch.zeros(wave_size, dtype=torch.long, device=device),
            "turn_id": torch.zeros(wave_size, dtype=torch.long, device=device),
            "castle_id": torch.zeros(wave_size, dtype=torch.long, device=device),
            "ep_file_id": torch.zeros(wave_size, dtype=torch.long, device=device),
            "halfmove_bucket_id": torch.zeros(wave_size, dtype=torch.long, device=device),
            "fullmove_bucket_id": torch.zeros(wave_size, dtype=torch.long, device=device),
            "prev_move_id": torch.zeros(wave_size, dtype=torch.long, device=device),
        }

    # Game 0: no suffix at all for this wave (e.g. every node is root-adjacent).
    req_no_suffix = _DecodeRequest(
        nodes=[object(), object()],
        boards=[None, None],
        new_token_batch=make_new_token_batch(2),
        positions=torch.tensor([1, 1], dtype=torch.long, device=device),
        suffix_kv=None,
        suffix_positions=None,
        suffix_mask=None,
        prefix_kv=make_prefix(3),
        prefix_len=3,
    )
    # Game 1: has a real suffix of length 2, explicitly on `device`.
    suffix_len = 2
    req_with_suffix = _DecodeRequest(
        nodes=[object()],
        boards=[None],
        new_token_batch=make_new_token_batch(1),
        positions=torch.tensor([5], dtype=torch.long, device=device),
        suffix_kv=[
            (
                torch.ones(1, heads, suffix_len, dqk, device=device),
                torch.ones(1, heads, suffix_len, dv, device=device),
            )
            for _ in range(num_layers)
        ],
        suffix_positions=torch.tensor([[3, 4]], dtype=torch.long, device=device),
        suffix_mask=torch.tensor([[True, True]], dtype=torch.bool, device=device),
        prefix_kv=make_prefix(4),
        prefix_len=4,
    )

    torch.set_default_device("meta")
    try:
        merged = module._merge_decode_requests([req_no_suffix, req_with_suffix])
    finally:
        torch.set_default_device("cpu")

    assert merged.suffix_positions.device == device
    assert merged.suffix_mask.device == device
    for k, v in merged.suffix_kv:
        assert k.device == device
        assert v.device == device

    # Rows 0-1 belong to the suffix-less game: zero-filled, all-False mask.
    assert torch.equal(merged.suffix_mask[:2], torch.zeros(2, suffix_len, dtype=torch.bool))
    assert torch.equal(
        merged.suffix_positions[:2], torch.zeros(2, suffix_len, dtype=torch.long)
    )
    for k, v in merged.suffix_kv:
        assert torch.equal(k[:2], torch.zeros_like(k[:2]))
        assert torch.equal(v[:2], torch.zeros_like(v[:2]))

    # Row 2 belongs to the suffix-bearing game: its real content survives.
    assert bool(merged.suffix_mask[2].all())
    assert torch.equal(merged.suffix_positions[2], torch.tensor([3, 4], dtype=torch.long))
