import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from imba_chess.config import ModelConfig
from imba_chess.model import HSTUChessConfig, HSTUChessModel
from imba_chess.model.hstu_model import build_hstu_chess_config


def _batch():
    seq_lens = torch.tensor([3, 2], dtype=torch.long)
    seq_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    total_tokens = int(seq_offsets[-1].item())

    return {
        "game_id": ["g1", "g2"],
        "game_result_white": torch.tensor([1, -1], dtype=torch.long),
        "num_games": 2,
        "total_tokens": total_tokens,
        "seq_lens": seq_lens,
        "seq_offsets": seq_offsets,
        "piece_ids": torch.zeros((total_tokens, 64), dtype=torch.long),
        "seq_token_id": torch.tensor([1, 0, 0, 1, 0], dtype=torch.long),
        "turn_id": torch.tensor([0, 0, 1, 0, 0], dtype=torch.long),
        "castle_id": torch.tensor([0, 15, 15, 0, 15], dtype=torch.long),
        "ep_file_id": torch.zeros(total_tokens, dtype=torch.long),
        "halfmove_bucket_id": torch.zeros(total_tokens, dtype=torch.long),
        "fullmove_bucket_id": torch.zeros(total_tokens, dtype=torch.long),
        "prev_move_id": torch.tensor([1, 1, 10, 1, 12], dtype=torch.long),
        "target_move_id": torch.tensor([-100, 10, 12, -100, 15], dtype=torch.long),
        "played_by_elo": torch.tensor([0, 2250, 2400, 0, 2300], dtype=torch.long),
    }


def test_hstu_chess_model_forward_shapes_and_loss():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,  # avoid flex attention dependency for unit smoke test
        max_position_embeddings=32,
    )
    model = HSTUChessModel(config)
    batch = _batch()

    out = model(batch)
    assert out["logits"].shape == (5, 128)
    assert out["loss"].ndim == 0


def test_build_hstu_chess_config_from_repo_model_section():
    repo_model = ModelConfig(model_dim=256, num_layers=4, dropout=0.2)
    cfg = build_hstu_chess_config(repo_model, move_vocab_size=4210)

    assert cfg.move_vocab_size == 4210
    assert cfg.model_dim == 256
    assert cfg.num_layers == 4
    assert cfg.dropout == 0.2


def test_hstu_chess_model_loss_is_finite_when_all_targets_ignored():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    batch["target_move_id"] = torch.full_like(batch["target_move_id"], -100)

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-8)


def test_hstu_chess_model_weighted_loss_requires_played_by_elo_column():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        elo_loss_weight_strength=1.0,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    del batch["played_by_elo"]

    with pytest.raises(KeyError, match="played_by_elo"):
        model(batch, return_loss=True)


def test_hstu_chess_model_elo_weighted_loss_matches_manual_formula():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        label_smoothing=0.05,
        elo_weight_min_elo=2200,
        elo_weight_max_elo=2800,
        elo_loss_weight_alpha=1.5,
        elo_loss_weight_strength=1.2,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    batch["played_by_elo"] = torch.tensor([0, 2200, 2500, 0, 3000], dtype=torch.long)

    out = model(batch, return_loss=True)
    logits = out["logits"]
    targets = batch["target_move_id"].to(dtype=torch.long, device=logits.device)
    valid_mask = targets != config.ignore_index
    safe_targets = targets.masked_fill(~valid_mask, 0)
    per_token_loss = F.cross_entropy(
        logits.float(),
        safe_targets,
        reduction="none",
        label_smoothing=config.label_smoothing,
    )
    played_by_elo = batch["played_by_elo"].to(dtype=per_token_loss.dtype, device=logits.device)
    elo_norm = (
        (played_by_elo - config.elo_weight_min_elo)
        / (config.elo_weight_max_elo - config.elo_weight_min_elo)
    ).clamp(min=0.0, max=1.0)
    elo_curve = elo_norm.pow(config.elo_loss_weight_alpha)
    token_weights = valid_mask.to(per_token_loss.dtype) * (
        1.0 + config.elo_loss_weight_strength * elo_curve
    )
    expected = (per_token_loss * token_weights).sum() / token_weights.sum().clamp_min(1.0)
    assert torch.allclose(out["policy_loss"], expected, atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_elo_strength_zero_matches_unweighted_loss():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        label_smoothing=0.05,
        elo_loss_weight_strength=0.0,
    )
    model = HSTUChessModel(config)
    batch = _batch()

    out = model(batch, return_loss=True)
    logits = out["logits"]
    targets = batch["target_move_id"].to(dtype=torch.long, device=logits.device)
    valid_mask = targets != config.ignore_index
    safe_targets = targets.masked_fill(~valid_mask, 0)
    per_token_loss = F.cross_entropy(
        logits.float(),
        safe_targets,
        reduction="none",
        label_smoothing=config.label_smoothing,
    )
    expected = (
        per_token_loss * valid_mask.to(per_token_loss.dtype)
    ).sum() / valid_mask.to(per_token_loss.dtype).sum().clamp_min(1.0)
    assert torch.allclose(out["policy_loss"], expected, atol=1e-6, rtol=1e-6)


def test_elo_normalization_clamps_at_config_bounds():
    config = HSTUChessConfig(
        move_vocab_size=128,
        elo_weight_min_elo=2200,
        elo_weight_max_elo=2800,
    )
    elo = torch.tensor([0.0, 2199.0, 2200.0, 2500.0, 2800.0, 4000.0])
    elo_norm = (
        (elo - config.elo_weight_min_elo)
        / (config.elo_weight_max_elo - config.elo_weight_min_elo)
    ).clamp(min=0.0, max=1.0)
    assert elo_norm.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5, 1.0, 1.0])


def test_hstu_chess_model_value_head_outputs_and_combines_loss():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.25,
    )
    model = HSTUChessModel(config)
    batch = _batch()

    out = model(batch, return_loss=True)
    assert out["value_logits"].shape == (5, 3)
    assert out["policy_loss"].ndim == 0
    assert out["value_loss"].ndim == 0
    expected = (
        out["policy_loss"]
        + config.value_loss_weight * out["value_loss"]
        + config.moves_left_loss_weight * out["moves_left_loss"]
    )
    assert torch.allclose(out["loss"], expected, atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_value_loss_matches_manual_formula():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
        value_weight_alpha=1.5,
        value_label_smoothing=0.05,
    )
    model = HSTUChessModel(config)
    batch = _batch()

    out = model(batch, return_loss=True)
    value_logits = out["value_logits"]
    seq_offsets = batch["seq_offsets"].to(device=value_logits.device, dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(
        torch.arange(batch["num_games"], device=value_logits.device),
        counts,
    )
    game_result_white = batch["game_result_white"].to(
        device=value_logits.device, dtype=torch.long
    )
    z_token = game_result_white[token_game_id]
    turn_id = batch["turn_id"].to(device=value_logits.device, dtype=torch.long)
    y = torch.where(turn_id == 0, z_token, -z_token)
    value_target = (y + 1).clamp(min=0, max=2)

    token_pos = torch.arange(value_logits.shape[0], device=value_logits.device) - seq_offsets[
        token_game_id
    ]
    seq_len = counts[token_game_id].clamp_min(1)
    progress = token_pos.to(torch.float32) / (seq_len.to(torch.float32) - 1.0).clamp_min(
        1.0
    )
    valid_mask = (
        batch["target_move_id"].to(device=value_logits.device, dtype=torch.long)
        != config.ignore_index
    )
    value_weights = progress.pow(config.value_weight_alpha) * valid_mask.to(torch.float32)
    per_token_loss = F.cross_entropy(
        value_logits.float(),
        value_target,
        reduction="none",
        label_smoothing=config.value_label_smoothing,
    )
    expected = (per_token_loss * value_weights).sum() / value_weights.sum().clamp_min(1.0)
    assert torch.allclose(out["value_loss"], expected, atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_value_loss_elo_weighted_matches_manual_formula():
    kwargs = dict(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
        value_weight_alpha=1.5,
        elo_weight_min_elo=1000,
        elo_weight_max_elo=3000,
        elo_loss_weight_alpha=1.0,
    )
    config = HSTUChessConfig(elo_loss_weight_strength=1.0, **kwargs)
    torch.manual_seed(0)
    model = HSTUChessModel(config)
    batch = _batch()
    batch["played_by_elo"] = torch.tensor(
        [0, 1500, 2800, 0, 2000, 2500, 1000], dtype=torch.long
    )[: batch["seq_token_id"].numel()]

    out = model(batch, return_loss=True)
    value_logits = out["value_logits"]
    device = value_logits.device
    seq_offsets = batch["seq_offsets"].to(device=device, dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(
        torch.arange(batch["num_games"], device=device), counts
    )
    z_token = batch["game_result_white"].to(device=device, dtype=torch.long)[token_game_id]
    turn_id = batch["turn_id"].to(device=device, dtype=torch.long)
    value_target = (torch.where(turn_id == 0, z_token, -z_token) + 1).clamp(min=0, max=2)

    token_pos = torch.arange(value_logits.shape[0], device=device) - seq_offsets[token_game_id]
    seq_len = counts[token_game_id].clamp_min(1)
    progress = token_pos.to(torch.float32) / (seq_len.to(torch.float32) - 1.0).clamp_min(1.0)
    valid_mask = (
        batch["target_move_id"].to(device=device, dtype=torch.long) != config.ignore_index
    )
    elo_norm = (
        (batch["played_by_elo"].to(device=device, dtype=torch.float32) - 1000) / 2000
    ).clamp(min=0.0, max=1.0)
    elo_scale = 1.0 + 1.0 * elo_norm.pow(1.0)
    value_weights = (
        progress.pow(config.value_weight_alpha) * valid_mask.to(torch.float32) * elo_scale
    )
    per_token_loss = F.cross_entropy(
        value_logits.float(), value_target, reduction="none"
    )
    expected = (per_token_loss * value_weights).sum() / value_weights.sum().clamp_min(1.0)
    assert torch.allclose(out["value_loss"], expected, atol=1e-6, rtol=1e-6)

    # strength=0 on identical weights must give a different (unweighted) loss.
    torch.manual_seed(0)
    model_unweighted = HSTUChessModel(HSTUChessConfig(elo_loss_weight_strength=0.0, **kwargs))
    out_unweighted = model_unweighted(batch, return_loss=True)
    assert not torch.allclose(out["value_loss"], out_unweighted["value_loss"])


def test_hstu_chess_model_value_loss_is_finite_when_all_targets_ignored():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    batch["target_move_id"] = torch.full_like(batch["target_move_id"], -100)

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["value_loss"])
    assert out["value_loss"].item() == pytest.approx(0.0, abs=1e-8)


def test_hstu_chess_model_moves_left_loss_matches_manual_formula():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        moves_left_loss_weight=0.07,
    )
    model = HSTUChessModel(config)
    batch = _batch()

    out = model(batch, return_loss=True)
    pred = out["moves_left_pred"]
    assert pred.shape == (5,)

    seq_offsets = batch["seq_offsets"]
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(
        torch.arange(batch["num_games"]), counts
    )
    token_pos = torch.arange(pred.shape[0]) - seq_offsets[token_game_id]
    seq_len = counts[token_game_id].clamp_min(1)
    plies_left = (seq_len - 1 - token_pos).clamp_min(0)
    target = torch.log1p(plies_left.to(torch.float32))

    valid_mask = (batch["target_move_id"] != config.ignore_index).to(torch.float32)
    per_token = F.huber_loss(pred, target, reduction="none")
    expected = (per_token * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    assert torch.allclose(out["moves_left_loss"], expected, atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_moves_left_head_receives_gradients():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
    )
    model = HSTUChessModel(config)
    out = model(_batch(), return_loss=True)
    out["loss"].backward()

    grads = [p.grad for p in model.moves_left_head.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_board_embedding_distinguishes_piece_placement():
    """Same material on different squares must embed differently (the old
    additive piece+square scheme collapsed to a bag of material)."""
    import chess

    from imba_chess.data.board_state import BoardStateEncoder

    config = HSTUChessConfig(move_vocab_size=32, num_layers=0, dropout=0.0)
    model = HSTUChessModel(config)
    encoder = BoardStateEncoder()

    startpos = chess.Board()
    same_material = chess.Board(
        "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    )
    board_missing_rook = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"
    )

    def embed(board: chess.Board) -> torch.Tensor:
        piece_ids = torch.tensor([encoder.encode(board).piece_ids])
        return model._embed_board(piece_ids)

    assert not torch.allclose(embed(startpos), embed(same_material))
    assert not torch.allclose(embed(startpos), embed(board_missing_rook))


def test_stu_layer_per_head_position_bias_forward_backward():
    from imba_chess.model.hstu_attention import SequentialTransductionUnitJagged

    layer = SequentialTransductionUnitJagged(
        embedding_dim=16,
        linear_hidden_dim=8,
        attention_dim=8,
        dropout_ratio=0.0,
        num_heads=2,
        max_seq_len=32,
    )
    assert layer._ps_w.shape == (2, 63)  # [num_heads, 2 * max_seq_len - 1]

    x = torch.randn(5, 16)
    out = layer(x, block_mask=None)
    assert out.shape == (5, 16)
    out.sum().backward()
    assert layer._ps_w.grad is not None
    # Each head must receive its own bias gradient.
    assert not torch.equal(layer._ps_w.grad[0], layer._ps_w.grad[1])


def test_optimizer_decay_groups_exclude_embeddings_norms_and_biases():
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_for_test", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = HSTUChessConfig(
        move_vocab_size=64,
        model_dim=32,
        linear_hidden_dim=8,
        attention_dim=8,
        num_heads=2,
        num_layers=2,
        max_position_embeddings=32,
        enable_value_head=True,
    )
    model = HSTUChessModel(config)
    groups = module._build_decay_param_groups(model, weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}

    named = dict(model.named_parameters())
    assert decay_ids.isdisjoint(no_decay_ids)
    assert len(decay_ids) + len(no_decay_ids) == len(named)

    for name, param in named.items():
        if (
            "embedding" in name
            or name.endswith(".bias")
            or "_ps_w" in name
            or "final_norm" in name
        ):
            assert id(param) in no_decay_ids, f"{name} should not decay"

    # prediction_head.weight is tied to prev_move_embedding.weight and must
    # get embedding treatment (no decay), counted once.
    assert model.prediction_head.weight is model.prev_move_embedding.weight
    assert id(model.prediction_head.weight) in no_decay_ids
    assert id(named["layers.0._uvqk.weight"]) in decay_ids
    assert id(named["value_head.0.weight"]) in decay_ids
    assert id(named["board_encoder.blocks.0.qkv.weight"]) in decay_ids


def test_hstu_chess_model_value_loss_uses_soft_ce_for_gated_tokens():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()

    value_target_soft = torch.zeros((total_tokens, 3), dtype=torch.float32)
    value_target_soft[1] = torch.tensor([0.1, 0.2, 0.7])
    has_rollout_value_target = torch.zeros(total_tokens, dtype=torch.bool)
    has_rollout_value_target[1] = True
    batch["value_target_soft"] = value_target_soft
    batch["has_rollout_value_target"] = has_rollout_value_target

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["value_loss"])

    # Manually recompute expected per-token loss: soft CE at token 1, hard CE elsewhere.
    value_logits = out["value_logits"]
    seq_offsets = batch["seq_offsets"].to(dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(torch.arange(batch["num_games"]), counts)
    z_token = batch["game_result_white"].to(dtype=torch.long)[token_game_id]
    turn_id = batch["turn_id"].to(dtype=torch.long)
    value_target = (torch.where(turn_id == 0, z_token, -z_token) + 1).clamp(min=0, max=2)
    hard_loss = F.cross_entropy(value_logits.float(), value_target, reduction="none")
    soft_loss = -(value_target_soft * F.log_softmax(value_logits.float(), dim=-1)).sum(dim=-1)
    expected_per_token = torch.where(has_rollout_value_target, soft_loss, hard_loss)

    token_pos = torch.arange(value_logits.shape[0]) - seq_offsets[token_game_id]
    seq_len = counts[token_game_id].clamp_min(1)
    progress = token_pos.to(torch.float32) / (seq_len.to(torch.float32) - 1.0).clamp_min(1.0)
    valid_mask = batch["target_move_id"].to(dtype=torch.long) != config.ignore_index
    value_weights = progress.pow(config.value_weight_alpha) * valid_mask.to(torch.float32)
    expected = (expected_per_token * value_weights).sum() / value_weights.sum().clamp_min(1.0)

    assert torch.allclose(out["value_loss"], expected, atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_value_loss_beta_zero_soft_target_matches_hard_ce():
    # A one-hot soft target must produce numerically the same per-token loss
    # as the existing hard-CE path (this is what beta=0 relies on).
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
    )
    torch.manual_seed(0)
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()

    out_baseline = model(batch, return_loss=True)

    # one_hot(value_target) as the soft target, gated everywhere valid.
    seq_offsets = batch["seq_offsets"].to(dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(torch.arange(batch["num_games"]), counts)
    z_token = batch["game_result_white"].to(dtype=torch.long)[token_game_id]
    turn_id = batch["turn_id"].to(dtype=torch.long)
    value_target = (torch.where(turn_id == 0, z_token, -z_token) + 1).clamp(min=0, max=2)
    batch["value_target_soft"] = F.one_hot(value_target, num_classes=3).to(torch.float32)
    batch["has_rollout_value_target"] = torch.ones(total_tokens, dtype=torch.bool)

    torch.manual_seed(0)
    model_gated = HSTUChessModel(config)
    out_gated = model_gated(batch, return_loss=True)

    assert torch.allclose(out_gated["value_loss"], out_baseline["value_loss"], atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_value_loss_soft_ce_applies_label_smoothing():
    # The soft-CE branch must smooth its (non-one-hot) targets the same way
    # F.cross_entropy's label_smoothing would smooth a hard target, so that
    # rollout-gated and non-gated tokens train against consistently-smoothed
    # targets when value_label_smoothing > 0.
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
        value_label_smoothing=0.1,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()

    # Non-trivial (non-one-hot) soft targets, gated on every token.
    value_target_soft = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.05, 0.85, 0.10],
            [0.33, 0.33, 0.34],
        ],
        dtype=torch.float32,
    )
    has_rollout_value_target = torch.ones(total_tokens, dtype=torch.bool)
    batch["value_target_soft"] = value_target_soft
    batch["has_rollout_value_target"] = has_rollout_value_target

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["value_loss"])

    value_logits = out["value_logits"]
    eps = config.value_label_smoothing
    num_classes = value_target_soft.shape[-1]
    smoothed_soft_targets = (1.0 - eps) * value_target_soft + eps / num_classes
    expected_per_token = -(
        smoothed_soft_targets * F.log_softmax(value_logits.float(), dim=-1)
    ).sum(dim=-1)

    seq_offsets = batch["seq_offsets"].to(dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(torch.arange(batch["num_games"]), counts)
    token_pos = torch.arange(value_logits.shape[0]) - seq_offsets[token_game_id]
    seq_len = counts[token_game_id].clamp_min(1)
    progress = token_pos.to(torch.float32) / (seq_len.to(torch.float32) - 1.0).clamp_min(1.0)
    valid_mask = batch["target_move_id"].to(dtype=torch.long) != config.ignore_index
    value_weights = progress.pow(config.value_weight_alpha) * valid_mask.to(torch.float32)
    expected = (expected_per_token * value_weights).sum() / value_weights.sum().clamp_min(1.0)

    assert torch.allclose(out["value_loss"], expected, atol=1e-6, rtol=1e-6)

    # Sanity: without smoothing (eps=0) the loss must differ, proving the
    # smoothing transform actually has an effect (not a no-op formula).
    config_unsmoothed = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
        value_label_smoothing=0.0,
    )
    torch.manual_seed(0)
    model_unsmoothed = HSTUChessModel(config_unsmoothed)
    torch.manual_seed(0)
    model_smoothed = HSTUChessModel(config)
    out_unsmoothed = model_unsmoothed(batch, return_loss=True)
    out_smoothed = model_smoothed(batch, return_loss=True)
    assert not torch.allclose(out_unsmoothed["value_loss"], out_smoothed["value_loss"])


def test_hstu_chess_model_value_loss_without_rollout_keys_unchanged():
    # Regression: a batch entirely lacking the two optional keys must behave
    # exactly as before this change.
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
    )
    torch.manual_seed(0)
    model_a = HSTUChessModel(config)
    torch.manual_seed(0)
    model_b = HSTUChessModel(config)
    batch = _batch()

    # Reseed immediately before each forward so the two calls draw identical
    # dropout masks; without this, model_a's forward advances the global RNG
    # and model_b's forward would (legitimately, for unrelated reasons) see a
    # different dropout stream even though both models have identical
    # weights. That would confound the invariant under test.
    torch.manual_seed(1)
    out_a = model_a(batch, return_loss=True)
    torch.manual_seed(1)
    out_b = model_b(batch, return_loss=True)
    assert torch.allclose(out_a["value_loss"], out_b["value_loss"], atol=1e-12)


def test_hstu_chess_model_policy_kl_loss_matches_manual_formula():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        policy_kl_weight=0.2,
        policy_kl_sigma=1.5,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()
    max_arms = 4

    arm_ids = torch.zeros((total_tokens, max_arms), dtype=torch.long)
    arm_qhat = torch.zeros((total_tokens, max_arms), dtype=torch.float32)
    arm_mask = torch.zeros((total_tokens, max_arms), dtype=torch.bool)
    has_rollout_policy_target = torch.zeros(total_tokens, dtype=torch.bool)

    # Token 1 gets two real arms (ids 10, 20); token 3 gets one real arm (id 5).
    arm_ids[1, :2] = torch.tensor([10, 20])
    arm_qhat[1, :2] = torch.tensor([0.4, -0.3])
    arm_mask[1, :2] = True
    has_rollout_policy_target[1] = True

    arm_ids[3, :1] = torch.tensor([5])
    arm_qhat[3, :1] = torch.tensor([0.9])
    arm_mask[3, :1] = True
    has_rollout_policy_target[3] = True

    batch["policy_kl_arm_ids"] = arm_ids
    batch["policy_kl_arm_qhat"] = arm_qhat
    batch["policy_kl_arm_mask"] = arm_mask
    batch["has_rollout_policy_target"] = has_rollout_policy_target

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["policy_kl_loss"])
    assert "policy_kl_loss" in out

    # Manually recompute expected per-token loss.
    policy_logits = out["logits"]
    student_arm_logits = torch.gather(policy_logits.float(), dim=-1, index=arm_ids)
    target_arm_logits = student_arm_logits.detach() + config.policy_kl_sigma * arm_qhat
    neg_inf_fill = torch.finfo(student_arm_logits.dtype).min
    masked_target = target_arm_logits.masked_fill(~arm_mask, neg_inf_fill)
    masked_student = student_arm_logits.masked_fill(~arm_mask, neg_inf_fill)
    target = F.softmax(masked_target, dim=-1)
    student_log_probs = F.log_softmax(masked_student, dim=-1)
    expected_per_token = -(target * student_log_probs).sum(dim=-1)

    valid_mask = batch["target_move_id"].to(dtype=torch.long) != config.ignore_index
    weights = has_rollout_policy_target.to(torch.float32) * valid_mask.to(torch.float32)
    expected = (expected_per_token * weights).sum() / weights.sum().clamp_min(1.0)

    assert torch.allclose(out["policy_kl_loss"], expected, atol=1e-5, rtol=1e-5)

    # total_loss must include policy_kl_weight * policy_kl_loss.
    assert torch.allclose(
        out["loss"],
        out["policy_loss"]
        + config.moves_left_loss_weight * out["moves_left_loss"]
        + config.policy_kl_weight * out["policy_kl_loss"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_hstu_chess_model_policy_kl_target_detaches_student_gradient():
    # The target side must not receive gradient through student_arm_logits --
    # otherwise this becomes a degenerate/self-referential loss (a classic
    # self-distillation footgun).
    #
    # Isolation matters here: `prediction_head` is shared with the main
    # next-move policy_loss, which flows real gradient into it independently
    # of policy_kl_loss (the _batch() fixture has non-ignored targets at
    # several tokens). Backward()-ing on the combined out["loss"] therefore
    # can't distinguish a correct vs. broken detach -- prediction_head gets
    # a large nonzero gradient from policy_loss alone either way. Instead,
    # backward specifically on out["policy_kl_loss"] to isolate the KL
    # term's own gradient contribution.
    #
    # With arm_qhat == 0 everywhere and a correct detach, target_arm_logits
    # == student_arm_logits.detach() numerically (detach changes graph
    # connectivity, not values), so target == softmax(student) exactly. The
    # cross-entropy gradient w.r.t. logits is softmax(logits) - target,
    # which is then exactly zero at this point -- so a correctly detached
    # loss must produce a ~0 (float-noise-level) gradient into
    # prediction_head via this path alone. A missing detach makes the
    # target implicitly a function of the student's own weights too, so
    # backprop picks up an extra, non-vanishing term through that
    # dependency -- empirically ~3 orders of magnitude larger than the
    # float-noise floor of the correct version.
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        policy_kl_weight=1.0,
        policy_kl_sigma=1.0,
    )
    torch.manual_seed(0)
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()
    max_arms = 4

    arm_ids = torch.zeros((total_tokens, max_arms), dtype=torch.long)
    arm_qhat = torch.zeros((total_tokens, max_arms), dtype=torch.float32)
    arm_mask = torch.zeros((total_tokens, max_arms), dtype=torch.bool)
    has_rollout_policy_target = torch.zeros(total_tokens, dtype=torch.bool)
    arm_ids[1, :2] = torch.tensor([10, 20])
    arm_mask[1, :2] = True
    has_rollout_policy_target[1] = True

    batch["policy_kl_arm_ids"] = arm_ids
    batch["policy_kl_arm_qhat"] = arm_qhat
    batch["policy_kl_arm_mask"] = arm_mask
    batch["has_rollout_policy_target"] = has_rollout_policy_target

    out = model(batch, return_loss=True)
    out["policy_kl_loss"].backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.prediction_head.parameters()
        if p.grad is not None
    ]
    assert grad_norms, "expected prediction_head to accumulate a gradient"
    assert all(norm < 1e-4 for norm in grad_norms)


def test_hstu_chess_model_policy_kl_masks_padding_and_uncovered_tokens():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        policy_kl_weight=1.0,
        policy_kl_sigma=1.0,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()
    max_arms = 4

    # No token has has_rollout_policy_target=True -- policy_kl_loss must be
    # a well-defined finite number (0-weight average, not NaN from an
    # empty/degenerate reduction).
    batch["policy_kl_arm_ids"] = torch.zeros((total_tokens, max_arms), dtype=torch.long)
    batch["policy_kl_arm_qhat"] = torch.zeros((total_tokens, max_arms), dtype=torch.float32)
    batch["policy_kl_arm_mask"] = torch.zeros((total_tokens, max_arms), dtype=torch.bool)
    batch["has_rollout_policy_target"] = torch.zeros(total_tokens, dtype=torch.bool)

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["policy_kl_loss"])
    assert out["policy_kl_loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_hstu_chess_model_policy_kl_weight_zero_skips_computation_entirely():
    # policy_kl_weight=0.0 must short-circuit the whole policy-KL block --
    # not just zero out its contribution to total_loss -- so
    # output["policy_kl_loss"] must not even be a key, mirroring
    # elo_loss_weight_strength's precedent in this same file. This holds
    # even when the batch carries real (non-empty) policy-KL rollout data.
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        policy_kl_weight=0.0,
        policy_kl_sigma=1.0,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()
    max_arms = 4

    arm_ids = torch.zeros((total_tokens, max_arms), dtype=torch.long)
    arm_ids[1, 0] = 10
    arm_qhat = torch.zeros((total_tokens, max_arms), dtype=torch.float32)
    arm_qhat[1, 0] = 0.4
    arm_mask = torch.zeros((total_tokens, max_arms), dtype=torch.bool)
    arm_mask[1, 0] = True
    has_rollout_policy_target = torch.zeros(total_tokens, dtype=torch.bool)
    has_rollout_policy_target[1] = True

    batch["policy_kl_arm_ids"] = arm_ids
    batch["policy_kl_arm_qhat"] = arm_qhat
    batch["policy_kl_arm_mask"] = arm_mask
    batch["has_rollout_policy_target"] = has_rollout_policy_target

    out = model(batch, return_loss=True)
    assert "policy_kl_loss" not in out


def test_hstu_chess_model_policy_kl_weight_zero_matches_no_rollout_keys():
    # Backward-compat invariant: policy_kl_weight=0.0 (the default) must
    # produce byte-identical total_loss/gradients to a batch that never had
    # the policy-KL keys at all -- mirrors the existing beta=0.0 invariant
    # (test_hstu_chess_model_value_loss_beta_zero_soft_target_matches_hard_ce).
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
    )
    assert config.policy_kl_weight == 0.0

    torch.manual_seed(0)
    model_baseline = HSTUChessModel(config)
    batch_baseline = _batch()

    torch.manual_seed(0)
    model_with_keys = HSTUChessModel(config)
    batch_with_keys = _batch()
    total_tokens = batch_with_keys["seq_token_id"].numel()
    max_arms = 4
    batch_with_keys["policy_kl_arm_ids"] = torch.zeros(
        (total_tokens, max_arms), dtype=torch.long
    )
    batch_with_keys["policy_kl_arm_ids"][1, 0] = 10
    batch_with_keys["policy_kl_arm_qhat"] = torch.zeros(
        (total_tokens, max_arms), dtype=torch.float32
    )
    batch_with_keys["policy_kl_arm_qhat"][1, 0] = 0.7
    batch_with_keys["policy_kl_arm_mask"] = torch.zeros(
        (total_tokens, max_arms), dtype=torch.bool
    )
    batch_with_keys["policy_kl_arm_mask"][1, 0] = True
    batch_with_keys["has_rollout_policy_target"] = torch.zeros(
        total_tokens, dtype=torch.bool
    )
    batch_with_keys["has_rollout_policy_target"][1] = True

    torch.manual_seed(1)
    out_baseline = model_baseline(batch_baseline, return_loss=True)
    torch.manual_seed(1)
    out_with_keys = model_with_keys(batch_with_keys, return_loss=True)

    assert torch.allclose(out_baseline["loss"], out_with_keys["loss"], atol=1e-12)


def test_build_hstu_chess_config_threads_policy_kl_fields():
    model_cfg = build_hstu_chess_config(
        ModelConfig(), move_vocab_size=64, policy_kl_weight=0.25, policy_kl_sigma=2.0
    )
    assert model_cfg.policy_kl_weight == 0.25
    assert model_cfg.policy_kl_sigma == 2.0


def test_build_hstu_chess_config_policy_kl_defaults_are_off():
    model_cfg = build_hstu_chess_config(ModelConfig(), move_vocab_size=64)
    assert model_cfg.policy_kl_weight == 0.0
    assert model_cfg.policy_kl_sigma == 1.0
