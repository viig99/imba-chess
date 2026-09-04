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
        "value_target": torch.zeros((total_tokens, 3), dtype=torch.float32),
        "has_value_target": torch.zeros(total_tokens, dtype=torch.bool),
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


def _value_config(**overrides):
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
    )
    kwargs.update(overrides)
    return HSTUChessConfig(**kwargs)


def test_hstu_chess_model_value_loss_is_mean_soft_ce_over_target_tokens():
    # Plain mean of soft CE over tokens that carry a value target, whatever
    # their game progress or the mover's Elo.
    config = _value_config(elo_loss_weight_strength=1.0)
    model = HSTUChessModel(config)
    batch = _batch()
    batch["value_target"][1] = torch.tensor([0.3, 0.0, 0.7])
    batch["value_target"][4] = torch.tensor([0.9, 0.0, 0.1])
    batch["has_value_target"][1] = True
    batch["has_value_target"][4] = True

    out = model(batch, return_loss=True)
    soft_loss = -(batch["value_target"] * F.log_softmax(out["value_logits"].float(), dim=-1)).sum(dim=-1)
    expected = soft_loss[batch["has_value_target"]].mean()
    assert torch.allclose(out["value_loss"], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["value_weight_sum"], torch.tensor(2.0))
    assert torch.allclose(
        out["loss"],
        out["policy_loss"] + config.value_loss_weight * out["value_loss"]
        + config.moves_left_loss_weight * out["moves_left_loss"],
        atol=1e-6,
    )


def test_hstu_chess_model_value_target_on_ignored_token_is_masked():
    # A value target on the BOS token (target_move_id == ignore_index) must
    # not train the head.
    model = HSTUChessModel(_value_config())
    batch = _batch()
    batch["value_target"][0] = torch.tensor([0.0, 0.0, 1.0])
    batch["has_value_target"][0] = True

    out = model(batch, return_loss=True)
    assert float(out["value_weight_sum"]) == 0.0
    assert float(out["value_loss"]) == 0.0


def test_hstu_chess_model_value_weight_sum_is_zero_when_no_value_tokens():
    model = HSTUChessModel(_value_config())
    out = model(_batch(), return_loss=True)
    assert float(out["value_weight_sum"]) == 0.0
    assert float(out["value_loss"]) == 0.0
    assert torch.isfinite(out["loss"])


def test_hstu_chess_model_value_loss_gradient_flows_only_from_target_tokens():
    model = HSTUChessModel(_value_config())
    batch = _batch()
    batch["value_target"][2] = torch.tensor([0.2, 0.0, 0.8])
    batch["has_value_target"][2] = True
    out = model(batch, return_loss=True)
    (grad,) = torch.autograd.grad(out["value_loss"], out["value_logits"])
    assert grad[2].abs().sum() > 0
    assert grad[[0, 1, 3, 4]].abs().sum() == 0


def test_value_head_default_keeps_legacy_module_names():
    # blocks=0 must stay module-for-module identical to the pre-deep-head
    # shape, or ckpt23/ckpt27 stop loading under strict=True.
    model = HSTUChessModel(_value_config())
    keys = sorted(k for k in model.state_dict() if k.startswith("value_head"))
    assert keys == [
        "value_head.0.bias",
        "value_head.0.weight",
        "value_head.2.bias",
        "value_head.2.weight",
    ]
    assert model.state_dict()["value_head.0.weight"].shape == (32, 64)


def test_value_block_starts_as_identity_but_is_trainable():
    from imba_chess.model.hstu_model import _ValueBlock

    block = _ValueBlock(dim=8, expansion=2)
    x = torch.randn(4, 8)
    assert torch.allclose(block(x), x)

    block(x).sum().backward()
    # The zero-init projection must still receive gradient, or the block is
    # permanently dead rather than merely starting as the identity.
    assert block.mlp[2].weight.grad.abs().sum() > 0


def test_deep_value_head_shape_and_training_step():
    config = _value_config(
        value_head_width=32, value_head_blocks=2, value_head_expansion=2
    )
    model = HSTUChessModel(config)
    names = [k for k in model.state_dict() if k.startswith("value_head")]
    assert "value_head.1.mlp.0.weight" in names
    assert "value_head.2.norm.weight" in names
    # Input projection, 2 blocks, final norm, output projection.
    assert model.value_head[-1].out_features == 3

    batch = _batch()
    batch["value_target"][1] = torch.tensor([0.3, 0.0, 0.7])
    batch["has_value_target"][1] = True
    out = model(batch, return_loss=True)
    assert out["value_logits"].shape == (5, 3)
    assert torch.isfinite(out["value_loss"])
    out["loss"].backward()
    assert model.value_head[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"value_head_width": 0}, "value_head_width"),
        ({"value_head_blocks": -1}, "value_head_blocks"),
        ({"value_head_expansion": 0}, "value_head_expansion"),
    ],
)
def test_value_head_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        HSTUChessModel(_value_config(**overrides))
