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
    assert torch.allclose(out["loss"], expected, atol=1e-6, rtol=1e-6)


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
    assert torch.allclose(out["loss"], expected, atol=1e-6, rtol=1e-6)


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
