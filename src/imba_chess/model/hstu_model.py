from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_gym.masks import generate_doc_mask_mod, generate_prefix_lm_mask
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .hstu_attention import SequentialTransductionUnitJagged
from .position_embedding import PositionEmbedding

_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)


@dataclass(frozen=True)
class HSTUChessConfig:
    move_vocab_size: int
    model_dim: int = 384
    linear_hidden_dim: int = 128
    attention_dim: int = 128
    num_heads: int = 4
    num_layers: int = 6
    dropout: float = 0.1
    max_position_embeddings: int = 6144
    halfmove_vocab_size: int = 128
    fullmove_vocab_size: int = 128
    ignore_index: int = -100
    relative_attention_bias: str = "position"
    label_smoothing: float = 0.0
    elo_weight_min_elo: int = 2200
    elo_weight_max_elo: int = 2800
    elo_loss_weight_alpha: float = 1.0
    elo_loss_weight_strength: float = 0.0
    enable_value_head: bool = False
    value_loss_weight: float = 0.15
    value_weight_alpha: float = 1.5
    value_label_smoothing: float = 0.0


def build_hstu_chess_config(
    model_config: Any, *, move_vocab_size: int
) -> HSTUChessConfig:
    """Create model config from repo config model section + runtime vocab size."""
    return HSTUChessConfig(
        move_vocab_size=move_vocab_size,
        model_dim=int(model_config.model_dim),
        linear_hidden_dim=int(model_config.linear_hidden_dim),
        attention_dim=int(model_config.attention_dim),
        num_heads=int(model_config.num_heads),
        num_layers=int(model_config.num_layers),
        dropout=float(model_config.dropout),
        max_position_embeddings=int(model_config.max_position_embeddings),
        halfmove_vocab_size=int(model_config.halfmove_vocab_size),
        fullmove_vocab_size=int(model_config.fullmove_vocab_size),
        ignore_index=int(model_config.ignore_index),
        relative_attention_bias=str(model_config.relative_attention_bias),
        label_smoothing=float(model_config.label_smoothing),
        elo_weight_min_elo=int(model_config.elo_weight_min_elo),
        elo_weight_max_elo=int(model_config.elo_weight_max_elo),
        elo_loss_weight_alpha=float(model_config.elo_loss_weight_alpha),
        elo_loss_weight_strength=float(model_config.elo_loss_weight_strength),
        enable_value_head=bool(model_config.enable_value_head),
        value_loss_weight=float(model_config.value_loss_weight),
        value_weight_alpha=float(model_config.value_weight_alpha),
        value_label_smoothing=float(model_config.value_label_smoothing),
    )


def create_batch_block_mask(
    seq_offsets: torch.Tensor,
    *,
    total_tokens: int | None = None,
    device: str | torch.device | None = None,
) -> BlockMask:
    if total_tokens is None:
        total_tokens = int(seq_offsets[-1].item())

    prefix_causal_mask = generate_prefix_lm_mask(0)
    doc_prefix_causal_mask = generate_doc_mask_mod(prefix_causal_mask, seq_offsets)
    return _compiled_create_block_mask(
        doc_prefix_causal_mask,
        B=1,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
    )


class HSTUChessModel(nn.Module):
    """HSTU backbone for jagged chess event batches."""

    def __init__(self, config: HSTUChessConfig) -> None:
        super().__init__()
        self.config = config
        if not 0.0 <= float(config.label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0)")
        if int(config.elo_weight_max_elo) <= int(config.elo_weight_min_elo):
            raise ValueError("elo_weight_max_elo must be > elo_weight_min_elo")
        if float(config.elo_loss_weight_alpha) <= 0.0:
            raise ValueError("elo_loss_weight_alpha must be > 0")
        if float(config.elo_loss_weight_strength) < 0.0:
            raise ValueError("elo_loss_weight_strength must be >= 0")
        if float(config.value_loss_weight) < 0.0:
            raise ValueError("value_loss_weight must be >= 0")
        if float(config.value_weight_alpha) <= 0.0:
            raise ValueError("value_weight_alpha must be > 0")
        if not 0.0 <= float(config.value_label_smoothing) < 1.0:
            raise ValueError("value_label_smoothing must be in [0.0, 1.0)")
        d = config.model_dim

        self.piece_embedding = nn.Embedding(13, d)
        self.square_embedding = nn.Embedding(64, d)
        self.seq_token_embedding = nn.Embedding(2, d)
        self.turn_embedding = nn.Embedding(2, d)
        self.castle_embedding = nn.Embedding(16, d)
        self.ep_embedding = nn.Embedding(9, d)
        self.halfmove_embedding = nn.Embedding(config.halfmove_vocab_size, d)
        self.fullmove_embedding = nn.Embedding(config.fullmove_vocab_size, d)
        self.prev_move_embedding = nn.Embedding(config.move_vocab_size, d)

        self.position_embedding = PositionEmbedding(
            max_seq_len=config.max_position_embeddings,
            embedding_dim=d,
            dropout_rate=config.dropout,
        )

        self.layers = nn.ModuleList(
            [
                SequentialTransductionUnitJagged(
                    embedding_dim=d,
                    linear_hidden_dim=config.linear_hidden_dim,
                    attention_dim=config.attention_dim,
                    dropout_ratio=config.dropout,
                    num_heads=config.num_heads,
                    max_seq_len=config.max_position_embeddings,
                    relative_attention_bias_module=config.relative_attention_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d)
        self.prediction_head = nn.Linear(d, config.move_vocab_size, bias=False)
        self.value_head = nn.Linear(d, 3) if config.enable_value_head else None

        self.register_buffer(
            "square_ids", torch.arange(64, dtype=torch.long), persistent=False
        )

    def _embed_board(self, piece_ids: torch.Tensor) -> torch.Tensor:
        # piece_ids: [S, 64]
        piece_emb = self.piece_embedding(piece_ids)
        square_emb = self.square_embedding(self.square_ids).unsqueeze(0)
        return (piece_emb + square_emb).mean(dim=1)

    def _clamp_ids(self, ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        return ids.clamp(min=0, max=num_embeddings - 1)

    def _build_content(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.piece_embedding.weight.device
        piece_ids = batch["piece_ids"].to(
            device=device, dtype=torch.long, non_blocking=True
        )
        seq_token_id = self._clamp_ids(
            batch["seq_token_id"].to(
                device=device, dtype=torch.long, non_blocking=True
            ),
            self.seq_token_embedding.num_embeddings,
        )
        turn_id = self._clamp_ids(
            batch["turn_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.turn_embedding.num_embeddings,
        )
        castle_id = self._clamp_ids(
            batch["castle_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.castle_embedding.num_embeddings,
        )
        ep_file_id = self._clamp_ids(
            batch["ep_file_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.ep_embedding.num_embeddings,
        )
        halfmove_bucket_id = self._clamp_ids(
            batch["halfmove_bucket_id"].to(
                device=device, dtype=torch.long, non_blocking=True
            ),
            self.halfmove_embedding.num_embeddings,
        )
        fullmove_bucket_id = self._clamp_ids(
            batch["fullmove_bucket_id"].to(
                device=device, dtype=torch.long, non_blocking=True
            ),
            self.fullmove_embedding.num_embeddings,
        )
        prev_move_id = self._clamp_ids(
            batch["prev_move_id"].to(
                device=device, dtype=torch.long, non_blocking=True
            ),
            self.prev_move_embedding.num_embeddings,
        )

        board = self._embed_board(piece_ids)
        return (
            board
            + self.seq_token_embedding(seq_token_id)
            + self.turn_embedding(turn_id)
            + self.castle_embedding(castle_id)
            + self.ep_embedding(ep_file_id)
            + self.halfmove_embedding(halfmove_bucket_id)
            + self.fullmove_embedding(fullmove_bucket_id)
            + self.prev_move_embedding(prev_move_id)
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        block_mask: BlockMask | None = None,
        return_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        device = self.piece_embedding.weight.device
        seq_offsets = batch["seq_offsets"].to(
            device=device, dtype=torch.long, non_blocking=True
        )
        content = self._build_content(batch)
        x = self.position_embedding(content, seq_offsets)

        if self.layers and block_mask is None:
            block_mask = create_batch_block_mask(
                seq_offsets=seq_offsets,
                total_tokens=int(batch["total_tokens"]),
                device=x.device,
            )

        for layer in self.layers:
            x = layer(x=x, block_mask=block_mask)

        x = self.final_norm(x)
        policy_logits = self.prediction_head(x)
        output: dict[str, torch.Tensor] = {
            "logits": policy_logits,
            "policy_logits": policy_logits,
        }
        value_logits: torch.Tensor | None = None
        if self.value_head is not None:
            value_logits = self.value_head(x)
            output["value_logits"] = value_logits

        if return_loss:
            target_move_id = batch["target_move_id"].to(
                device=policy_logits.device, dtype=torch.long, non_blocking=True
            )
            valid_mask = target_move_id != self.config.ignore_index
            safe_targets = target_move_id.masked_fill(~valid_mask, 0)
            per_token_policy_loss = F.cross_entropy(
                policy_logits.float(),
                safe_targets,
                reduction="none",
                label_smoothing=self.config.label_smoothing,
            )
            policy_token_weights = valid_mask.to(per_token_policy_loss.dtype)
            if self.config.elo_loss_weight_strength > 0.0:
                played_by_elo = batch["played_by_elo"].to(
                    device=policy_logits.device,
                    dtype=per_token_policy_loss.dtype,
                    non_blocking=True,
                )
                min_elo = self.config.elo_weight_min_elo
                max_elo = self.config.elo_weight_max_elo
                elo_norm = ((played_by_elo - min_elo) / (max_elo - min_elo)).clamp(
                    min=0.0, max=1.0
                )
                elo_curve = elo_norm.pow(self.config.elo_loss_weight_alpha)
                elo_scale = 1.0 + self.config.elo_loss_weight_strength * elo_curve
                policy_token_weights = policy_token_weights * elo_scale

            policy_loss_sum = (per_token_policy_loss * policy_token_weights).sum()
            policy_weight_sum = policy_token_weights.sum().clamp_min(1.0)
            policy_loss = policy_loss_sum / policy_weight_sum
            output["policy_loss"] = policy_loss

            if value_logits is not None:
                counts = seq_offsets[1:] - seq_offsets[:-1]
                batch_games = int(counts.numel())
                game_result_white = batch["game_result_white"].to(
                    device=policy_logits.device, dtype=torch.long, non_blocking=True
                )
                if game_result_white.ndim != 1 or int(game_result_white.shape[0]) != batch_games:
                    raise ValueError(
                        "game_result_white must have shape [B] where B == num_games"
                    )
                token_game_id = torch.repeat_interleave(
                    torch.arange(batch_games, device=policy_logits.device),
                    counts,
                )
                z_token = game_result_white.index_select(0, token_game_id)
                turn_id = batch["turn_id"].to(
                    device=policy_logits.device, dtype=torch.long, non_blocking=True
                )
                y = torch.where(turn_id == 0, z_token, -z_token)
                value_target = (y + 1).clamp(min=0, max=2)

                token_pos_in_game = torch.arange(
                    policy_logits.shape[0], device=policy_logits.device
                ) - seq_offsets.index_select(0, token_game_id)
                seq_len_for_token = counts.index_select(0, token_game_id).clamp_min(1)
                progress = token_pos_in_game.to(torch.float32) / (
                    seq_len_for_token.to(torch.float32) - 1.0
                ).clamp_min(1.0)
                value_weights = progress.pow(self.config.value_weight_alpha)
                value_weights = value_weights * valid_mask.to(value_weights.dtype)

                per_token_value_loss = F.cross_entropy(
                    value_logits.float(),
                    value_target,
                    reduction="none",
                    label_smoothing=self.config.value_label_smoothing,
                )
                value_loss_sum = (per_token_value_loss * value_weights).sum()
                value_weight_sum = value_weights.sum().clamp_min(1.0)
                value_loss = value_loss_sum / value_weight_sum
                output["value_loss"] = value_loss
                output["loss"] = policy_loss + (
                    self.config.value_loss_weight * value_loss
                )
            else:
                output["loss"] = policy_loss

        return output
