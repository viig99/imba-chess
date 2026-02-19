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
            batch["castle_id"].to(
                device=device, dtype=torch.long, non_blocking=True
            ),
            self.castle_embedding.num_embeddings,
        )
        ep_file_id = self._clamp_ids(
            batch["ep_file_id"].to(
                device=device, dtype=torch.long, non_blocking=True
            ),
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
        logits = self.prediction_head(x)
        output: dict[str, torch.Tensor] = {"logits": logits}

        if return_loss:
            if "target_move_id" not in batch:
                raise KeyError("batch['target_move_id'] is required when return_loss=True")
            target_move_id = batch["target_move_id"].to(
                device=logits.device, dtype=torch.long, non_blocking=True
            )
            has_valid_target = (target_move_id != self.config.ignore_index).any()
            torch._assert(
                has_valid_target,
                "No valid target tokens in batch (all target_move_id == ignore_index). "
                "Check dataset/event construction and sequence truncation settings.",
            )
            output["loss"] = F.cross_entropy(
                logits.float(),
                target_move_id,
                ignore_index=self.config.ignore_index,
            )

        return output
