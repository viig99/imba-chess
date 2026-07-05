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
    moves_left_loss_weight: float = 0.05


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
        moves_left_loss_weight=float(model_config.moves_left_loss_weight),
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


# 64-dim keeps the encoder at ~2/3 of the trunk's per-token FLOPs; 128-dim
# quadruples it and roughly triples the training step.
_BOARD_ENCODER_DIM = 64
_BOARD_ENCODER_HEADS = 4
_BOARD_ENCODER_LAYERS = 2


class _SquareAttentionBlock(nn.Module):
    def __init__(self, *, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boards, squares, dim = x.shape
        qkv = self.qkv(self.attn_norm(x))
        qkv = qkv.view(boards, squares, 3, self.num_heads, dim // self.num_heads)
        # permute+unbind leaves q/k/v as non-contiguous views sharing one
        # buffer, which caused a stride mismatch between torch.compile's
        # fake kernel and the real one; materializing avoids that (cheap at
        # this size: 64 squares).
        q, k, v = (t.contiguous() for t in qkv.permute(2, 0, 3, 1, 4).unbind(0))
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(boards, squares, dim)
        x = x + self.attn_out(attn)
        return x + self.mlp(self.mlp_norm(x))


class BoardSquareEncoder(nn.Module):
    """Bidirectional attention over the 64 squares of each position.

    Mean-pooling (piece, square) vectors is a linear aggregation: no square
    conditions on any other before the board collapses to one vector, so
    square interactions (attacks, pins, pawn structure) have to be recovered
    statistically by the trunk. A couple of attention layers over the squares
    let the board vector carry those interactions directly.
    """

    def __init__(
        self, *, dim: int, num_heads: int, num_layers: int, out_dim: int
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _SquareAttentionBlock(dim=dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        # Not checkpointed: under torch.compile, checkpointing a block that
        # contains SDPA makes AOT-autograd wrap the kernel in
        # graphsafe_run_with_rng_state (to keep RNG state consistent across
        # the backward recompute) — that wrapped op hits a CUDA "invalid
        # argument" during the flash-attention backward. Eager checkpointing
        # is fine; it's specifically the compiled-recompute path that's broken.
        for block in self.blocks:
            squares = block(squares)
        return self.out_proj(self.final_norm(squares).mean(dim=1))


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
        if float(config.moves_left_loss_weight) < 0.0:
            raise ValueError("moves_left_loss_weight must be >= 0")
        d = config.model_dim

        # Joint (piece, square) table: an additive piece+square scheme collapses
        # under mean pooling to a bag of material (the square term is constant),
        # making piece placement invisible to the model.
        self.piece_square_embedding = nn.Embedding(13 * 64, _BOARD_ENCODER_DIM)
        self.board_encoder = BoardSquareEncoder(
            dim=_BOARD_ENCODER_DIM,
            num_heads=_BOARD_ENCODER_HEADS,
            num_layers=_BOARD_ENCODER_LAYERS,
            out_dim=d,
        )
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
        # Same move vocab on the input and output side; sharing the matrix
        # saves ~1M params and regularizes both representations.
        self.prediction_head.weight = self.prev_move_embedding.weight
        # Small private MLP: trunk features are dominated by the policy
        # objective, so the value head needs its own capacity.
        self.value_head = (
            nn.Sequential(
                nn.Linear(d, d // 2),
                nn.SiLU(),
                nn.Linear(d // 2, 3),
            )
            if config.enable_value_head
            else None
        )
        # Auxiliary target: predicting log(plies remaining) forces the trunk
        # to represent how close the game is to being decided — a feature the
        # value head needs but the policy objective never asks for. The head's
        # output is unused at inference.
        self.moves_left_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )

        self.register_buffer(
            "square_ids", torch.arange(64, dtype=torch.long), persistent=False
        )

    def _embed_board(self, piece_ids: torch.Tensor) -> torch.Tensor:
        # piece_ids: [S, 64] -> unique id per (piece, square) pair.
        pair_ids = piece_ids * 64 + self.square_ids
        return self.board_encoder(self.piece_square_embedding(pair_ids))

    def _clamp_ids(self, ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        return ids.clamp(min=0, max=num_embeddings - 1)

    def _build_content(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.piece_square_embedding.weight.device
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
        return_kv: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = self.piece_square_embedding.weight.device
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

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            if return_kv:
                x, layer_kv = layer(x=x, block_mask=block_mask, return_kv=True)
                kv_caches.append(layer_kv)
            else:
                x = layer(x=x, block_mask=block_mask)

        x = self.final_norm(x)
        policy_logits = self.prediction_head(x)
        output: dict[str, torch.Tensor] = {
            "logits": policy_logits,
            "policy_logits": policy_logits,
        }
        if return_kv:
            output["kv_caches"] = kv_caches  # type: ignore[assignment]
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

            total_loss = policy_loss

            counts = seq_offsets[1:] - seq_offsets[:-1]
            batch_games = int(counts.numel())
            token_game_id = torch.repeat_interleave(
                torch.arange(batch_games, device=policy_logits.device),
                counts,
            )
            token_pos_in_game = torch.arange(
                policy_logits.shape[0], device=policy_logits.device
            ) - seq_offsets.index_select(0, token_game_id)
            seq_len_for_token = counts.index_select(0, token_game_id).clamp_min(1)

            if value_logits is not None:
                game_result_white = batch["game_result_white"].to(
                    device=policy_logits.device, dtype=torch.long, non_blocking=True
                )
                if game_result_white.ndim != 1 or int(game_result_white.shape[0]) != batch_games:
                    raise ValueError(
                        "game_result_white must have shape [B] where B == num_games"
                    )
                z_token = game_result_white.index_select(0, token_game_id)
                turn_id = batch["turn_id"].to(
                    device=policy_logits.device, dtype=torch.long, non_blocking=True
                )
                y = torch.where(turn_id == 0, z_token, -z_token)
                value_target = (y + 1).clamp(min=0, max=2)

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
                total_loss = total_loss + self.config.value_loss_weight * value_loss

            # log1p compresses the target so errors near the end of the game
            # (where decidedness is informative) dominate errors at move 10.
            plies_left = (seq_len_for_token - 1 - token_pos_in_game).clamp_min(0)
            moves_left_target = torch.log1p(plies_left.to(torch.float32))
            moves_left_pred = self.moves_left_head(x).squeeze(-1).float()
            output["moves_left_pred"] = moves_left_pred
            per_token_moves_left_loss = F.huber_loss(
                moves_left_pred, moves_left_target, reduction="none"
            )
            moves_left_weights = valid_mask.to(per_token_moves_left_loss.dtype)
            moves_left_loss = (
                per_token_moves_left_loss * moves_left_weights
            ).sum() / moves_left_weights.sum().clamp_min(1.0)
            output["moves_left_loss"] = moves_left_loss
            total_loss = (
                total_loss + self.config.moves_left_loss_weight * moves_left_loss
            )

            output["loss"] = total_loss

        return output

    def forward_decode(
        self,
        *,
        new_token_batch: dict[str, Any],
        positions: torch.Tensor,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]],
        suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        suffix_positions: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Decode one new token per batch row against per-layer cached K/V.

        Inference-only companion to forward(return_kv=True): new_token_batch
        carries the per-token id tensors _build_content reads; positions are
        absolute (prefix_len + suffix depth). Returns logits/value_logits at
        the new tokens plus each layer's (k, v) for growing suffix caches.
        """
        assert not self.training, "forward_decode is inference-only"
        device = self.piece_square_embedding.weight.device
        positions = positions.to(device=device, dtype=torch.long)
        content = self._build_content(new_token_batch)
        x = self.position_embedding.at_positions(content, positions)

        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.layers):
            prefix_k, prefix_v = prefix_kv[layer_idx]
            if suffix_kv is not None:
                layer_suffix_k, layer_suffix_v = suffix_kv[layer_idx]
            else:
                layer_suffix_k = layer_suffix_v = None
            x, k_new, v_new = layer.forward_decode(
                x,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                q_positions=positions,
                suffix_k=layer_suffix_k,
                suffix_v=layer_suffix_v,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
            new_kv.append((k_new, v_new))

        x = self.final_norm(x)
        output: dict[str, Any] = {
            "logits": self.prediction_head(x),
            "kv": new_kv,
        }
        if self.value_head is not None:
            output["value_logits"] = self.value_head(x)
        return output
