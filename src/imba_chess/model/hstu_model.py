from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_gym.masks import generate_doc_mask_mod, generate_prefix_lm_mask
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .hstu_attention import (
    SequentialTransductionUnitJagged,
    build_grouped_decode_cache,
)
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
    # Value readout shape. blocks=0 is the original single-hidden-layer MLP
    # and is kept as the default so checkpoints trained before the deep head
    # still load under strict=True. width defaults to model_dim // 2.
    value_head_width: int | None = None
    value_head_blocks: int = 0
    value_head_expansion: int = 2
    moves_left_loss_weight: float = 0.05


def build_hstu_chess_config(
    model_config: Any,
    *,
    move_vocab_size: int,
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
        value_head_width=(
            None
            if model_config.value_head_width is None
            else int(model_config.value_head_width)
        ),
        value_head_blocks=int(model_config.value_head_blocks),
        value_head_expansion=int(model_config.value_head_expansion),
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


def create_batch_dense_mask(
    seq_offsets: torch.Tensor,
    *,
    total_tokens: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """[S, S] bool: token q attends to k iff same document and k <= q.

    Admits exactly the positions create_batch_block_mask admits, materialized
    instead of block-sparse, so attention can run as one fused SDPA call.
    Eager flex_attention costs ~95 ms per model forward regardless of length
    (pure dispatch overhead); this is 5-38x faster. See
    docs/superpowers/notes/2026-08-20-rl-throughput-bottleneck.md.

    Choose by BATCH SHAPE, not by train-vs-eval -- the per-layer additive mask
    the attention layer derives from this is [1, H, S, S], quadratic in S:

      * dense (this): play/search batches -- a few sequences, a few hundred
        tokens. ~12 MiB at S=512, H=12.
      * BlockMask: dataset-sized batches -- ~4,000 tokens over ~50 games
        (~805 MiB dense, and BlockMask skips the ~98% of the block grid that
        the block-diagonal document structure masks out anyway). This is
        training AND dataset evaluation: ignite_evaluator, eval_value_loss,
        eval_value_by_progress all correctly stay on BlockMask.

    Exceeding the dense budget raises rather than allocating -- see
    SequentialTransductionUnitJagged._additive_mask.
    """
    if total_tokens is None:
        total_tokens = int(seq_offsets[-1].item())
    dev = device if device is not None else seq_offsets.device
    pos = torch.arange(total_tokens, device=dev)
    # Document id per token = how many document starts lie at or before it.
    starts = seq_offsets[1:-1].to(dev)
    if starts.numel():
        doc = (pos.unsqueeze(1) >= starts.unsqueeze(0)).sum(dim=1)
    else:
        doc = torch.zeros(total_tokens, dtype=torch.long, device=dev)
    causal = pos.unsqueeze(1) >= pos.unsqueeze(0)
    return causal & (doc.unsqueeze(1) == doc.unsqueeze(0))


# 64-dim keeps the encoder at ~2/3 of the trunk's per-token FLOPs; 128-dim
# quadruples it and roughly triples the training step.
_BOARD_ENCODER_DIM = 64
_BOARD_ENCODER_HEADS = 4
_BOARD_ENCODER_LAYERS = 2


class _ValueBlock(nn.Module):
    """Pre-norm residual MLP block for the value head.

    The output projection is zero-initialised, so a freshly built deep head
    starts out as exactly the plain linear readout it replaces. Without that,
    a deep head pushes a large and meaningless gradient into the trunk on
    step 0, before the trunk has any features worth reading.
    """

    def __init__(self, *, dim: int, expansion: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, expansion * dim),
            nn.SiLU(),
            nn.Linear(expansion * dim, dim),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


def _build_value_head(
    *, dim: int, width: int | None, blocks: int, expansion: int
) -> nn.Sequential:
    """Value readout: `dim` trunk features -> 3 WDL logits.

    blocks=0 reproduces the original head module-for-module (same indices,
    same parameter names), so pre-deep-head checkpoints load unchanged.
    """
    hidden = dim // 2 if width is None else int(width)
    if blocks == 0:
        return nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
    return nn.Sequential(
        nn.Linear(dim, hidden),
        *(_ValueBlock(dim=hidden, expansion=expansion) for _ in range(blocks)),
        # Pre-norm blocks leave the residual stream unnormalised; this is
        # required, not cosmetic.
        nn.LayerNorm(hidden),
        nn.Linear(hidden, 3),
    )


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
        if config.value_head_width is not None and int(config.value_head_width) < 1:
            raise ValueError("value_head_width must be >= 1 when set")
        if int(config.value_head_blocks) < 0:
            raise ValueError("value_head_blocks must be >= 0")
        if int(config.value_head_expansion) < 1:
            raise ValueError("value_head_expansion must be >= 1")
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
            _build_value_head(
                dim=d,
                width=config.value_head_width,
                blocks=int(config.value_head_blocks),
                expansion=int(config.value_head_expansion),
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
        block_mask: BlockMask | torch.Tensor | None = None,
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
            if x.device.type == "cpu":
                # torch 2.13 raises NotImplementedError("FlexAttention does not
                # support backward on CPU"), so CPU cannot go through
                # create_batch_block_mask at all once anything requires grad.
                # The dense mask admits exactly the same positions and runs as
                # one SDPA call, which does support CPU backward. CPU runs are
                # tests and small-batch debugging; the [1, H, S, S] budget in
                # _additive_mask still raises loudly rather than allocating if a
                # dataset-sized batch ever lands here.
                block_mask = create_batch_dense_mask(
                    seq_offsets=seq_offsets,
                    total_tokens=int(batch["total_tokens"]),
                    device=x.device,
                )
            else:
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
            # Shared Elo weighting: stronger players' tokens pull harder on
            # both losses — their moves are better policy targets, and their
            # game outcomes are lower-noise value labels (better conversion).
            elo_scale: torch.Tensor | None = None
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

            policy_token_weights = valid_mask.to(per_token_policy_loss.dtype)
            if elo_scale is not None:
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
                # Side-to-move WDL target from the ply's Lichess eval
                # (data/stockfish_evals.winpercent_wdl). Only plies that carry
                # an eval train the head; everything else has weight 0. No
                # data-dependent Python branching: torch.compile(fullgraph)
                # traces the static key structure only.
                value_target = batch["value_target"].to(
                    device=policy_logits.device, dtype=torch.float32, non_blocking=True
                )
                has_value_target = batch["has_value_target"].to(
                    device=policy_logits.device, dtype=torch.bool, non_blocking=True
                )
                per_token_value_loss = -(
                    value_target * F.log_softmax(value_logits.float(), dim=-1)
                ).sum(dim=-1)
                value_weights = (has_value_target & valid_mask).to(torch.float32)
                value_loss_sum = (per_token_value_loss * value_weights).sum()
                raw_value_weight_sum = value_weights.sum()
                value_loss = value_loss_sum / raw_value_weight_sum.clamp_min(1.0)
                output["value_loss"] = value_loss
                # The UNclamped count: an evaluator pooling batch means by it
                # must see 0 for a batch with no value tokens, not a phantom
                # weight-1 zero-loss contribution.
                output["value_weight_sum"] = raw_value_weight_sum
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

    def forward_decode_grouped(
        self,
        *,
        new_token_batch: dict[str, Any],
        positions: torch.Tensor,
        group_index: torch.Tensor,
        prefix_kv_grouped: list[tuple[torch.Tensor, torch.Tensor]],
        prefix_lens: torch.Tensor,
        prefix_lens_list: list[int],
        suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        suffix_positions: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
        group_sizes: list[int] | None = None,
    ) -> dict[str, Any]:
        """Decode one new token per batch row against per-game grouped
        prefix K/V caches (cross-game merged wave).

        `group_sizes` is an optional host-side promise that the rows are
        already laid out contiguously by group -- group g owning
        [sum(group_sizes[:g]), +group_sizes[g]). When given, `group_index`
        is never read: the bounds check and the per-group row indices both
        become Python integer arithmetic, removing G+1 device->host syncs
        per wave from the critical path ahead of the layer loop. Omit it and
        the arbitrary-order `nonzero` path is used instead, which is what
        an interleaved group_index needs.

        Companion to forward_decode for the multi-game batched search
        executor: rows from up to G different games share one call, each
        row reading only its own game's prefix via `group_index` [B] -> g.
        prefix_kv_grouped is per-layer (k, v) with shape [G, H, maxP, d]
        (games zero-padded on the token dim to the batch's longest prefix);
        prefix_lens [G] gives each game's real (unpadded) length, used only
        for this function's own boundary validation below (num_groups,
        group_index range). prefix_lens_list is the same G values as a
        plain host-side list[int] -- passed straight through to each
        layer's grouped decode, which indexes it once per group per layer;
        threading the already-known Python ints instead of the device
        tensor avoids a device->host sync there (see
        SequentialTransductionUnitJagged.forward_decode_grouped's
        docstring). Suffix args are unchanged from forward_decode --
        already per-row.

        Single-prefix forward_decode above is untouched and remains the
        G=1 / eval path; this is purely additive.

        Returns the same dict shape as forward_decode: logits/value_logits
        at the new tokens plus each layer's (k, v) for growing suffix
        caches, in original row order.
        """
        assert not self.training, "forward_decode_grouped is inference-only"
        device = self.piece_square_embedding.weight.device
        positions = positions.to(device=device, dtype=torch.long)
        if group_sizes is None:
            group_index = group_index.to(device=device, dtype=torch.long)
        prefix_lens = prefix_lens.to(device=device, dtype=torch.long)
        content = self._build_content(new_token_batch)
        batch_size = int(content.shape[0])
        num_groups = int(prefix_lens.numel())
        if len(prefix_lens_list) != num_groups:
            raise ValueError(
                "prefix_lens_list must have length num_groups "
                f"(== prefix_lens.numel() == {num_groups}), got "
                f"{len(prefix_lens_list)}"
            )
        # Every attn_output row is written by whichever group claims it (see
        # forward_decode_grouped in hstu_attention.py); a row whose
        # group_index falls outside [0, num_groups) would never be claimed
        # and silently carry uninitialized memory into logits/value_logits.
        # Validated here once, at the model boundary, rather than per-layer.
        if int(group_index.numel()) != batch_size:
            raise ValueError(
                "group_index must have shape [B] matching new_token_batch"
            )
        if group_sizes is not None:
            # Contiguous-groups fast path. The caller asserts that row i
            # belongs to the unique group g with
            # offset_g <= i < offset_g + group_sizes[g] -- which is exactly
            # how _merge_decode_requests lays a wave out (it builds
            # group_index by concatenating one torch.full per request, so a
            # game's rows are adjacent by construction). Under that promise
            # the same "no row goes unclaimed" property the check below
            # enforces follows from sum(group_sizes) == batch_size, and it
            # costs no device round trip.
            if len(group_sizes) != num_groups:
                raise ValueError(
                    "group_sizes must have length num_groups "
                    f"(== {num_groups}), got {len(group_sizes)}"
                )
            if any(n < 0 for n in group_sizes):
                raise ValueError("group_sizes entries must be non-negative")
            if sum(group_sizes) != batch_size:
                raise ValueError(
                    "group_sizes must sum to the batch size "
                    f"({batch_size}), got {sum(group_sizes)}"
                )
        else:
            # No `num_groups > 0` short-circuit here: a non-empty batch with
            # zero groups (empty prefix_lens) must still raise below rather
            # than let every row skip the per-group loop and return
            # uninitialized (torch.empty) attn_output as silent NaN logits.
            # With num_groups==0 the comparison `group_index >= 0` is already
            # sufficient to catch any row (since no g in range(0) will ever
            # claim it), and .any() on an empty group_index (batch_size==0)
            # is correctly False.
            if bool(((group_index < 0) | (group_index >= num_groups)).any()):
                raise ValueError("group_index values must be in [0, num_groups)")
        x = self.position_embedding.at_positions(content, positions)

        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Row indices per group depend only on group_index, which is fixed for
        # the whole wave. Each layer used to re-derive them, so a G=8 / 8-layer
        # wave ran 64 `nonzero` calls where 8 suffice -- and since nonzero's
        # output shape is data-dependent, each was a device->host sync too. It
        # measured as the largest single torch op in a rollout profile.
        if group_sizes is not None:
            # ...and when the caller names the group boundaries, the G
            # surviving `nonzero` syncs go away too: each group's rows are
            # the contiguous span [offset, offset + n).
            row_idx_per_group = []
            offset = 0
            for n in group_sizes:
                row_idx_per_group.append(
                    torch.arange(offset, offset + n, device=device, dtype=torch.long)
                )
                offset += n
        else:
            row_idx_per_group = [
                (group_index == g).nonzero(as_tuple=True)[0] for g in range(num_groups)
            ]

        # The rest of the per-group decode scratch is layer-invariant as
        # well, so build it once here instead of num_layers times inside the
        # loop below. It needs the layers' shared _max_seq_len: they are all
        # constructed from config.max_position_embeddings, but a layer that
        # disagreed would silently receive the wrong relative bias, so check
        # rather than assume.
        max_seq_lens = {int(layer._max_seq_len) for layer in self.layers}
        if len(max_seq_lens) != 1:
            raise ValueError(
                "layers must share a single _max_seq_len for the per-wave "
                f"decode cache; got {sorted(max_seq_lens)}"
            )
        decode_cache = build_grouped_decode_cache(
            row_idx_per_group=row_idx_per_group,
            prefix_lens_list=prefix_lens_list,
            q_positions=positions,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
            max_seq_len=next(iter(max_seq_lens)),
        )

        for layer_idx, layer in enumerate(self.layers):
            prefix_k, prefix_v = prefix_kv_grouped[layer_idx]
            if suffix_kv is not None:
                layer_suffix_k, layer_suffix_v = suffix_kv[layer_idx]
            else:
                layer_suffix_k = layer_suffix_v = None
            x, k_new, v_new = layer.forward_decode_grouped(
                x,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                prefix_lens_list=prefix_lens_list,
                group_index=group_index,
                row_idx_per_group=row_idx_per_group,
                q_positions=positions,
                suffix_k=layer_suffix_k,
                suffix_v=layer_suffix_v,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
                decode_cache=decode_cache,
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
