import torch
import torch.nn.functional as F
from typing import Literal
from torch.nn.attention.flex_attention import BlockMask, flex_attention


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        num_heads: int,
        max_seq_len: int = 2048,
        relative_attention_bias_module: Literal["position"] = "position",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim
        self._attention_dim = attention_dim
        self._dropout_ratio = dropout_ratio
        self._num_heads = num_heads
        self._rel_attn_bias = relative_attention_bias_module
        self._eps = epsilon
        self._max_seq_len = max_seq_len

        self._uvqk: torch.nn.Linear = torch.nn.Linear(
            embedding_dim,
            linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2,
            bias=False,
        )
        torch.nn.init.normal_(self._uvqk.weight, mean=0, std=0.02)

        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads,
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)

        # Per-head relative position bias (T5-style): heads can learn distinct
        # distance priors (e.g. previous-move vs long-range opening context).
        self._ps_w = torch.nn.Parameter(
            torch.empty(num_heads, 2 * self._max_seq_len - 1).normal_(
                mean=0, std=0.02
            ),
        )

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._num_heads * self._linear_dim], eps=self._eps
        )

    def _reshape_uvqk_for_mm(
        self, x: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        return x.unflatten(-1, (num_heads, head_dim)).transpose(1, 2).contiguous()

    def _position_score_mod(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
    ):
        idx = torch.clamp(
            (k_idx - q_idx) + (self._max_seq_len - 1), 0, 2 * self._max_seq_len - 2
        )
        score = score + self._ps_w[h, idx].to(score.dtype)
        return score

    def _generate_rab_score_mod(self):
        return self._position_score_mod

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask | None = None,
        return_kv: bool = False,
    ):
        # x: [S, D] — total tokens across all sessions
        S = x.size(0)
        x = x.unsqueeze(0)
        normed_x = self._norm_input(x)
        uvqk_x = self._uvqk(
            normed_x
        )  # shape: [1, S, linear_dim * 2 * num_heads + attention_dim * 2 * num_heads]
        uvqk_x = F.silu(uvqk_x)
        u, v, q, k = torch.split(
            uvqk_x,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )

        q_heads = self._reshape_uvqk_for_mm(q, self._num_heads, self._attention_dim)
        k_heads = self._reshape_uvqk_for_mm(k, self._num_heads, self._attention_dim)
        v_heads = self._reshape_uvqk_for_mm(v, self._num_heads, self._linear_dim)

        # output shape: [1, num_heads, S, linear_dim]
        attn_output: torch.Tensor = flex_attention(
            query=q_heads,
            key=k_heads,
            value=v_heads,
            block_mask=block_mask,
            score_mod=self._generate_rab_score_mod(),
            kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1},
        )  # type: ignore

        attn_output = self._norm_attn_output(
            attn_output.permute(0, 2, 1, 3).reshape(
                1, S, self._num_heads * self._linear_dim
            )
        )

        o_input = F.dropout(
            u * attn_output, p=self._dropout_ratio, training=self.training
        )
        out = (self._o(o_input) + x).squeeze(0)
        if return_kv:
            return out, (k_heads.squeeze(0), v_heads.squeeze(0))
        return out

    def _relative_bias(
        self, k_positions: torch.Tensor, q_positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-head relative bias for decode: [B, H, 1, K] from positions.

        k_positions: [B, K] (or [1, K] broadcastable), q_positions: [B].
        Replicates _position_score_mod's clamped (k_idx - q_idx) indexing.
        """
        rel = torch.clamp(
            k_positions - q_positions.view(-1, 1) + (self._max_seq_len - 1),
            0,
            2 * self._max_seq_len - 2,
        )  # [B, K]
        # _ps_w: [H, 2*max-1]; gather -> [H, B, K] -> [B, H, 1, K]
        return self._ps_w[:, rel].permute(1, 0, 2).unsqueeze(2)

    def forward_decode(
        self,
        x_new: torch.Tensor,
        *,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        q_positions: torch.Tensor,
        suffix_k: torch.Tensor | None = None,
        suffix_v: torch.Tensor | None = None,
        suffix_positions: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one new token per batch row against cached K/V.

        x_new: [B, D]; prefix_k/v: [H, T, d] shared across rows;
        suffix_k/v: [B, H, s, d] per-row ancestor tokens (zero-padded, with
        suffix_mask [B, s] marking real entries); q_positions/suffix_positions
        are absolute within each row's virtual sequence. The new token always
        attends to prefix + its real suffix + itself (causal <=), replicating
        forward()'s flex_attention semantics: scores scaled by
        1/sqrt(attention_dim) then biased by _ps_w, softmax over all keys.

        Returns (x_out [B, D], k_new [B, H, 1, d_qk], v_new [B, H, 1, d_v]).
        """
        assert (suffix_k is None) == (suffix_v is None) == (
            suffix_positions is None
        ) == (suffix_mask is None), "suffix tensors must be provided together"
        batch_size = x_new.size(0)
        x = x_new.unsqueeze(1)  # [B, 1, D]
        normed_x = self._norm_input(x)
        uvqk_x = F.silu(self._uvqk(normed_x))
        u, v, q, k = torch.split(
            uvqk_x,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )
        q_heads = self._reshape_uvqk_for_mm(q, self._num_heads, self._attention_dim)
        k_new = self._reshape_uvqk_for_mm(k, self._num_heads, self._attention_dim)
        v_new = self._reshape_uvqk_for_mm(v, self._num_heads, self._linear_dim)

        scale = self._attention_dim**-0.5
        prefix_len = prefix_k.size(1)
        device = x_new.device
        bias_dtype = q_heads.dtype

        # Scores vs the shared prefix (broadcast, never materialized per row).
        prefix_scores = (
            torch.einsum("bhqd,htd->bhqt", q_heads, prefix_k.to(q_heads.dtype)) * scale
        )
        prefix_positions = torch.arange(prefix_len, device=device).view(1, prefix_len)
        prefix_scores = prefix_scores + self._relative_bias(
            prefix_positions, q_positions
        ).to(bias_dtype)

        score_parts = [prefix_scores]
        has_suffix = suffix_k is not None and suffix_k.size(2) > 0
        if has_suffix:
            suffix_scores = (
                torch.einsum("bhqd,bhsd->bhqs", q_heads, suffix_k.to(q_heads.dtype))
                * scale
            )
            suffix_scores = suffix_scores + self._relative_bias(
                suffix_positions, q_positions
            ).to(bias_dtype)
            suffix_scores = suffix_scores.masked_fill(
                ~suffix_mask.view(batch_size, 1, 1, -1), float("-inf")
            )
            score_parts.append(suffix_scores)

        # Self-attention term: distance 0.
        self_scores = (q_heads * k_new).sum(dim=-1, keepdim=True) * scale
        self_scores = self_scores + self._ps_w[:, self._max_seq_len - 1].view(
            1, -1, 1, 1
        ).to(bias_dtype)
        score_parts.append(self_scores)

        scores = torch.cat(score_parts, dim=-1)  # [B, H, 1, T + s + 1]
        weights = torch.softmax(scores.float(), dim=-1).to(q_heads.dtype)

        attn_output = torch.einsum(
            "bhqt,htd->bhqd", weights[..., :prefix_len], prefix_v.to(weights.dtype)
        )
        offset = prefix_len
        if has_suffix:
            suffix_len = suffix_k.size(2)
            attn_output = attn_output + torch.einsum(
                "bhqs,bhsd->bhqd",
                weights[..., offset : offset + suffix_len],
                suffix_v.to(weights.dtype),
            )
            offset += suffix_len
        attn_output = attn_output + weights[..., offset:] * v_new

        attn_output = self._norm_attn_output(
            attn_output.permute(0, 2, 1, 3).reshape(
                batch_size, 1, self._num_heads * self._linear_dim
            )
        )
        o_input = F.dropout(
            u * attn_output, p=self._dropout_ratio, training=self.training
        )
        x_out = (self._o(o_input) + x).squeeze(1)
        return x_out, k_new, v_new
