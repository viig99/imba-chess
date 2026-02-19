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

        # Position bias — always present
        self._ps_w = torch.nn.Parameter(
            torch.empty(2 * self._max_seq_len - 1).normal_(mean=0, std=0.02),
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
        score = score + self._ps_w[idx].to(score.dtype)
        return score

    def _generate_rab_score_mod(self):
        return self._position_score_mod

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask | None = None,
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

        # output shape: [1, num_heads, S, linear_dim]
        attn_output: torch.Tensor = flex_attention(
            query=self._reshape_uvqk_for_mm(q, self._num_heads, self._attention_dim),
            key=self._reshape_uvqk_for_mm(k, self._num_heads, self._attention_dim),
            value=self._reshape_uvqk_for_mm(v, self._num_heads, self._linear_dim),
            block_mask=block_mask,
            score_mod=self._generate_rab_score_mod(),
        )  # type: ignore

        attn_output = self._norm_attn_output(
            attn_output.permute(0, 2, 1, 3).reshape(
                1, S, self._num_heads * self._linear_dim
            )
        )

        o_input = F.dropout(
            u * attn_output, p=self._dropout_ratio, training=self.training
        )
        return (self._o(o_input) + x).squeeze(0)
