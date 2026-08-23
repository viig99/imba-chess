from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import Literal
from torch.nn.attention.flex_attention import BlockMask, flex_attention


# A dense additive mask is [1, H, S, S], so it grows quadratically in S: fine
# for play/search batches (a few hundred tokens), ruinous for dataset-sized
# ones (~805 MiB at S=4096, H=12, fp32). Cross this and something picked the
# wrong mask -- fail loudly rather than allocate it.
_MAX_ADDITIVE_MASK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class GroupedDecodeCache:
    """Per-wave, layer-invariant scratch for `forward_decode_grouped`.

    Every field here is a pure function of data that is fixed for the whole
    decode wave -- the row bucketing, the query positions, and the suffix
    positions/mask -- none of which a layer can change. Recomputing them
    inside the per-group loop therefore ran identical work `num_layers`
    times over. Building them once per wave is the same class of fix as the
    `row_idx_per_group` hoist (commit 292277b) and, for the same reason,
    cannot move a single bit of the result.

    All lists are indexed by group `g` and have length G.

    row_idx: [B_g] row indices this group owns.
    prefix_rel: [B_g, T_g] clamped relative-position indices for the prefix,
        i.e. `_relative_bias`'s `rel` -- layer-invariant; only the
        `_ps_w[:, rel]` gather that follows it is per-layer.
    suffix_rel: [B_g, s] same for the suffix keys, or None when the wave has
        no suffix tokens at all.
    suffix_fill_mask: [B_g, 1, 1, s] pre-negated, pre-viewed suffix mask
        ready for `masked_fill`, or None alongside `suffix_rel`.
    """

    row_idx: list[torch.Tensor]
    prefix_rel: list[torch.Tensor]
    suffix_rel: list[torch.Tensor | None]
    suffix_fill_mask: list[torch.Tensor | None]


def build_grouped_decode_cache(
    *,
    row_idx_per_group: list[torch.Tensor],
    prefix_lens_list: list[int],
    q_positions: torch.Tensor,
    suffix_positions: torch.Tensor | None,
    suffix_mask: torch.Tensor | None,
    max_seq_len: int,
) -> GroupedDecodeCache:
    """Compute a wave's layer-invariant decode scratch (see GroupedDecodeCache).

    `max_seq_len` is the layers' shared `_max_seq_len`; the caller is
    responsible for asserting the layers agree on it, since `prefix_rel`
    would otherwise be wrong for a layer that disagreed.
    """
    device = q_positions.device
    lo, hi = 0, 2 * max_seq_len - 2
    has_suffix = (
        suffix_positions is not None
        and suffix_mask is not None
        and suffix_positions.size(-1) > 0
    )

    row_idx: list[torch.Tensor] = []
    prefix_rel: list[torch.Tensor] = []
    suffix_rel: list[torch.Tensor | None] = []
    suffix_fill_mask: list[torch.Tensor | None] = []
    for g, rows in enumerate(row_idx_per_group):
        row_idx.append(rows)
        if rows.numel() == 0:
            # Skipped by the group loop anyway; keep the lists index-aligned.
            prefix_rel.append(rows)
            suffix_rel.append(None)
            suffix_fill_mask.append(None)
            continue
        q_pos_g = q_positions.index_select(0, rows).view(-1, 1)
        actual_len = prefix_lens_list[g]
        prefix_positions = torch.arange(actual_len, device=device).view(1, actual_len)
        prefix_rel.append(
            torch.clamp(prefix_positions - q_pos_g + (max_seq_len - 1), lo, hi)
        )
        if has_suffix:
            suffix_pos_g = suffix_positions.index_select(0, rows)
            suffix_rel.append(
                torch.clamp(suffix_pos_g - q_pos_g + (max_seq_len - 1), lo, hi)
            )
            suffix_fill_mask.append(
                ~suffix_mask.index_select(0, rows).view(rows.numel(), 1, 1, -1)
            )
        else:
            suffix_rel.append(None)
            suffix_fill_mask.append(None)

    return GroupedDecodeCache(
        row_idx=row_idx,
        prefix_rel=prefix_rel,
        suffix_rel=suffix_rel,
        suffix_fill_mask=suffix_fill_mask,
    )


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

    def _additive_mask(
        self, allowed: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """[1, H, S, S] additive mask equivalent to block_mask + score_mod.

        Folds this layer's relative-position bias into an explicit bias tensor
        so SDPA computes the same scores flex_attention would, then blocks
        disallowed positions with -inf.

        `_ps_w` is a PER-LAYER parameter, so this is built per layer. Sharing
        one layer's bias across the stack runs fine and is fast, but silently
        changes move selection -- pinned by
        tests/test_dense_attn_mask.py::test_dense_path_uses_each_layer_own_relative_position_bias.
        """
        S = allowed.size(-1)
        want = self._num_heads * S * S * torch.empty((), dtype=dtype).element_size()
        if want > _MAX_ADDITIVE_MASK_BYTES:
            raise ValueError(
                f"dense attention mask would need {want / 2**20:.0f} MiB "
                f"(S={S}, heads={self._num_heads}, {dtype}). The dense path is "
                "for small play/search batches; pass a BlockMask from "
                "create_batch_block_mask for dataset-sized batches."
            )
        pos = torch.arange(S, device=allowed.device)
        idx = torch.clamp(
            (pos.unsqueeze(0) - pos.unsqueeze(1)) + (self._max_seq_len - 1),
            0,
            2 * self._max_seq_len - 2,
        )
        bias = self._ps_w[:, idx].unsqueeze(0).to(dtype)
        return bias.masked_fill(~allowed, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask | torch.Tensor | None = None,
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

        # A plain Tensor is a dense [S, S] bool mask from
        # create_batch_dense_mask (inference); a BlockMask goes to flex
        # (training). Both admit the same positions and compute the same
        # scores -- see tests/test_dense_attn_mask.py.
        # output shape: [1, num_heads, S, linear_dim]
        if block_mask is None and q_heads.device.type == "cpu":
            # torch 2.13: flex_attention has no CPU backward, and a layer's own
            # parameters require grad, so it raises during FORWARD on CPU even
            # under .eval(). block_mask=None means "attend everywhere", whose
            # dense equivalent is an all-true mask -- same scores, and SDPA does
            # support CPU backward. _additive_mask still folds in _ps_w.
            block_mask = torch.ones(S, S, dtype=torch.bool, device=q_heads.device)
        attn_output: torch.Tensor
        if isinstance(block_mask, torch.Tensor):
            attn_output = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                attn_mask=self._additive_mask(block_mask, dtype=q_heads.dtype),
            )
        else:
            attn_output = flex_attention(
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
        return self._bias_from_rel(rel)

    def _bias_from_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """The per-layer half of `_relative_bias`: gather `_ps_w` at `rel`.

        Split out because `rel` is layer-invariant (it is pure position
        arithmetic) while `_ps_w` is per-layer. The grouped decode path
        computes `rel` once per wave and calls this once per layer; see
        GroupedDecodeCache.
        """
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

    def forward_decode_grouped(
        self,
        x_new: torch.Tensor,
        *,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        prefix_lens_list: list[int],
        group_index: torch.Tensor,
        row_idx_per_group: list[torch.Tensor],
        q_positions: torch.Tensor,
        suffix_k: torch.Tensor | None = None,
        suffix_v: torch.Tensor | None = None,
        suffix_positions: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
        decode_cache: GroupedDecodeCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one new token per batch row against per-game grouped
        prefix K/V (cross-game merged wave).

        x_new: [B, D]; prefix_k/v: [G, H, maxP, d] (games padded on the
        token dim); prefix_lens_list: length-G plain Python list, real
        (unpadded) length per game -- deliberately a host-side list[int],
        not a tensor: the per-group loop below reads one entry per group per
        layer, and indexing a device tensor there (`.item()`) would force a
        device->host sync G*num_layers times per decode step. The caller
        already has these lengths as Python ints before any tensor is built
        (see _DecodeRequest.prefix_len / _merge_decode_requests in
        scripts/generate_search_rollouts.py), so no tensor round-trip -- and
        no sync -- is needed to get them here. group_index: [B] row -> game
        index g. suffix_k/v/positions/mask are unchanged from
        forward_decode -- already per-row.

        Rows are bucketed by group and each group's prefix attention is
        computed as ONE einsum over that group's real (unpadded) prefix
        slice `prefix_k[g, :, :prefix_lens[g], :]` -- this reproduces
        forward_decode's single-prefix math exactly (same shapes, same
        score/bias/softmax path) for each group in turn, so it inherits
        forward_decode's correctness rather than re-deriving the masking
        convention. Never gather `prefix_k[group_index]`: that would
        materialize a [B, H, maxP, d] per-row copy of every game's prefix,
        the exact memory blowup grouping exists to avoid. Only the (small)
        per-row query/key/value/suffix tensors are ever index_select'd.

        Returns (x_out [B, D], k_new [B, H, 1, d_qk], v_new [B, H, 1, d_v])
        -- k_new/v_new for the FULL batch, in original row order (they only
        depend on each row's own token, not on its group's prefix).
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
        # q/k/v projections are per-row (independent of grouping) -- compute
        # once for the whole batch, same as forward_decode.
        q_heads = self._reshape_uvqk_for_mm(q, self._num_heads, self._attention_dim)
        k_new = self._reshape_uvqk_for_mm(k, self._num_heads, self._attention_dim)
        v_new = self._reshape_uvqk_for_mm(v, self._num_heads, self._linear_dim)

        scale = self._attention_dim**-0.5
        device = x_new.device
        bias_dtype = q_heads.dtype
        has_suffix = suffix_k is not None and suffix_k.size(2) > 0

        attn_output = torch.empty(
            batch_size, self._num_heads, 1, self._linear_dim,
            dtype=q_heads.dtype, device=device,
        )

        num_groups = prefix_k.size(0)
        if len(row_idx_per_group) != num_groups:
            raise ValueError(
                "row_idx_per_group must have one entry per group "
                f"(== prefix_k.size(0) == {num_groups}), got "
                f"{len(row_idx_per_group)}"
            )
        if decode_cache is None:
            # Standalone call (tests, or any caller that has not been taught
            # about the wave cache): build it here so there is exactly one
            # implementation of this arithmetic. The model's own decode loop
            # builds it once per wave and passes it to all num_layers layers.
            decode_cache = build_grouped_decode_cache(
                row_idx_per_group=row_idx_per_group,
                prefix_lens_list=prefix_lens_list,
                q_positions=q_positions,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
                max_seq_len=self._max_seq_len,
            )
        elif len(decode_cache.row_idx) != num_groups:
            raise ValueError(
                "decode_cache must have one entry per group "
                f"(== prefix_k.size(0) == {num_groups}), got "
                f"{len(decode_cache.row_idx)}"
            )
        # Group-invariant (it indexes _ps_w at a fixed offset), so it was the
        # same tensor G times per layer.
        self_bias = self._ps_w[:, self._max_seq_len - 1].view(1, -1, 1, 1).to(
            bias_dtype
        )
        for g in range(num_groups):
            # Precomputed by the caller once per wave. These depend only on
            # group_index, which is fixed across layers, so deriving them here
            # ran `nonzero` num_layers times over for every group -- and
            # `nonzero` has a data-dependent output shape, so each one was also
            # a device->host sync.
            row_idx = decode_cache.row_idx[g]
            if row_idx.numel() == 0:
                continue
            actual_len = prefix_lens_list[g]
            max_p = prefix_k.size(2)
            if not 0 <= actual_len <= max_p:
                raise ValueError(
                    f"prefix_lens[{g}]={actual_len} out of range for padded "
                    f"prefix length {max_p}"
                )

            q_g = q_heads.index_select(0, row_idx)
            # Real (unpadded) prefix slice for this game only -- a view, not
            # a per-row copy: identical shape/semantics to forward_decode's
            # prefix_k [H, T, d].
            prefix_k_g = prefix_k[g, :, :actual_len, :]
            prefix_v_g = prefix_v[g, :, :actual_len, :]

            prefix_scores = (
                torch.einsum("bhqd,htd->bhqt", q_g, prefix_k_g.to(q_g.dtype)) * scale
            )
            prefix_scores = prefix_scores + self._bias_from_rel(
                decode_cache.prefix_rel[g]
            ).to(bias_dtype)

            score_parts = [prefix_scores]
            if has_suffix:
                suffix_k_g = suffix_k.index_select(0, row_idx)
                suffix_v_g = suffix_v.index_select(0, row_idx)
                suffix_rel_g = decode_cache.suffix_rel[g]
                suffix_fill_g = decode_cache.suffix_fill_mask[g]
                if suffix_rel_g is None or suffix_fill_g is None:
                    raise ValueError(
                        "decode_cache has no suffix entries but the wave has "
                        "suffix tokens -- it was built for a different wave"
                    )
                suffix_scores = (
                    torch.einsum(
                        "bhqd,bhsd->bhqs", q_g, suffix_k_g.to(q_g.dtype)
                    )
                    * scale
                )
                suffix_scores = suffix_scores + self._bias_from_rel(
                    suffix_rel_g
                ).to(bias_dtype)
                suffix_scores = suffix_scores.masked_fill(
                    suffix_fill_g, float("-inf")
                )
                score_parts.append(suffix_scores)

            k_new_g = k_new.index_select(0, row_idx)
            v_new_g = v_new.index_select(0, row_idx)
            self_scores = (q_g * k_new_g).sum(dim=-1, keepdim=True) * scale
            self_scores = self_scores + self_bias
            score_parts.append(self_scores)

            scores = torch.cat(score_parts, dim=-1)  # [Bg, H, 1, T_g + s + 1]
            weights = torch.softmax(scores.float(), dim=-1).to(q_g.dtype)

            out_g = torch.einsum(
                "bhqt,htd->bhqd", weights[..., :actual_len], prefix_v_g.to(weights.dtype)
            )
            offset = actual_len
            if has_suffix:
                suffix_len = suffix_k_g.size(2)
                out_g = out_g + torch.einsum(
                    "bhqs,bhsd->bhqd",
                    weights[..., offset : offset + suffix_len],
                    suffix_v_g.to(weights.dtype),
                )
                offset += suffix_len
            out_g = out_g + weights[..., offset:] * v_new_g

            attn_output.index_copy_(0, row_idx, out_g)

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
