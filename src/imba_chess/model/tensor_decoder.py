"""Whole-neural-decoder compilation and an experimental reusable SDPA workspace.

The workspace belongs to one executor/model, never to a search node. Returned
K/V are fresh projections, not views into mutable workspace storage.
"""

import torch
import torch.nn.functional as F

from .hstu_attention import GroupedDecodeCache, _batch_bias, _one_query_attention


class TensorDecoder(torch.nn.Module):
    """A single tensor graph from board features through logits and new K/V."""

    def __init__(self, model, sdpa=False):
        super().__init__()
        self.model = model
        self.sdpa = sdpa

    def forward(
        self,
        batch,
        positions,
        prefix,
        suffix,
        prefix_rel,
        prefix_fill,
        suffix_rel,
        suffix_fill,
        workspace,
    ):
        model = self.model
        x = model.position_embedding.at_positions(
            model._build_content(batch), positions
        )
        cache = GroupedDecodeCache(
            [], [], [], [], prefix_rel, prefix_fill, suffix_rel, suffix_fill
        )
        new_kv = []
        for index, layer in enumerate(model.layers):
            residual = x.unsqueeze(1)
            uvqk = F.silu(layer._uvqk(layer._norm_input(residual)))
            u, v, q, k = uvqk.split(
                [
                    layer._linear_dim * layer._num_heads,
                    layer._linear_dim * layer._num_heads,
                    layer._attention_dim * layer._num_heads,
                    layer._attention_dim * layer._num_heads,
                ],
                dim=-1,
            )
            q = layer._reshape_uvqk_for_mm(q, layer._num_heads, layer._attention_dim)
            k = layer._reshape_uvqk_for_mm(k, layer._num_heads, layer._attention_dim)
            v = layer._reshape_uvqk_for_mm(v, layer._num_heads, layer._linear_dim)
            self_bias = layer._ps_w[:, layer._max_seq_len - 1].view(1, -1, 1, 1)
            if self.sdpa:
                wk, wv = workspace[index]
                wk[:, :, -1:] = k
                wv[:, :, -1:] = v
                bias = torch.cat(
                    [
                        _batch_bias(layer._ps_w, prefix_rel).masked_fill(
                            prefix_fill, -torch.inf
                        ),
                        _batch_bias(layer._ps_w, suffix_rel).masked_fill(
                            suffix_fill, -torch.inf
                        ),
                        self_bias.expand(x.size(0), -1, -1, -1),
                    ],
                    dim=-1,
                )
                attn = F.scaled_dot_product_attention(
                    q, wk, wv, attn_mask=bias, dropout_p=0.0, is_causal=False
                )
            else:
                attn = _one_query_attention(
                    q,
                    *prefix[index],
                    k,
                    v,
                    *suffix[index],
                    cache,
                    layer._ps_w,
                    self_bias,
                    layer._attention_dim**-0.5,
                )
            attn = layer._norm_attn_output(
                attn.permute(0, 2, 1, 3).reshape(
                    x.size(0), 1, layer._num_heads * layer._linear_dim
                )
            )
            x = (layer._o(u * attn) + residual).squeeze(1)
            new_kv.append((k, v))
        x = model.final_norm(x)
        output = {"logits": model.prediction_head(x), "kv": new_kv}
        if model.value_head is not None:
            output["value_logits"] = model.value_head(x)
        return output


class DecoderRunner:
    """Host-side validation/preparation, kept outside the compiled graph.

    Prefix workspaces are reused while the ordered root-owner cohort is stable.
    Cohort changes invalidate the prefix contents; this is deliberately not a
    persistent-slot/paged cache. Memory and retained owners are bounded to one wave.
    """

    def __init__(self, model, mode, suffix_capacity=32):
        if model.training:
            raise ValueError("tensor decoder requires an evaluation-mode model")
        if mode not in ("tensor", "compiled", "sdpa", "compiled-sdpa"):
            raise ValueError("unknown tensor decoder mode")
        self.model = model
        self.sdpa = "sdpa" in mode
        self.suffix_capacity = suffix_capacity
        decoder = TensorDecoder(model, self.sdpa).eval()
        self.decode = (
            torch.compile(decoder, fullgraph=True, dynamic=True)
            if mode.startswith("compiled")
            else decoder
        )
        self.clear()

    def clear(self):
        self.prefix_owner = None
        self.workspace = None

    def __call__(self, merged, *, workspace=None):
        if self.model.training:
            raise ValueError("tensor decoder is inference-only")
        if merged.group_sizes != [1] * len(merged.group_sizes):
            raise ValueError("tensor decoder requires one query per game")
        p = merged.prefix_kv_grouped[0][0].size(2)
        lengths = merged.prefix_lens_list
        if any(n < 0 or n > p for n in lengths):
            raise ValueError("invalid prefix lengths")
        s = 0 if merged.suffix_positions is None else merged.suffix_positions.size(1)
        capacity = self.suffix_capacity
        if s > capacity:
            raise ValueError("search suffix exceeds decoder workspace capacity")
        device = self.model.piece_square_embedding.weight.device
        positions = merged.positions.to(device, non_blocking=True)
        prefix_lens = merged.prefix_lens.to(device, non_blocking=True)
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in merged.new_token_batch.items()
        }
        g = len(lengths)
        suffix = []
        for i, (k, v) in enumerate(merged.prefix_kv_grouped):
            if merged.suffix_kv is None:
                suffix.append(
                    (
                        k.new_zeros((g, k.size(1), capacity, k.size(3))),
                        v.new_zeros((g, v.size(1), capacity, v.size(3))),
                    )
                )
            else:
                sk, sv = merged.suffix_kv[i]
                suffix.append(
                    (
                        F.pad(sk, (0, 0, 0, capacity - s)),
                        F.pad(sv, (0, 0, 0, capacity - s)),
                    )
                )
        q = positions[:, None]
        max_pos = self.model.layers[0]._max_seq_len
        prefix_rel = (torch.arange(p, device=device)[None, :] - q + max_pos - 1).clamp(
            0, 2 * max_pos - 2
        )
        prefix_fill = (torch.arange(p, device=device)[None, :] >= prefix_lens[:, None])[
            :, None, None, :
        ]
        sp = torch.zeros((g, capacity), device=device, dtype=torch.long)
        sm = torch.zeros((g, capacity), device=device, dtype=torch.bool)
        if s:
            sp[:, :s] = merged.suffix_positions.to(device, non_blocking=True)
            sm[:, :s] = merged.suffix_mask.to(device, non_blocking=True)
        suffix_rel = (sp - q + max_pos - 1).clamp(0, 2 * max_pos - 2)
        suffix_fill = ~sm[:, None, None, :]
        if self.sdpa:
            if workspace is not None:
                # The executor packed roots directly into this allocation.
                # Prefix views and the mutable branch share storage, but never
                # overlap. No second prefix copy is needed here.
                self.workspace = workspace
                self.prefix_owner = merged.prefix_kv_grouped
            elif self.prefix_owner is not merged.prefix_kv_grouped:
                self.workspace = [
                    (
                        k.new_zeros((*k.shape[:2], p + capacity + 1, k.size(-1))),
                        v.new_zeros((*v.shape[:2], p + capacity + 1, v.size(-1))),
                    )
                    for k, v in merged.prefix_kv_grouped
                ]
                for (wk, wv), (k, v) in zip(self.workspace, merged.prefix_kv_grouped):
                    wk[:, :, :p].copy_(k)
                    wv[:, :, :p].copy_(v)
                self.prefix_owner = merged.prefix_kv_grouped
            for (wk, wv), (sk, sv) in zip(self.workspace, suffix):
                wk[:, :, p:-1].copy_(sk)
                wv[:, :, p:-1].copy_(sv)
        return self.decode(
            batch,
            positions,
            merged.prefix_kv_grouped,
            suffix,
            prefix_rel,
            prefix_fill,
            suffix_rel,
            suffix_fill,
            self.workspace if self.sdpa else (),
        )
