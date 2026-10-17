"""Compiled neural decoder over executor-owned inputs.

Staging and caches belong to the executor's DecodeWorkspace. Returned K/V are
fresh projections, not views into mutable staging storage.
"""

import torch
import torch.nn.functional as F
from .decode_attention import decode_attention_silu


class TensorDecoder(torch.nn.Module):
    """A single tensor graph from board features through logits and new K/V."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, positions, prefix, arena, chain, lengths, depths):
        """prefix: per layer (K, V) [g, H, P, d]; arena: [layers, 2 (K, V),
        rows, H, d] holding every evaluated node; chain: [g, S] each row's
        ancestor arena rows; lengths/depths: [g] real prefix and suffix sizes."""
        model = self.model
        x = model.position_embedding.at_positions(
            model._build_content(batch), positions
        )
        new_kv = []
        for index, layer in enumerate(model.layers):
            residual = x.unsqueeze(1)
            heads = layer._num_heads
            u, v, q, k = layer._uvqk(layer._norm_input(residual)).split(
                [
                    layer._linear_dim * heads,
                    layer._linear_dim * heads,
                    layer._attention_dim * heads,
                    layer._attention_dim * heads,
                ],
                dim=-1,
            )
            # Raw projections as [g, H, 1, d] views; the kernel applies SiLU and
            # returns the new K/V, so no per-head copies are materialized.
            q, k, v = (t.unflatten(-1, (heads, -1)).transpose(1, 2) for t in (q, k, v))
            attn, k, v = decode_attention_silu(
                q,
                k,
                v,
                *prefix[index],
                arena[index, 0],
                arena[index, 1],
                chain,
                lengths,
                depths,
                layer._ps_w,
                layer._attention_dim ** (-0.5),
                layer._max_seq_len,
            )
            attn = layer._norm_attn_output(
                attn.permute(0, 2, 1, 3).reshape(
                    x.size(0), 1, layer._num_heads * layer._linear_dim
                )
            )
            x = (layer._o(F.silu(u) * attn) + residual).squeeze(1)
            new_kv.append((k, v))
        x = model.final_norm(x)
        output = {"logits": model.prediction_head(x), "kv": new_kv}
        if model.value_head is not None:
            output["value_logits"] = model.value_logits(x, output["logits"])
        return output


class DecoderRunner:
    """Host-side validation/preparation, kept outside the compiled graph.

    The caller owns prefix and arena storage; the runner retains no search
    owners or cached decode inputs between calls.
    """

    def __init__(self, model, suffix_capacity=32):
        if model.training:
            raise ValueError("tensor decoder requires an evaluation-mode model")
        self.model = model
        self.suffix_capacity = suffix_capacity
        self.decode = torch.compile(
            TensorDecoder(model).eval(), fullgraph=True, dynamic=True
        )

    def __call__(self, merged):
        if self.model.training:
            raise ValueError("tensor decoder is inference-only")
        if merged.group_sizes != [1] * len(merged.group_sizes):
            raise ValueError("tensor decoder requires one query per game")
        p = merged.prefix_kv_grouped[0][0].size(2)
        lengths = merged.prefix_lens_list
        if any((n < 0 or n > p for n in lengths)):
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
        depths = positions - prefix_lens
        if s:
            # The decoder reads suffix key j at position length + j for j < depth.
            j = torch.arange(s, device=device)
            mask = merged.suffix_mask.to(device)
            expected = j[None, :] < depths[:, None]
            suffix_pos = merged.suffix_positions.to(device)
            if not torch.equal(mask, expected) or not torch.equal(
                suffix_pos[mask], (prefix_lens[:, None] + j)[mask]
            ):
                raise ValueError("suffix must be each row's contiguous ancestors")
        elif bool((depths != 0).any()):
            raise ValueError("positions exceed prefix lengths without a suffix")
        # Stage every row's suffix as arena rows; row 0 stays zero padding.
        k0 = merged.prefix_kv_grouped[0][0]
        arena = k0.new_zeros(
            (len(merged.prefix_kv_grouped), 2, 1 + g * s, k0.size(1), k0.size(-1))
        )
        if s:
            for i, pair in enumerate(merged.suffix_kv):
                for kind, t in enumerate(pair):
                    arena[i, kind, 1:] = t.permute(0, 2, 1, 3).reshape(
                        g * s, t.size(1), t.size(-1)
                    )
        chain = torch.zeros((g, capacity), dtype=torch.long, device=device)
        if s:
            chain[:, :s] = 1 + torch.arange(g * s, device=device).view(g, s)
        return self.decode(
            batch,
            positions,
            merged.prefix_kv_grouped,
            arena,
            chain,
            prefix_lens,
            depths,
        )
