from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imba_chess.model import create_batch_block_mask
from imba_chess.model.hstu_attention import SequentialTransductionUnitJagged
from imba_chess.model.position_embedding import PositionEmbedding

ATOL = 1e-5
RTOL = 1e-5


def _layer() -> SequentialTransductionUnitJagged:
    torch.manual_seed(0)
    return SequentialTransductionUnitJagged(
        embedding_dim=32,
        linear_hidden_dim=8,
        attention_dim=8,
        dropout_ratio=0.0,
        num_heads=2,
        max_seq_len=64,
    ).eval()


def _full_forward(layer, x):
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, x.size(0)]),
        total_tokens=x.size(0),
        device=x.device,
    )
    return layer(x=x, block_mask=block_mask)


def test_forward_return_kv_output_unchanged():
    layer = _layer()
    x = torch.randn(10, 32)
    full = _full_forward(layer, x)
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, 10]), total_tokens=10, device=x.device
    )
    out, (k, v) = layer(x=x, block_mask=block_mask, return_kv=True)
    torch.testing.assert_close(out, full, atol=ATOL, rtol=RTOL)
    assert k.shape == (2, 10, 8)  # [H, S, attention_dim]
    assert v.shape == (2, 10, 8)  # [H, S, linear_hidden_dim]


def test_layer_decode_matches_full_forward_token_by_token():
    layer = _layer()
    S, T = 13, 9  # prefill 9 tokens, decode tokens 9..12 sequentially
    x = torch.randn(S, 32)
    full = _full_forward(layer, x)

    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, T]), total_tokens=T, device=x.device
    )
    out_prefix, (prefix_k, prefix_v) = layer(
        x=x[:T], block_mask=block_mask, return_kv=True
    )
    torch.testing.assert_close(out_prefix, full[:T], atol=ATOL, rtol=RTOL)

    suffix_k_parts: list[torch.Tensor] = []
    suffix_v_parts: list[torch.Tensor] = []
    for i in range(T, S):
        if suffix_k_parts:
            suffix_k = torch.cat(suffix_k_parts, dim=2)  # [1, H, s, d]
            suffix_v = torch.cat(suffix_v_parts, dim=2)
            s = suffix_k.size(2)
            suffix_positions = torch.arange(T, T + s).view(1, s)
            suffix_mask = torch.ones(1, s, dtype=torch.bool)
        else:
            suffix_k = suffix_v = suffix_positions = suffix_mask = None
        x_out, k_new, v_new = layer.forward_decode(
            x[i : i + 1],
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            q_positions=torch.tensor([i]),
            suffix_k=suffix_k,
            suffix_v=suffix_v,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
        )
        torch.testing.assert_close(x_out.squeeze(0), full[i], atol=ATOL, rtol=RTOL)
        suffix_k_parts.append(k_new)
        suffix_v_parts.append(v_new)


def test_layer_decode_batched_wave_with_mixed_suffix_lengths():
    layer = _layer()
    T = 7
    prefix = torch.randn(T, 32)
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, T]), total_tokens=T, device=prefix.device
    )
    _, (prefix_k, prefix_v) = layer(x=prefix, block_mask=block_mask, return_kv=True)

    # Node A: depth 0 (no suffix). Node B: depth 1 (one ancestor token).
    tok_a = torch.randn(1, 32)
    tok_b_parent = torch.randn(1, 32)
    tok_b = torch.randn(1, 32)

    # References via full forwards over explicit sequences.
    full_a = _full_forward(layer, torch.cat([prefix, tok_a]))[T]
    full_b = _full_forward(layer, torch.cat([prefix, tok_b_parent, tok_b]))[T + 1]

    # Evaluate B's parent first to obtain its (k, v).
    _, kp, vp = layer.forward_decode(
        tok_b_parent,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        q_positions=torch.tensor([T]),
    )

    # One wave containing A (depth 0, padded suffix) and B (depth 1).
    x_new = torch.cat([tok_a, tok_b])  # [2, 32]
    suffix_k = torch.cat([torch.zeros_like(kp), kp])  # [2, H, 1, d]
    suffix_v = torch.cat([torch.zeros_like(vp), vp])
    suffix_positions = torch.tensor([[0], [T]])
    suffix_mask = torch.tensor([[False], [True]])
    x_out, _, _ = layer.forward_decode(
        x_new,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        q_positions=torch.tensor([T, T + 1]),
        suffix_k=suffix_k,
        suffix_v=suffix_v,
        suffix_positions=suffix_positions,
        suffix_mask=suffix_mask,
    )
    torch.testing.assert_close(x_out[0], full_a, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(x_out[1], full_b, atol=ATOL, rtol=RTOL)


def test_position_embedding_at_positions_matches_forward():
    torch.manual_seed(1)
    pe = PositionEmbedding(max_seq_len=16, embedding_dim=8, dropout_rate=0.0).eval()
    content = torch.randn(5, 8)
    offsets = torch.tensor([0, 5])
    full = pe(content, offsets)
    picked = pe.at_positions(content, torch.arange(5))
    torch.testing.assert_close(picked, full, atol=ATOL, rtol=RTOL)
    # Clamp behavior matches forward's clamp.
    over = pe.at_positions(content[:1], torch.tensor([99]))
    ref = pe.at_positions(content[:1], torch.tensor([15]))
    torch.testing.assert_close(over, ref, atol=ATOL, rtol=RTOL)
