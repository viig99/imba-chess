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


from imba_chess.config import ModelConfig
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


def _tiny_model(vocab_size: int = 32) -> HSTUChessModel:
    torch.manual_seed(2)
    config = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=8,
            attention_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            max_position_embeddings=64,
            enable_value_head=True,
        ),
        move_vocab_size=vocab_size,
    )
    return HSTUChessModel(config).eval()


def _random_token_ids(n: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "piece_ids": torch.randint(0, 13, (n, 64), generator=g),
        "seq_token_id": torch.randint(0, 2, (n,), generator=g),
        "turn_id": torch.randint(0, 2, (n,), generator=g),
        "castle_id": torch.randint(0, 16, (n,), generator=g),
        "ep_file_id": torch.randint(0, 9, (n,), generator=g),
        "halfmove_bucket_id": torch.randint(0, 50, (n,), generator=g),
        "fullmove_bucket_id": torch.randint(0, 100, (n,), generator=g),
        "prev_move_id": torch.randint(0, 32, (n,), generator=g),
    }


def _full_batch(token_ids: dict[str, torch.Tensor]) -> dict:
    n = token_ids["piece_ids"].size(0)
    batch = dict(token_ids)
    batch.update(
        {
            "total_tokens": n,
            "seq_offsets": torch.tensor([0, n]),
            "target_move_id": torch.full((n,), -100, dtype=torch.long),
        }
    )
    return batch


def test_model_decode_matches_full_forward_over_depths():
    model = _tiny_model()
    T, max_depth = 9, 4
    ids = _random_token_ids(T + max_depth, seed=7)
    prefix_ids = {key: value[:T] for key, value in ids.items()}

    with torch.no_grad():
        full = model(_full_batch(ids), return_loss=False)
        prefill = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)

    prefix_kv = prefill["kv_caches"]
    suffix_kv = None
    suffix_positions = suffix_mask = None
    for depth in range(max_depth):
        i = T + depth
        step_ids = {key: value[i : i + 1] for key, value in ids.items()}
        with torch.no_grad():
            out = model.forward_decode(
                new_token_batch=step_ids,
                positions=torch.tensor([i]),
                prefix_kv=prefix_kv,
                suffix_kv=suffix_kv,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
        torch.testing.assert_close(
            out["logits"].squeeze(0), full["logits"][i], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out["value_logits"].squeeze(0),
            full["value_logits"][i],
            atol=ATOL,
            rtol=RTOL,
        )
        # Grow the suffix cache with this token's per-layer (k, v).
        if suffix_kv is None:
            suffix_kv = [(k, v) for k, v in out["kv"]]
        else:
            suffix_kv = [
                (torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2))
                for (sk, sv), (k, v) in zip(suffix_kv, out["kv"])
            ]
        s = suffix_kv[0][0].size(2)
        suffix_positions = torch.arange(T, T + s).view(1, s)
        suffix_mask = torch.ones(1, s, dtype=torch.bool)


def test_model_decode_mixed_depth_wave():
    model = _tiny_model()
    T = 8
    ids = _random_token_ids(T + 3, seed=11)  # prefix + [a, b_parent, b]
    prefix_ids = {key: value[:T] for key, value in ids.items()}
    tok_a = {key: value[T : T + 1] for key, value in ids.items()}
    tok_bp = {key: value[T + 1 : T + 2] for key, value in ids.items()}
    tok_b = {key: value[T + 2 : T + 3] for key, value in ids.items()}

    seq_a = {key: torch.cat([prefix_ids[key], tok_a[key]]) for key in ids}
    seq_b = {
        key: torch.cat([prefix_ids[key], tok_bp[key], tok_b[key]]) for key in ids
    }
    with torch.no_grad():
        full_a = model(_full_batch(seq_a), return_loss=False)
        full_b = model(_full_batch(seq_b), return_loss=False)
        prefill = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)
        parent_out = model.forward_decode(
            new_token_batch=tok_bp,
            positions=torch.tensor([T]),
            prefix_kv=prefill["kv_caches"],
        )
        wave_ids = {key: torch.cat([tok_a[key], tok_b[key]]) for key in ids}
        suffix_kv = [
            (
                torch.cat([torch.zeros_like(k), k], dim=0),
                torch.cat([torch.zeros_like(v), v], dim=0),
            )
            for k, v in parent_out["kv"]
        ]
        wave = model.forward_decode(
            new_token_batch=wave_ids,
            positions=torch.tensor([T, T + 1]),
            prefix_kv=prefill["kv_caches"],
            suffix_kv=suffix_kv,
            suffix_positions=torch.tensor([[0], [T]]),
            suffix_mask=torch.tensor([[False], [True]]),
        )
    torch.testing.assert_close(
        wave["logits"][0], full_a["logits"][T], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        wave["logits"][1], full_b["logits"][T + 1], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        wave["value_logits"][1], full_b["value_logits"][T + 1], atol=ATOL, rtol=RTOL
    )
