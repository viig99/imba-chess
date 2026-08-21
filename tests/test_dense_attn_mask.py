"""Dense-mask SDPA inference path: equivalence with the flex_attention path.

Why this path exists: eager `flex_attention` costs ~95 ms per model forward
regardless of sequence length (pure dispatch overhead; see
docs/superpowers/notes/2026-08-20-rl-throughput-bottleneck.md). At inference
lengths a materialized `[1, H, S, S]` mask is ~12 MB, so SDPA with an explicit
additive mask does the same math 5-38x faster. Training keeps flex: there S is
~4,000 tokens (mask would be ~800 MB) and the batch is block-diagonal over ~50
games, which BlockMask skips and a dense mask would not.
"""

import pytest

torch = pytest.importorskip("torch")

from imba_chess.model import HSTUChessConfig, HSTUChessModel
from imba_chess.model.hstu_model import (
    create_batch_block_mask,
    create_batch_dense_mask,
)


def _config(*, num_layers: int) -> HSTUChessConfig:
    return HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=num_layers,
        max_position_embeddings=32,
    )


def _batch(seq_lens: list[int]) -> dict:
    offsets = [0]
    for n in seq_lens:
        offsets.append(offsets[-1] + n)
    total = offsets[-1]
    return {
        "game_id": [f"g{i}" for i in range(len(seq_lens))],
        "game_result_white": torch.zeros(len(seq_lens), dtype=torch.long),
        "num_games": len(seq_lens),
        "total_tokens": total,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "seq_offsets": torch.tensor(offsets, dtype=torch.long),
        "piece_ids": torch.zeros((total, 64), dtype=torch.long),
        "seq_token_id": torch.zeros(total, dtype=torch.long),
        "turn_id": torch.zeros(total, dtype=torch.long),
        "castle_id": torch.zeros(total, dtype=torch.long),
        "ep_file_id": torch.zeros(total, dtype=torch.long),
        "halfmove_bucket_id": torch.zeros(total, dtype=torch.long),
        "fullmove_bucket_id": torch.zeros(total, dtype=torch.long),
        "prev_move_id": torch.arange(total, dtype=torch.long) % 100,
        "target_move_id": torch.full((total,), -100, dtype=torch.long),
        "played_by_elo": torch.zeros(total, dtype=torch.long),
    }


def test_dense_mask_is_causal_within_each_document():
    """Token q may attend to k iff same game and k <= q."""
    mask = create_batch_dense_mask(
        torch.tensor([0, 3, 5]), total_tokens=5, device="cpu"
    )

    expected = torch.tensor(
        [
            # doc 0 = tokens 0,1,2 ; doc 1 = tokens 3,4
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_dense_mask_matches_block_mask_allowed_positions():
    """The dense mask must admit exactly the positions BlockMask admits.

    Guards the doc-boundary arithmetic against BlockMask's own mask_mod
    (prefix-LM with prefix 0, per-document) rather than against our own
    restatement of it.
    """
    seq_offsets = torch.tensor([0, 4, 7, 11])
    total = 11
    dense = create_batch_dense_mask(seq_offsets, total_tokens=total, device="cpu")

    block = create_batch_block_mask(
        seq_offsets, total_tokens=total, device="cpu"
    )
    # BlockMask.to_dense() is block-granular; compare the exact mask_mod instead.
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,  # noqa: F401  (import guard: API presence)
    )

    q = torch.arange(total).unsqueeze(1).expand(total, total)
    k = torch.arange(total).unsqueeze(0).expand(total, total)
    zero = torch.zeros_like(q)
    reference = block.mask_mod(zero, zero, q, k)

    assert torch.equal(dense, reference)


def test_dense_path_uses_each_layer_own_relative_position_bias():
    """Regression pin: `_ps_w` is per-layer, so the additive mask must be too.

    A first implementation built one mask from layers[0] and reused it for all
    layers. It ran, was fast, and silently changed move selection
    (max|dlogit| ~3.0, top-1 agreement 77-94%). This test fails for that bug:
    if layer 1's own `_ps_w` is ignored, perturbing it cannot change the
    output.
    """
    torch.manual_seed(0)
    model = HSTUChessModel(_config(num_layers=2))
    model.eval()
    batch = _batch([6])
    dense = create_batch_dense_mask(
        batch["seq_offsets"], total_tokens=batch["total_tokens"], device="cpu"
    )

    with torch.no_grad():
        before = model(batch, block_mask=dense, return_loss=False)["policy_logits"]
        # NON-UNIFORM on purpose: softmax is invariant to a constant added to
        # every score, so `_ps_w += 1.0` would leave the output untouched even
        # under a correct implementation and prove nothing.
        model.layers[1]._ps_w.copy_(torch.randn_like(model.layers[1]._ps_w) * 3.0)
        after = model(batch, block_mask=dense, return_loss=False)["policy_logits"]

    assert not torch.allclose(before, after), (
        "perturbing layer 1's _ps_w did not change the output -- the dense "
        "path is reusing another layer's relative-position bias"
    )


def test_dense_mask_forward_matches_flex_forward_on_cpu():
    """Same math, both paths: logits agree to fp32 round-off."""
    torch.manual_seed(0)
    model = HSTUChessModel(_config(num_layers=2))
    model.eval()
    batch = _batch([5, 4])
    total = batch["total_tokens"]

    flex = create_batch_block_mask(
        batch["seq_offsets"], total_tokens=total, device="cpu"
    )
    dense = create_batch_dense_mask(
        batch["seq_offsets"], total_tokens=total, device="cpu"
    )

    with torch.no_grad():
        a = model(batch, block_mask=flex, return_loss=False)["policy_logits"]
        b = model(batch, block_mask=dense, return_loss=False)["policy_logits"]

    assert torch.equal(a.argmax(-1), b.argmax(-1)), "move selection differs"
    assert (a - b).abs().max().item() < 1e-4


def test_dense_mask_returns_kv_caches_like_flex():
    """The root-eval path requests return_kv=True; both paths must supply it."""
    torch.manual_seed(0)
    model = HSTUChessModel(_config(num_layers=2))
    model.eval()
    batch = _batch([5])
    dense = create_batch_dense_mask(
        batch["seq_offsets"], total_tokens=batch["total_tokens"], device="cpu"
    )

    with torch.no_grad():
        out = model(batch, block_mask=dense, return_loss=False, return_kv=True)

    assert len(out["kv_caches"]) == 2
    for k, v in out["kv_caches"]:
        assert k.shape[-2] == batch["total_tokens"]
        assert v.shape[-2] == batch["total_tokens"]



def test_additive_mask_refuses_dataset_sized_batches():
    """A dense mask for a training-sized batch must fail loudly, not allocate.

    At the production shape (S=4096 tokens, 12 heads, fp32) the additive mask
    is 12 * 4096^2 * 4 B = 805 MiB *per layer*. Picking the dense path for a
    dataset-sized batch is a mistake, and this repo's rule is that infra
    mistakes fail loudly rather than quietly eat the GPU.
    """
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=12,
        num_layers=1,
        max_position_embeddings=32,
    )
    layer = HSTUChessModel(config).layers[0]
    allowed = torch.ones(4096, 4096, dtype=torch.bool)

    with pytest.raises(ValueError, match="dense attention mask would need"):
        layer._additive_mask(allowed, dtype=torch.float32)


def test_dense_mask_survives_torch_compile_across_shapes():
    """Compiled + varying shapes must work: eval runs with `compile = true`.

    This is the case flex_attention could NOT do. Under
    torch.compile(dynamic=True), Inductor fails to lower flex's sdpa_mask0 /
    sdpa_score0 subgraphs once a second distinct shape appears
    (InductorError: LoweringException), which is why
    generate_search_rollouts.py hardcodes compile_model=False. SDPA is a
    plain op with no mask subgraph, so the dense path compiles and
    generalizes -- pinned here because eval_vs_stockfish resolves
    `compile = true` from config and would otherwise regress silently.
    """
    torch.manual_seed(0)
    model = HSTUChessModel(_config(num_layers=2))
    model.eval()
    compiled = torch.compile(model, dynamic=True, fullgraph=False)

    # Varying token counts AND document counts, to force reshape/recompile.
    for seq_lens in ([5], [12], [7, 5], [3, 9, 4]):
        batch = _batch(seq_lens)
        dense = create_batch_dense_mask(
            batch["seq_offsets"], total_tokens=batch["total_tokens"], device="cpu"
        )
        with torch.no_grad():
            eager = model(batch, block_mask=dense, return_loss=False)
            comp = compiled(batch, block_mask=dense, return_loss=False)

        a, b = eager["policy_logits"], comp["policy_logits"]
        assert torch.equal(a.argmax(-1), b.argmax(-1)), f"argmax differs at {seq_lens}"
        assert (a - b).abs().max().item() < 1e-4, f"logits differ at {seq_lens}"