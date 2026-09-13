"""Device-level parity for the single-row-per-game attention workload."""

import pytest
import torch
from imba_chess.model.hstu_attention import SequentialTransductionUnitJagged


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)])
@pytest.mark.parametrize("groups,suffix_len", [(1, 0), (3, 0), (8, 7), (12, 31)])
def test_one_query_attention_mixed_lengths(device, groups, suffix_len):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(731)
    layer = (
        SequentialTransductionUnitJagged(
            embedding_dim=128,
            linear_hidden_dim=32,
            attention_dim=32,
            dropout_ratio=0,
            num_heads=4,
            max_seq_len=512,
        )
        .to(device)
        .eval()
    )
    lengths = ([0, 1, 31, 120, 7, 256, 450, 510] * 2)[:groups]
    max_prefix = max(lengths)
    x = torch.randn(groups, 128, device=device)
    prefix_k = torch.randn(groups, 4, max_prefix, 32, device=device)
    prefix_v = torch.randn_like(prefix_k)
    positions = torch.tensor(lengths, device=device) + suffix_len
    suffix_k = suffix_v = suffix_positions = suffix_mask = None
    if suffix_len:
        suffix_k = torch.randn(groups, 4, suffix_len, 32, device=device)
        suffix_v = torch.randn_like(suffix_k)
        suffix_positions = torch.tensor(lengths, device=device)[:, None] + torch.arange(
            suffix_len, device=device
        )
        suffix_mask = torch.arange(suffix_len, device=device)[None, :] < (
            torch.arange(groups, device=device)[:, None] % (suffix_len + 1)
        )
    kwargs = dict(
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        prefix_lens_list=lengths,
        group_index=torch.arange(groups, device=device),
        row_idx_per_group=[torch.tensor([g], device=device) for g in range(groups)],
        q_positions=positions,
        suffix_k=suffix_k,
        suffix_v=suffix_v,
        suffix_positions=suffix_positions,
        suffix_mask=suffix_mask,
    )
    with torch.inference_mode():
        baseline = layer.forward_decode_grouped(x, **kwargs)
        candidate = layer.forward_decode_grouped(x, **kwargs, one_query_per_game=True)
    for actual, reference in zip(candidate, baseline):
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
