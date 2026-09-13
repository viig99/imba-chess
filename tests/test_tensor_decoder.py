from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from imba_chess.eval.merged_executors import _MergedDecodeRequest, _pack_prefixes
from imba_chess.model.tensor_decoder import DecoderRunner
from tests.test_prefix_decode import _random_token_ids, _tiny_model


def request(model, device, groups=3, depth=7):
    layers = model.layers
    prefix = [
        (
            torch.randn(groups, 2, 29, 8, device=device),
            torch.randn(groups, 2, 29, 8, device=device),
        )
        for _ in layers
    ]
    suffix = [
        (
            torch.randn(groups, 2, depth, 8, device=device),
            torch.randn(groups, 2, depth, 8, device=device),
        )
        for _ in layers
    ]
    lengths = ([1, 17, 29] * groups)[:groups]
    return _MergedDecodeRequest(
        _random_token_ids(groups, 42),
        torch.tensor(lengths) + depth,
        torch.arange(groups),
        prefix,
        torch.tensor(lengths),
        lengths,
        [1] * groups,
        suffix if depth else None,
        torch.tensor(lengths, device=device)[:, None]
        + torch.arange(depth, device=device)
        if depth
        else None,
        torch.ones(groups, depth, dtype=torch.bool, device=device) if depth else None,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)])
@pytest.mark.parametrize("mode", ["tensor", "sdpa",
    pytest.param("compiled", marks=pytest.mark.extended),
    pytest.param("compiled-sdpa", marks=pytest.mark.extended),
])
def test_whole_decoder_branch_reuse(device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cpu" and mode.startswith("compiled"):
        pytest.skip("Inductor performance/correctness checked on CUDA")
    torch.set_num_threads(4)
    model = _tiny_model().to(device).eval()
    runner = DecoderRunner(model, mode)
    with torch.inference_mode():
        merged = request(model, device)
        workspace = None
        if runner.sdpa:
            workspace = [
                (F.pad(k, (0, 0, 0, 33)), F.pad(v, (0, 0, 0, 33)))
                for k, v in merged.prefix_kv_grouped
            ]
            merged.prefix_kv_grouped = [
                (k[:, :, :-33], v[:, :, :-33]) for k, v in workspace
            ]
            for (k, _), (wk, _) in zip(merged.prefix_kv_grouped, workspace):
                assert k.data_ptr() == wk.data_ptr()
        previous_kv = None
        workspace_ptr = None
        original_prefix = [(k.clone(), v.clone()) for k, v in merged.prefix_kv_grouped]
        for depth in (7, 1, 0, 31):
            next_request = request(model, device, depth=depth)
            next_request.prefix_kv_grouped = merged.prefix_kv_grouped
            merged = next_request
            kwargs = {name: getattr(merged, name) for name in asdict(merged)}
            expected = model.forward_decode_grouped(**kwargs, one_query_per_game=True)
            actual = runner(merged, workspace=workspace)
            for current, saved in zip(merged.prefix_kv_grouped, original_prefix):
                for a, b in zip(current, saved):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            for key in ("logits", "value_logits"):
                torch.testing.assert_close(
                    actual[key], expected[key], atol=1e-5, rtol=1e-5
                )
            for pair, ref in zip(actual["kv"], expected["kv"]):
                for a, b in zip(pair, ref):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            if previous_kv is not None:
                torch.testing.assert_close(
                    previous_kv[0], previous_kv[1], atol=0, rtol=0
                )
            previous_kv = (actual["kv"][0][0], actual["kv"][0][0].clone())
            if runner.sdpa:
                pointer = runner.workspace[0][0].data_ptr()
                assert workspace_ptr is None or workspace_ptr == pointer
                workspace_ptr = pointer
        # Root/owner replacement must not reuse stale prefix content.
        merged = request(model, device, groups=1, depth=3)
        kwargs = {name: getattr(merged, name) for name in asdict(merged)}
        expected = model.forward_decode_grouped(**kwargs, one_query_per_game=True)
        torch.testing.assert_close(
            runner(merged)["logits"], expected["logits"], atol=1e-5, rtol=1e-5
        )
        runner.clear()
        assert runner.workspace is None and runner.prefix_owner is None
        # The same physical model is trained between collection phases. A
        # compiled callable must observe updated weights, not frozen constants.
        model.prediction_head.weight.mul_(1.1)
        expected = model.forward_decode_grouped(**kwargs, one_query_per_game=True)
        torch.testing.assert_close(
            runner(merged)["logits"], expected["logits"], atol=1e-5, rtol=1e-5
        )


def test_runner_rejects_training_model_without_changing_its_mode():
    model = _tiny_model().train()
    with pytest.raises(ValueError, match="evaluation-mode"):
        DecoderRunner(model, "tensor")
    assert model.training


def test_runner_rejects_oversized_suffix():
    model = _tiny_model().eval()
    with pytest.raises(ValueError, match="capacity"):
        DecoderRunner(model, "tensor")(request(model, "cpu", depth=33))


def test_reserved_prefix_packing_matches_reference_without_padding_temporaries():
    model = _tiny_model().eval()
    merged = request(model, "cpu")
    requests = [
        SimpleNamespace(
            prefix_len=length,
            prefix_kv=[
                (k[g, :, :length], v[g, :, :length])
                for k, v in merged.prefix_kv_grouped
            ],
        )
        for g, length in enumerate(merged.prefix_lens_list)
    ]
    reference = _pack_prefixes(requests)
    workspace = _pack_prefixes(requests, reserve_tokens=33)
    for pair, expected in zip(workspace, reference):
        for buffer, ref in zip(pair, expected):
            torch.testing.assert_close(buffer[:, :, :-33], ref, atol=0, rtol=0)
            assert torch.count_nonzero(buffer[:, :, -33:]) == 0
