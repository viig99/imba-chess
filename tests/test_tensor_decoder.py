from dataclasses import asdict
import pytest
import torch
from imba_chess.eval.merged_executors import _MergedDecodeRequest
from tests.search_references import DecoderRunner
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


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.extended)]
)
@pytest.mark.parametrize(
    "mode", ["tensor", pytest.param("compiled", marks=pytest.mark.extended)]
)
def test_whole_decoder_branch_reuse(device, mode):
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("CUDA unavailable")
    if device == "cpu" and mode.startswith("compiled"):
        pytest.skip("Inductor performance/correctness checked on CUDA")
    torch.set_num_threads(4)
    model = _tiny_model().to(device).eval()
    runner = DecoderRunner(model, mode)
    with torch.inference_mode():
        merged = request(model, device)
        workspace = None
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
                    actual[key], expected[key], atol=1e-05, rtol=1e-05
                )
            for pair, ref in zip(actual["kv"], expected["kv"]):
                for a, b in zip(pair, ref):
                    torch.testing.assert_close(a, b, atol=1e-05, rtol=1e-05)
            if previous_kv is not None:
                torch.testing.assert_close(
                    previous_kv[0], previous_kv[1], atol=0, rtol=0
                )
            previous_kv = (actual["kv"][0][0], actual["kv"][0][0].clone())
        merged = request(model, device, groups=1, depth=3)
        kwargs = {name: getattr(merged, name) for name in asdict(merged)}
        expected = model.forward_decode_grouped(**kwargs, one_query_per_game=True)
        torch.testing.assert_close(
            runner(merged)["logits"], expected["logits"], atol=1e-05, rtol=1e-05
        )
        runner.clear()
        assert runner.workspace is None and runner.prefix_owner is None
        model.prediction_head.weight.mul_(1.1)
        expected = model.forward_decode_grouped(**kwargs, one_query_per_game=True)
        torch.testing.assert_close(
            runner(merged)["logits"], expected["logits"], atol=1e-05, rtol=1e-05
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
