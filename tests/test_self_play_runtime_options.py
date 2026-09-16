"""Production CUDA defaults and the explicit reference path remain distinct."""

from types import SimpleNamespace

import pytest
from imba_chess.self_play.config import SelfPlayConfig
from imba_chess.self_play import runtime


@pytest.mark.parametrize(
    "device,optimized,mode,expected",
    [
        ("cpu", True, None, False),
        ("cuda", True, None, True),
        ("cuda", False, None, False),
        ("cuda", True, "current", True),
    ],
)
def test_runtime_resolves_decoder_options_without_changing_checkpoint(
    monkeypatch, device, optimized, mode, expected
):
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=513))
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "load_hstu_checkpoint", lambda **kw: (model, None))
    monkeypatch.setattr(runtime, "InferenceRuntime", lambda **kw: SimpleNamespace(**kw))
    actual, positions = runtime.load_runtime(
        SelfPlayConfig(), "unused.pt", device, optimized=optimized, decoder_mode=mode
    )
    assert actual.model is model and positions == 513
    for option in (
        "one_query_per_game",
        "cache_prefixes",
        "batch_projection",
        "batch_inputs",
        "batch_suffix",
    ):
        assert getattr(actual, option) is expected
    assert actual.decoder_mode == (mode or ("compiled" if expected else "current"))
    assert actual.reuse_decode_buffers is (expected and mode != "current")
    assert actual.native_gumbel is actual.reuse_decode_buffers
    assert actual.history_cache_mode == (
        "direct" if actual.reuse_decode_buffers else "current"
    )


@pytest.mark.parametrize(
    "overrides,reuse,native",
    [
        ({"reuse_decode_buffers": False}, False, False),
        ({"native_gumbel": False}, True, False),
        ({"reuse_decode_buffers": False, "native_gumbel": True}, False, True),
        ({"decoder_mode": "sdpa"}, False, False),
        ({"decoder_mode": "compiled-sdpa"}, False, False),
        ({"decoder_mode": "tensor"}, True, True),
        *[
            ({option: False}, False, False)
            for option in (
                "one_query_per_game",
                "cache_prefixes",
                "batch_projection",
                "batch_inputs",
                "batch_suffix",
            )
        ],
    ],
)
def test_runtime_retains_reference_switches_and_ablation_options(
    monkeypatch, overrides, reuse, native
):
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=513))
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "load_hstu_checkpoint", lambda **kw: (model, None))
    monkeypatch.setattr(runtime, "InferenceRuntime", lambda **kw: SimpleNamespace(**kw))
    actual, _ = runtime.load_runtime(SelfPlayConfig(), "unused.pt", "cuda", **overrides)
    assert actual.reuse_decode_buffers is reuse
    assert actual.native_gumbel is native
    assert actual.history_cache_mode == ("direct" if reuse else "current")


@pytest.mark.parametrize("mode", ["current", "revision", "direct"])
def test_runtime_preserves_explicit_history_cache_mode(monkeypatch, mode):
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=513))
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "load_hstu_checkpoint", lambda **kw: (model, None))
    monkeypatch.setattr(runtime, "InferenceRuntime", lambda **kw: SimpleNamespace(**kw))
    actual, _ = runtime.load_runtime(
        SelfPlayConfig(), "unused.pt", "cuda", history_cache_mode=mode
    )
    assert actual.history_cache_mode == mode
