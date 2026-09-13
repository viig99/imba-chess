"""Production CUDA defaults and the explicit reference path remain distinct."""

from types import SimpleNamespace

import pytest
from imba_chess.self_play.config import SelfPlayConfig
from imba_chess.self_play import runtime


@pytest.mark.parametrize(
    "device,optimized,mode,expected",
    [("cpu", True, None, False), ("cuda", True, None, True),
     ("cuda", False, None, False), ("cuda", True, "current", True)],
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
