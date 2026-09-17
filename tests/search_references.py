"""CPU-only construction for mathematical/runtime tests, never a production option."""

from collections import Counter
import torch
from unittest.mock import patch
from imba_chess.model.tensor_decoder import (
    DecoderRunner as CompiledRunner,
    TensorDecoder,
)
from imba_chess.eval.inference_runtime import InferenceRuntime as SearchRuntime
from imba_chess.eval.merged_executors import (
    _make_root_eval_executor,
    _make_decode_wave_executor,
)


def DecoderRunner(model, mode="tensor", suffix_capacity=32):
    if mode == "compiled":
        return CompiledRunner(model, suffix_capacity=suffix_capacity)
    if model.training:
        raise ValueError("tensor decoder requires an evaluation-mode model")
    runner = object.__new__(CompiledRunner)
    runner.model = model
    runner.suffix_capacity = suffix_capacity
    runner.decode = TensorDecoder(model).eval()
    runner.clear()
    return runner


def InferenceRuntime(
    *,
    model,
    move_vocab,
    encoder,
    device,
    root_batch_tokens=1024,
    reuse_decode_buffers=True,
    algorithm="gumbel",
    **reference_options,
):
    runtime = object.__new__(SearchRuntime)
    runtime.model, runtime.move_vocab, runtime.encoder, runtime.device = (
        model,
        move_vocab,
        encoder,
        torch.device("cpu"),
    )
    runtime.algorithm = algorithm
    runtime._cache_token = object()
    runtime.options = dict(
        algorithm="gumbel",
        dtype="float32",
        tf32=False,
        runtime_revision="test-reference",
    )
    runtime.waves = dict(root_eval=Counter(), decode_wave=Counter())
    runtime.seconds = Counter()
    runtime.inference_rows = Counter()
    with patch("imba_chess.model.tensor_decoder.DecoderRunner", DecoderRunner):
        root = _make_root_eval_executor(
            model=model,
            device=runtime.device,
            dtype=torch.float32,
            stats=None,
            max_tokens=root_batch_tokens,
        )
        leaf = _make_decode_wave_executor(
            model=model,
            device=runtime.device,
            dtype=torch.float32,
            stats=None,
            algorithm="gumbel"
            if algorithm == "gumbel" and reuse_decode_buffers
            else "value_search_halving",
        )
    runtime.executors = {
        kind: runtime._identified(kind, fn)
        for kind, fn in [("root_eval", root), ("decode_wave", leaf)]
    }
    return runtime
