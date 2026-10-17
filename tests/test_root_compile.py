"""CUDA root compilation preserves ragged outputs, KV ownership and refreshes."""

from dataclasses import replace

import pytest
import torch

from imba_chess.eval.merged_executors import _make_root_eval_executor
from imba_chess.model import HSTUChessModel
from tests.test_dense_attn_mask import _batch, _config


@pytest.mark.extended
@pytest.mark.skipif(not torch.cuda.is_available(), reason="root compilation needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_roots_match_eager_and_follow_weight_refresh(dtype):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = HSTUChessModel(replace(
        _config(num_layers=2), enable_value_head=True, enable_action_value_head=True,
    )).cuda().to(dtype).eval()
    state_keys = set(model.state_dict())
    kwargs = dict(model=model, device=torch.device("cuda"), dtype=dtype,
                  stats=None, max_tokens=16)
    eager = _make_root_eval_executor(**kwargs, compile_model=False)
    compiled = _make_root_eval_executor(**kwargs)
    assert compiled.compile_enabled is (dtype == torch.float32)
    assert not eager.compile_enabled
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (0., 0.)
    retained = None
    for lengths in ([5], [12], [7, 5], [3, 9, 4], [10, 7, 6], [5]):
        payloads = [_batch([length]) for length in lengths]
        expected, actual = eager(payloads), compiled(payloads)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        if retained is None:
            retained = actual
            saved = [x["logits"].clone() for x in retained]
        for output, snapshot in zip(retained, saved):
            torch.testing.assert_close(output["logits"], snapshot, atol=0, rtol=0)

    before = actual[0]["logits"].clone()
    with torch.no_grad():
        model.layers[-1]._ps_w.copy_(torch.randn_like(model.layers[-1]._ps_w) * 3)
    expected, actual = eager(payloads), compiled(payloads)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert not torch.equal(actual[0]["logits"], before)
    assert set(model.state_dict()) == state_keys
    assert not any(key.startswith("_orig_mod.") for key in state_keys)
