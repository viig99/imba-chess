from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_policy_kl_loss.py"
    spec = importlib.util.spec_from_file_location("eval_policy_kl_loss_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_policy_kl_loss.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_is_holdout_row_is_deterministic():
    module = _load_script_module()
    a = module._is_holdout_row("game1", 5, holdout_fraction=0.1)
    b = module._is_holdout_row("game1", 5, holdout_fraction=0.1)
    assert a == b


def test_is_holdout_row_varies_by_game_id_and_ply():
    module = _load_script_module()
    results = {
        module._is_holdout_row(f"game{i}", i % 7, holdout_fraction=0.1) for i in range(200)
    }
    # With 200 distinct (game_id, ply) pairs at holdout_fraction=0.1, both
    # True and False must appear -- not a constant function.
    assert results == {True, False}


def test_is_holdout_row_fraction_zero_never_holds_out():
    module = _load_script_module()
    assert all(
        not module._is_holdout_row(f"game{i}", i, holdout_fraction=0.0) for i in range(50)
    )


def test_is_holdout_row_fraction_one_always_holds_out():
    module = _load_script_module()
    assert all(
        module._is_holdout_row(f"game{i}", i, holdout_fraction=1.0) for i in range(50)
    )
