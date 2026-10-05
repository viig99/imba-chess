"""Strict model-weight loading for raw, distributed and compiled checkpoints."""
from collections.abc import Mapping

import torch


def normalized_model_state(checkpoint: Mapping) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain a model state dictionary")
    result = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint state_dict keys must be strings")
        key = key.removeprefix("module.").removeprefix("_orig_mod.")
        if key in result:
            raise ValueError("Checkpoint has colliding compiled/uncompiled keys")
        result[key] = value
    return result


def load_initial_weights(
    model: torch.nn.Module, checkpoint: Mapping, *, allow_missing_prefixes=()
) -> None:
    """Load compatible weights without altering shapes or optimizer state.

    `allow_missing_prefixes` names freshly added heads (e.g. KLENT's
    "action_value_head.") that a checkpoint may lack; every other missing or
    unexpected key still raises.
    """
    state = normalized_model_state(checkpoint)
    target = getattr(model, "_orig_mod", model)
    if not allow_missing_prefixes:
        target.load_state_dict(state, strict=True)
        return
    result = target.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(tuple(allow_missing_prefixes))]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing} unexpected={result.unexpected_keys}"
        )
