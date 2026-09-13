"""Parameter grouping shared by stage-1 and stage-2 trainers."""

from typing import Any
import torch


def build_decay_param_groups(
    model: torch.nn.Module, *, weight_decay: float
) -> list[dict[str, Any]]:
    """Decay only Linear weights; embeddings, norms, biases, and bare
    parameters (e.g. relative-position bias tables) get no decay.

    Sparsely-updated embedding rows otherwise shrink toward zero between the
    steps that actually touch them.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if isinstance(module, torch.nn.Linear) and param_name == "weight":
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
