from __future__ import annotations

import contextlib
import math
from typing import Iterable

import torch

from ..model import create_batch_block_mask
from .metrics import (
    BatchCount,
    GameCount,
    NextMoveCrossEntropy,
    NextMoveMRR,
    NextMoveTokenCount,
    NextMoveTopKAccuracy,
    WeightedValueLoss,
    normalize_topk,
)

try:
    from ignite.engine import Engine, Events
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pytorch-ignite is required for Ignite evaluation. "
        "Install dependency 'pytorch-ignite'."
    ) from exc


def create_next_move_evaluator(
    *,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    ignore_index: int,
    topk: Iterable[int] = (1, 3, 5, 10),
    track_value_loss: bool = False,
) -> Engine:
    """track_value_loss: also run the model's loss path and report the
    weighted value_loss as metric "value_loss". The value head is what
    engine-eval finetunes actually train, and the policy metrics alone are
    blind to it. Requires a model with a value head."""
    resolved_topk = normalize_topk(topk)
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    @torch.inference_mode()
    def _eval_step(engine: Engine, batch: dict[str, object]) -> dict[str, object]:
        seq_offsets = batch["seq_offsets"].to(  # type: ignore[union-attr]
            device=device, dtype=torch.long, non_blocking=True
        )
        block_mask = create_batch_block_mask(
            seq_offsets=seq_offsets,
            total_tokens=int(batch["total_tokens"]),  # type: ignore[arg-type]
            device=device,
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            output = model(
                batch, block_mask=block_mask, return_loss=track_value_loss
            )

        logits = output["logits"].detach()
        targets = batch["target_move_id"].to(  # type: ignore[union-attr]
            device=logits.device, dtype=torch.long, non_blocking=True
        )
        result: dict[str, object] = {
            "y_pred": logits,
            "y": targets,
            "logits": logits,
            "targets": targets,
            "num_games": float(int(batch["num_games"])),  # type: ignore[arg-type]
        }
        if track_value_loss:
            if "value_loss" not in output:
                raise KeyError(
                    "track_value_loss=True but the model produced no value_loss "
                    "(enable_value_head is off?)"
                )
            result["value_loss"] = float(output["value_loss"].item())
            result["value_weight_sum"] = float(output["value_weight_sum"].item())
        return result

    evaluator = Engine(_eval_step)

    NextMoveCrossEntropy(ignore_index=ignore_index).attach(evaluator, "loss_ce")
    if track_value_loss:
        WeightedValueLoss().attach(evaluator, "value_loss")
    NextMoveMRR(ignore_index=ignore_index).attach(evaluator, "mrr")
    NextMoveTokenCount(ignore_index=ignore_index).attach(evaluator, "token_count")
    BatchCount().attach(evaluator, "batch_count")
    GameCount().attach(evaluator, "game_count")
    for k in resolved_topk:
        NextMoveTopKAccuracy(k=k, ignore_index=ignore_index).attach(
            evaluator, f"top{k}_acc"
        )

    @evaluator.on(Events.STARTED)
    def _set_eval_mode(engine: Engine) -> None:
        engine.state._was_training = bool(model.training)
        model.eval()

    @evaluator.on(Events.COMPLETED)
    def _add_derived_metrics(engine: Engine) -> None:
        loss_ce = float(engine.state.metrics["loss_ce"])
        engine.state.metrics["ppl"] = (
            float("nan") if math.isnan(loss_ce) else float(math.exp(loss_ce))
        )
        if bool(getattr(engine.state, "_was_training", False)):
            model.train()

    return evaluator
