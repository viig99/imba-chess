from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


def normalize_topk(topk: Iterable[int]) -> tuple[int, ...]:
    values = sorted({int(k) for k in topk if int(k) > 0})
    if not values:
        raise ValueError("topk must contain at least one positive integer")
    return tuple(values)


class _BaseNextMoveMetric(Metric):
    required_output_keys = ("logits", "targets")

    def __init__(
        self,
        *,
        ignore_index: int,
        output_transform=lambda x: x,
    ) -> None:
        self.ignore_index = int(ignore_index)
        super().__init__(output_transform=output_transform)

    @staticmethod
    def _unpack_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(output, dict):
            logits = output.get("logits", output.get("y_pred"))
            targets = output.get("targets", output.get("y"))
            if logits is None:
                raise KeyError("Expected output['logits'] or output['y_pred']")
            if targets is None:
                raise KeyError("Expected output['targets'] or output['y']")
        elif isinstance(output, (tuple, list)) and len(output) == 2:
            logits, targets = output
        else:
            raise TypeError(
                "Expected evaluator output to be dict or (predictions, targets) tuple"
            )
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Predictions must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Targets must be a torch.Tensor")
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, V], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets must have shape [N], got {tuple(targets.shape)}")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "logits/targets first dimension mismatch: "
                f"{logits.shape[0]} vs {targets.shape[0]}"
            )
        return logits, targets

    def _valid_mask(self, targets: torch.Tensor) -> torch.Tensor:
        return targets != self.ignore_index


class NextMoveCrossEntropy(_BaseNextMoveMetric):
    """Mean CE over non-ignore-index tokens."""

    @reinit__is_reduced
    def reset(self) -> None:
        self._loss_sum = 0.0
        self._token_count = 0.0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        logits, targets = self._unpack_output(output)
        valid_mask = self._valid_mask(targets)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            raise ValueError(
                "NextMoveCrossEntropy received a batch with no valid targets. "
                "All targets are ignore_index."
            )
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        loss_sum = F.cross_entropy(valid_logits.float(), valid_targets, reduction="sum")
        self._loss_sum += float(loss_sum.item())
        self._token_count += float(valid_count)

    @sync_all_reduce("_loss_sum", "_token_count")
    def compute(self) -> float:
        if self._token_count == 0.0:
            return float("nan")
        return self._loss_sum / self._token_count


class NextMoveTopKAccuracy(_BaseNextMoveMetric):
    """Top-k accuracy over non-ignore-index tokens."""

    def __init__(self, *, k: int, ignore_index: int, output_transform=lambda x: x) -> None:
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        super().__init__(ignore_index=ignore_index, output_transform=output_transform)

    @reinit__is_reduced
    def reset(self) -> None:
        self._correct = 0.0
        self._token_count = 0.0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        logits, targets = self._unpack_output(output)
        valid_mask = self._valid_mask(targets)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            raise ValueError(
                "NextMoveTopKAccuracy received a batch with no valid targets. "
                "All targets are ignore_index."
            )

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        vocab_size = int(valid_logits.shape[1])
        k_eff = min(self.k, vocab_size)
        pred_ids = valid_logits.topk(k_eff, dim=-1).indices
        hits = (pred_ids == valid_targets.unsqueeze(1)).any(dim=1).sum()

        self._correct += float(hits.item())
        self._token_count += float(valid_count)

    @sync_all_reduce("_correct", "_token_count")
    def compute(self) -> float:
        if self._token_count == 0.0:
            return float("nan")
        return self._correct / self._token_count


class NextMoveMRR(_BaseNextMoveMetric):
    """Mean reciprocal rank over non-ignore-index tokens."""

    @reinit__is_reduced
    def reset(self) -> None:
        self._rr_sum = 0.0
        self._token_count = 0.0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        logits, targets = self._unpack_output(output)
        valid_mask = self._valid_mask(targets)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            raise ValueError(
                "NextMoveMRR received a batch with no valid targets. "
                "All targets are ignore_index."
            )

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        target_col = valid_targets.unsqueeze(1)
        target_logits = valid_logits.gather(1, target_col).squeeze(1)
        rank = (
            (valid_logits > target_logits.unsqueeze(1)).sum(dim=1).to(torch.float32) + 1.0
        )
        self._rr_sum += float((1.0 / rank).sum().item())
        self._token_count += float(valid_count)

    @sync_all_reduce("_rr_sum", "_token_count")
    def compute(self) -> float:
        if self._token_count == 0.0:
            return float("nan")
        return self._rr_sum / self._token_count


class NextMoveTokenCount(_BaseNextMoveMetric):
    """Count of valid (non-ignore-index) tokens."""

    @reinit__is_reduced
    def reset(self) -> None:
        self._token_count = 0.0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        _, targets = self._unpack_output(output)
        self._token_count += float(self._valid_mask(targets).sum().item())

    @sync_all_reduce("_token_count")
    def compute(self) -> float:
        return self._token_count


class BatchCount(Metric):
    required_output_keys = ("num_games",)

    @reinit__is_reduced
    def reset(self) -> None:
        self._count = 0.0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        self._count += 1.0

    @sync_all_reduce("_count")
    def compute(self) -> float:
        return self._count


class GameCount(Metric):
    required_output_keys = ("num_games",)

    @reinit__is_reduced
    def reset(self) -> None:
        self._count = 0.0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        if isinstance(output, dict):
            num_games = output.get("num_games")
            if num_games is None:
                raise KeyError("Expected output['num_games']")
        elif isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError(
                    "Expected single-value tuple/list for num_games metric output"
                )
            num_games = output[0]
        else:
            num_games = output
        if isinstance(num_games, torch.Tensor):
            if num_games.numel() != 1:
                raise ValueError("num_games tensor must be scalar")
            num_games = num_games.item()
        self._count += float(num_games)

    @sync_all_reduce("_count")
    def compute(self) -> float:
        return self._count
