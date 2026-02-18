from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from ignite.metrics import Metric


def normalize_topk(topk: Iterable[int]) -> tuple[int, ...]:
    values = sorted({int(k) for k in topk if int(k) > 0})
    if not values:
        raise ValueError("topk must contain at least one positive integer")
    return tuple(values)


class _BaseNextMoveMetric(Metric):
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
        if not isinstance(output, dict):
            raise TypeError("Expected evaluator output to be a dict")
        logits = output["logits"]
        targets = output["targets"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("output['logits'] must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("output['targets'] must be a torch.Tensor")
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

    def reset(self) -> None:
        self._loss_sum = 0.0
        self._token_count = 0.0
        super().reset()

    def update(self, output: Any) -> None:
        logits, targets = self._unpack_output(output)
        valid_mask = self._valid_mask(targets)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        loss_sum = F.cross_entropy(valid_logits, valid_targets, reduction="sum")
        self._loss_sum += float(loss_sum.item())
        self._token_count += float(valid_count)

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

    def reset(self) -> None:
        self._correct = 0.0
        self._token_count = 0.0
        super().reset()

    def update(self, output: Any) -> None:
        logits, targets = self._unpack_output(output)
        valid_mask = self._valid_mask(targets)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        vocab_size = int(valid_logits.shape[1])
        k_eff = min(self.k, vocab_size)
        pred_ids = valid_logits.topk(k_eff, dim=-1).indices
        hits = (pred_ids == valid_targets.unsqueeze(1)).any(dim=1).sum()

        self._correct += float(hits.item())
        self._token_count += float(valid_count)

    def compute(self) -> float:
        if self._token_count == 0.0:
            return float("nan")
        return self._correct / self._token_count


class NextMoveMRR(_BaseNextMoveMetric):
    """Mean reciprocal rank over non-ignore-index tokens."""

    def reset(self) -> None:
        self._rr_sum = 0.0
        self._token_count = 0.0
        super().reset()

    def update(self, output: Any) -> None:
        logits, targets = self._unpack_output(output)
        valid_mask = self._valid_mask(targets)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        target_col = valid_targets.unsqueeze(1)
        target_logits = valid_logits.gather(1, target_col).squeeze(1)
        rank = (
            (valid_logits > target_logits.unsqueeze(1)).sum(dim=1).to(torch.float32) + 1.0
        )
        self._rr_sum += float((1.0 / rank).sum().item())
        self._token_count += float(valid_count)

    def compute(self) -> float:
        if self._token_count == 0.0:
            return float("nan")
        return self._rr_sum / self._token_count


class NextMoveTokenCount(_BaseNextMoveMetric):
    """Count of valid (non-ignore-index) tokens."""

    def reset(self) -> None:
        self._token_count = 0.0
        super().reset()

    def update(self, output: Any) -> None:
        _, targets = self._unpack_output(output)
        self._token_count += float(self._valid_mask(targets).sum().item())

    def compute(self) -> float:
        return self._token_count


class BatchCount(Metric):
    def reset(self) -> None:
        self._count = 0.0
        super().reset()

    def update(self, output: Any) -> None:
        self._count += 1.0

    def compute(self) -> float:
        return self._count


class GameCount(Metric):
    def reset(self) -> None:
        self._count = 0.0
        super().reset()

    def update(self, output: Any) -> None:
        if not isinstance(output, dict):
            raise TypeError("Expected evaluator output to be a dict")
        num_games = output.get("num_games")
        if num_games is None:
            raise KeyError("Expected output['num_games']")
        self._count += float(num_games)

    def compute(self) -> float:
        return self._count
