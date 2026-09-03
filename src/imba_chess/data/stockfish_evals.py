from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_TO_WHITE_SCORE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def eval_from_comment(comment: str) -> tuple[float | None, int | None]:
    """Return a Lichess ``[%eval]`` as White-POV centipawns or mate count."""
    start = comment.find("[%eval ")
    if start < 0:
        return None, None
    end = comment.find("]", start)
    if end < 0:
        return None, None
    token = comment[start + 7 : end].strip()
    if token.startswith("#"):
        try:
            return None, int(token[1:])
        except ValueError:
            return None, None
    try:
        return float(token) * 100.0, None
    except ValueError:
        return None, None


def _normalise_wdl(
    values: tuple[float, float, float], *, label: str
) -> tuple[float, float, float]:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite probability")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{label} probabilities must be in [0, 1]")
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} probabilities must have a positive finite sum")
    return (values[0] / total, values[1] / total, values[2] / total)


def _read_wdl(record: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    try:
        values = (
            float(record["p_loss"]),
            float(record["p_draw"]),
            float(record["p_win"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric p_loss/p_draw/p_win") from exc
    _normalise_wdl(values, label=label)
    return values


@dataclass(frozen=True)
class CpToWdlCalibration:
    """Scalar, worker-pickleable port of the corpus cp/mate -> WDL calibration."""

    centers: tuple[float, ...]
    probs: tuple[tuple[float, float, float], ...]
    mate_probs: dict[int, tuple[float, float, float]]

    @classmethod
    def load(cls, path: str | Path) -> "CpToWdlCalibration":
        calibration_path = Path(path)
        try:
            raw = json.loads(calibration_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed calibration JSON: {calibration_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Calibration root must be an object")

        cp_bins = raw.get("cp_bins")
        if not isinstance(cp_bins, list) or len(cp_bins) < 2:
            raise ValueError("Calibration cp_bins must contain at least two bins")
        parsed_bins: list[tuple[float, tuple[float, float, float]]] = []
        for index, record in enumerate(cp_bins):
            if not isinstance(record, dict) or "cp_center" not in record:
                raise ValueError(f"cp_bins[{index}] must contain cp_center")
            try:
                center = float(record["cp_center"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"cp_bins[{index}].cp_center must be numeric") from exc
            if not math.isfinite(center):
                raise ValueError(f"cp_bins[{index}].cp_center must be finite")
            parsed_bins.append((center, _read_wdl(record, label=f"cp_bins[{index}]")))
        parsed_bins.sort(key=lambda item: item[0])
        centers = tuple(center for center, _ in parsed_bins)
        if any(right <= left for left, right in zip(centers, centers[1:])):
            raise ValueError("Calibration cp_center values must be unique")

        mate = raw.get("mate")
        if not isinstance(mate, dict):
            raise ValueError("Calibration mate must be an object")
        if set(mate) != {"-1", "1"}:
            raise ValueError("Calibration mate must contain exactly the -1 and 1 buckets")
        mate_probs = {
            sign: _read_wdl(mate[str(sign)], label=f"mate[{sign}]")
            for sign in (-1, 1)
        }
        return cls(
            centers=centers,
            probs=tuple(probabilities for _, probabilities in parsed_bins),
            mate_probs=mate_probs,
        )

    def wdl(
        self,
        cp_stm: float | None,
        mate_stm: int | None,
    ) -> tuple[float, float, float]:
        if (cp_stm is None) == (mate_stm is None):
            raise ValueError("Exactly one of cp_stm and mate_stm must be set")
        if mate_stm is not None:
            if mate_stm == 0:
                raise ValueError("mate_stm must be non-zero")
            return _normalise_wdl(
                self.mate_probs[1 if mate_stm > 0 else -1],
                label="mate WDL",
            )

        cp = float(cp_stm)
        if not math.isfinite(cp):
            raise ValueError("cp_stm must be finite")
        if cp <= self.centers[0]:
            values = self.probs[0]
        elif cp >= self.centers[-1]:
            values = self.probs[-1]
        else:
            right = bisect.bisect_right(self.centers, cp)
            left = right - 1
            lo = self.centers[left]
            hi = self.centers[right]
            fraction = (cp - lo) / (hi - lo)
            values = tuple(
                self.probs[left][index]
                + fraction * (self.probs[right][index] - self.probs[left][index])
                for index in range(3)
            )
        return _normalise_wdl(values, label="interpolated WDL")
