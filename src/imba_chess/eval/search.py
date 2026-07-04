"""Move-selection strategies for eval play, decoupled from the model.

Strategies consume a PositionEvaluator: `handle` is opaque (the eval script
uses a _SequenceHistory clone; tests use whatever they need), `extend` derives
the handle for the position after a move, and `evaluate` batch-scores
positions, returning the value-head scalar (side-to-move POV) plus the legal
moves that map to the move vocab and their log-softmax policy priors.

This module must stay torch-free so strategy unit tests need no model.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Protocol

import chess


class PositionEval(NamedTuple):
    value_stm: float
    legal_moves: list[chess.Move]
    legal_log_priors: list[float]


class PositionEvaluator(Protocol):
    def extend(
        self, handle: Any, board_before: chess.Board, move: chess.Move
    ) -> Any: ...

    def evaluate(
        self, batch: list[tuple[Any, chess.Board]]
    ) -> list[PositionEval]: ...


@dataclass(frozen=True)
class HalvingConfig:
    budget: int = 256
    top_m: int = 16
    rounds: int = 0  # 0 = auto ceil(log2(num_arms))
    refutation_top_r: int = 2
    expand_top: int = 3
    max_depth: int = 4
    lam: float = 0.05


def _auto_rounds(num_arms: int) -> int:
    return max(1, math.ceil(math.log2(max(2, num_arms))))


def terminal_value_for_color(
    board: chess.Board, *, color: chess.Color
) -> Optional[float]:
    if not board.is_game_over(claim_draw=True):
        return None
    result = board.result(claim_draw=True)
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if color == chess.BLACK else -1.0
    return 0.0


def select_greedy(legal_log_priors: list[float]) -> int:
    return max(range(len(legal_log_priors)), key=legal_log_priors.__getitem__)


def _is_forcing(board: chess.Board, move: chess.Move) -> bool:
    return (
        move.promotion is not None
        or board.is_capture(move)
        or board.gives_check(move)
    )


def _prior_order(legal_log_priors: list[float]) -> list[int]:
    return sorted(
        range(len(legal_log_priors)),
        key=legal_log_priors.__getitem__,
        reverse=True,
    )
