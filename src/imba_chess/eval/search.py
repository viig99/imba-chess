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


def select_value_rerank(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
    lam: float,
) -> tuple[int, list[dict[str, Any]]]:
    root_color = board.turn
    local_indices = _prior_order(legal_log_priors)[: min(top_k, len(legal_moves))]

    candidates: list[dict[str, Any]] = []
    batch: list[tuple[Any, chess.Board]] = []
    batch_to_candidate: list[int] = []
    for idx in local_indices:
        candidate_move = legal_moves[idx]
        # Keep the move stack so claimable draws (repetition/50-move) count as terminal.
        next_board = board.copy()
        next_board.push(candidate_move)
        # Terminal boards never appear as training tokens; use the exact game
        # result instead of the value head there.
        terminal_value = terminal_value_for_color(next_board, color=root_color)
        candidates.append(
            {
                "local_idx": idx,
                "move": candidate_move,
                "terminal": terminal_value is not None,
                "value_root": terminal_value,
            }
        )
        if terminal_value is not None:
            continue
        batch.append((evaluator.extend(root_handle, board, candidate_move), next_board))
        batch_to_candidate.append(len(candidates) - 1)

    if batch:
        for cand_idx, position_eval in zip(batch_to_candidate, evaluator.evaluate(batch)):
            # Side-to-move at next_board is the opponent; negate to root POV.
            candidates[cand_idx]["value_root"] = -position_eval.value_stm

    chosen_index = local_indices[0]
    best_score = float("-inf")
    rerank_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        local_idx = int(candidate["local_idx"])
        value_root = float(candidate["value_root"])
        log_prior = float(legal_log_priors[local_idx])
        # Value-dominant score with a small log-prob policy prior as tiebreak.
        rerank_score = value_root + (lam * log_prior)
        rerank_rows.append(
            {
                "move_uci": candidate["move"].uci(),
                "policy_logit": log_prior,
                "policy_log_prob": log_prior,
                "value_next": value_root,
                "terminal": bool(candidate["terminal"]),
                "rerank_score": rerank_score,
            }
        )
        if rerank_score > best_score:
            best_score = rerank_score
            chosen_index = local_idx

    return chosen_index, rerank_rows


def select_value_search_d2(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
    lam: float,
) -> tuple[int, list[dict[str, Any]]]:
    root_color = board.turn
    local_indices = _prior_order(legal_log_priors)[: min(top_k, len(legal_moves))]

    root_candidates: list[dict[str, Any]] = []
    board1_batch: list[tuple[Any, chess.Board]] = []
    board1_batch_to_root: list[int] = []
    for local_idx in local_indices:
        move = legal_moves[local_idx]
        board1 = board.copy()
        board1.push(move)
        terminal_value = terminal_value_for_color(board1, color=root_color)
        if terminal_value is not None and terminal_value >= 1.0:
            # Immediate win (checkmate delivered): no other move can score higher.
            return local_idx, [
                {
                    "move_uci": move.uci(),
                    "policy_logit": float(legal_log_priors[local_idx]),
                    "policy_log_prob": float(legal_log_priors[local_idx]),
                    "worst_reply_value": 1.0,
                    "best_reply_uci": None,
                    "search_score": 1.0,
                }
            ]
        root_candidate: dict[str, Any] = {
            "local_idx": local_idx,
            "move": move,
            "board1": board1,
            "log_prior": float(legal_log_priors[local_idx]),
            "terminal_value": terminal_value,
            "board1_eval": None,
            "handle1": None,
            "reply_candidates": [],
        }
        root_candidates.append(root_candidate)
        if terminal_value is not None:
            continue
        handle1 = evaluator.extend(root_handle, board, move)
        root_candidate["handle1"] = handle1
        board1_batch.append((handle1, board1))
        board1_batch_to_root.append(len(root_candidates) - 1)

    if board1_batch:
        for root_idx, position_eval in zip(
            board1_batch_to_root, evaluator.evaluate(board1_batch)
        ):
            root_candidates[root_idx]["board1_eval"] = position_eval

    board2_batch: list[tuple[Any, chess.Board]] = []
    board2_meta: list[tuple[int, int]] = []
    for root_idx, root_candidate in enumerate(root_candidates):
        if root_candidate["terminal_value"] is not None:
            continue
        board1_eval: Optional[PositionEval] = root_candidate["board1_eval"]
        if board1_eval is None:
            continue
        board1 = root_candidate["board1"]
        if not board1_eval.legal_moves:
            root_candidate["worst_reply_value"] = -float(board1_eval.value_stm)
            continue

        opp_indices = _prior_order(board1_eval.legal_log_priors)[
            : min(top_k, len(board1_eval.legal_moves))
        ]
        # Always consider forcing replies (captures/checks/promotions): the
        # tactical refutation is often a low-probability move under a
        # human-imitation policy, so policy top-k alone misses it.
        opp_seen = set(opp_indices)
        for opp_idx, opp_move in enumerate(board1_eval.legal_moves):
            if opp_idx in opp_seen:
                continue
            if _is_forcing(board1, opp_move):
                opp_indices.append(opp_idx)
                opp_seen.add(opp_idx)

        reply_rows: list[dict[str, Any]] = []
        for opp_local_idx in opp_indices:
            opp_move = board1_eval.legal_moves[opp_local_idx]
            board2 = board1.copy()
            board2.push(opp_move)
            terminal_value = terminal_value_for_color(board2, color=root_color)
            reply_rows.append(
                {
                    "move_uci": opp_move.uci(),
                    "opp_policy_logit": float(
                        board1_eval.legal_log_priors[opp_local_idx]
                    ),
                    "value_after_reply": terminal_value,
                    "terminal": terminal_value is not None,
                }
            )
            if terminal_value is not None:
                continue
            board2_batch.append(
                (evaluator.extend(root_candidate["handle1"], board1, opp_move), board2)
            )
            board2_meta.append((root_idx, len(reply_rows) - 1))
        root_candidate["reply_candidates"] = reply_rows

    if board2_batch:
        for (root_idx, reply_idx), position_eval in zip(
            board2_meta, evaluator.evaluate(board2_batch)
        ):
            # Side-to-move at board2 is the root color again: POV matches root.
            root_candidates[root_idx]["reply_candidates"][reply_idx][
                "value_after_reply"
            ] = float(position_eval.value_stm)

    chosen_index = local_indices[0]
    best_score = float("-inf")
    search_rows: list[dict[str, Any]] = []
    for root_candidate in root_candidates:
        if root_candidate["terminal_value"] is not None:
            worst_reply_value = float(root_candidate["terminal_value"])
            best_reply_uci = None
        elif "worst_reply_value" in root_candidate:
            worst_reply_value = float(root_candidate["worst_reply_value"])
            best_reply_uci = None
        else:
            reply_rows = [
                row
                for row in root_candidate["reply_candidates"]
                if row.get("value_after_reply") is not None
            ]
            if not reply_rows:
                board1_eval = root_candidate.get("board1_eval")
                worst_reply_value = (
                    -float(board1_eval.value_stm) if board1_eval is not None else 0.0
                )
                best_reply_uci = None
            else:
                best_reply = min(
                    reply_rows, key=lambda row: float(row["value_after_reply"])
                )
                worst_reply_value = float(best_reply["value_after_reply"])
                best_reply_uci = str(best_reply["move_uci"])

        log_prior = float(root_candidate["log_prior"])
        # Value-dominant score with a small log-prob policy prior as tiebreak.
        search_score = worst_reply_value + (lam * log_prior)
        search_rows.append(
            {
                "move_uci": root_candidate["move"].uci(),
                "policy_logit": log_prior,
                "policy_log_prob": log_prior,
                "worst_reply_value": worst_reply_value,
                "best_reply_uci": best_reply_uci,
                "search_score": search_score,
            }
        )
        if search_score > best_score:
            best_score = search_score
            chosen_index = int(root_candidate["local_idx"])

    return chosen_index, search_rows
