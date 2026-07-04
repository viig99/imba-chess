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
    # A repetition/50-move claim needs >= 8 reversible plies of history for
    # the third occurrence, and python-chess allows claiming one reversible
    # ply early via a move that reaches it — so below halfmove_clock 7 no
    # claim is possible and the O(stack) repetition scan can be skipped
    # (~20x on this hot path; verified against the unguarded version on 28k
    # random positions).
    outcome = board.outcome(claim_draw=board.halfmove_clock >= 7)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == color else -1.0


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


def _search_copy(board: chess.Board) -> chess.Board:
    # Search only needs enough move-stack history for draw-claim detection
    # (bounded by halfmove_clock); copying a late-game full stack is ~150x
    # slower for no benefit.
    return board.copy(stack=board.halfmove_clock)


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
        next_board = _search_copy(board)
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
        board1 = _search_copy(board)
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
            "worst_reply_value": None,
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
            board2 = _search_copy(board1)
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
        elif root_candidate["worst_reply_value"] is not None:
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


@dataclass
class _TreeNode:
    board: chess.Board
    handle: Any
    depth: int  # plies below the arm root (arm root = 0)
    path_log_prior: float
    value_stm: Optional[float] = None  # set when evaluated by the value head
    terminal_value_stm: Optional[float] = None  # exact, side-to-move POV
    children: list["_TreeNode"] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.value_stm is not None or self.terminal_value_stm is not None


@dataclass
class _Arm:
    local_idx: int
    move: chess.Move
    root_log_prior: float
    root_node: Optional[_TreeNode]
    terminal_value_root: Optional[float]
    frontier: list = field(default_factory=list)
    evals_spent: int = 0
    max_depth_reached: int = 0
    eliminated_round: Optional[int] = None
    backed_value: Optional[float] = None
    score: float = float("-inf")


def _backed_stm(node: _TreeNode) -> float:
    """Negamax over the realized (partially scored) tree, side-to-move POV."""
    if node.terminal_value_stm is not None:
        return node.terminal_value_stm
    child_values = [-_backed_stm(child) for child in node.children if child.scored]
    if child_values:
        return max(child_values)
    assert node.value_stm is not None
    return node.value_stm


def _score_arm(arm: _Arm, lam: float) -> None:
    if arm.terminal_value_root is not None:
        backed_root = float(arm.terminal_value_root)
    elif arm.root_node is not None and arm.root_node.scored:
        backed_root = -_backed_stm(arm.root_node)
    else:
        arm.backed_value = None
        arm.score = float("-inf")
        return
    arm.backed_value = backed_root
    arm.score = backed_root + lam * arm.root_log_prior


def _push_children(
    arm: _Arm,
    node: _TreeNode,
    position_eval: PositionEval,
    evaluator: PositionEvaluator,
    config: HalvingConfig,
    counter: "itertools.count",
    root_color: chess.Color,
) -> None:
    if node.depth >= config.max_depth or not position_eval.legal_moves:
        return
    opponent_to_move = node.board.turn != root_color
    order = _prior_order(position_eval.legal_log_priors)
    if opponent_to_move:
        # Refutation floor: top-r replies by prior plus ALL forcing replies.
        picks = list(order[: config.refutation_top_r])
        seen = set(picks)
        for idx, move in enumerate(position_eval.legal_moves):
            if idx not in seen and _is_forcing(node.board, move):
                picks.append(idx)
                seen.add(idx)
    else:
        picks = list(order[: config.expand_top])

    for idx in picks:
        move = position_eval.legal_moves[idx]
        child_board = _search_copy(node.board)
        child_board.push(move)
        # Forcing replies inherit the parent's priority (no decay for their
        # own low prior): a refutation must compete at the plausibility of
        # the line it refutes, not of the reply itself.
        floor_pick = opponent_to_move and _is_forcing(node.board, move)
        child_prior = node.path_log_prior + (
            0.0 if floor_pick else position_eval.legal_log_priors[idx]
        )
        child = _TreeNode(
            board=child_board,
            handle=None,
            depth=node.depth + 1,
            path_log_prior=child_prior,
        )
        terminal_stm = terminal_value_for_color(child_board, color=child_board.turn)
        if terminal_stm is not None:
            child.terminal_value_stm = terminal_stm
            node.children.append(child)
            continue
        child.handle = evaluator.extend(node.handle, node.board, move)
        node.children.append(child)
        heapq.heappush(arm.frontier, (-child.path_log_prior, next(counter), child))


def select_value_search_halving(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    config: HalvingConfig,
) -> tuple[int, list[dict[str, Any]]]:
    """Pick a root move by sequential halving over value-backed subtrees.

    Precondition: legal_moves is non-empty (the caller projects legal moves
    and raises before dispatch when none map to the vocab).
    """
    root_color = board.turn
    order = _prior_order(legal_log_priors)
    picks = list(order[: min(config.top_m, len(order))])
    seen = set(picks)
    for idx, move in enumerate(legal_moves):
        if idx not in seen and _is_forcing(board, move):
            picks.append(idx)
            seen.add(idx)

    counter = itertools.count()
    arms: list[_Arm] = []
    for idx in picks:
        move = legal_moves[idx]
        board1 = _search_copy(board)
        board1.push(move)
        terminal_root = terminal_value_for_color(board1, color=root_color)
        if terminal_root is not None and terminal_root >= 1.0:
            # Immediate win (checkmate delivered): no other move can score higher.
            return idx, [
                {
                    "move_uci": move.uci(),
                    "policy_log_prob": float(legal_log_priors[idx]),
                    "evals_spent": 0,
                    "max_depth": 0,
                    "backed_value": 1.0,
                    "search_score": 1.0,
                    "eliminated_round": None,
                }
            ]
        arm = _Arm(
            local_idx=idx,
            move=move,
            root_log_prior=float(legal_log_priors[idx]),
            root_node=None,
            terminal_value_root=terminal_root,
        )
        if terminal_root is None:
            node = _TreeNode(
                board=board1,
                handle=evaluator.extend(root_handle, board, move),
                depth=0,
                path_log_prior=float(legal_log_priors[idx]),
            )
            arm.root_node = node
            heapq.heappush(arm.frontier, (-node.path_log_prior, next(counter), node))
        arms.append(arm)

    rounds = config.rounds if config.rounds > 0 else _auto_rounds(len(arms))
    spent = 0
    survivors = list(arms)
    for round_idx in range(rounds):
        active = [arm for arm in survivors if arm.frontier]
        if not active or spent >= config.budget:
            break
        per_arm = max(
            1, (config.budget - spent) // ((rounds - round_idx) * len(active))
        )
        remaining = {id(arm): per_arm for arm in active}
        # Waves: pop -> batched evaluate -> expand, until the round budget is
        # spent or frontiers empty. One batched evaluate per wave (per level).
        while spent < config.budget:
            wave: list[tuple[_Arm, _TreeNode]] = []
            for arm in active:
                take = min(
                    remaining[id(arm)],
                    len(arm.frontier),
                    config.budget - spent - len(wave),
                )
                for _ in range(max(0, take)):
                    _, _, node = heapq.heappop(arm.frontier)
                    wave.append((arm, node))
                    remaining[id(arm)] -= 1
            if not wave:
                break
            evals = evaluator.evaluate([(node.handle, node.board) for _, node in wave])
            spent += len(wave)
            for (arm, node), position_eval in zip(wave, evals):
                node.value_stm = float(position_eval.value_stm)
                arm.evals_spent += 1
                arm.max_depth_reached = max(arm.max_depth_reached, node.depth)
                _push_children(
                    arm, node, position_eval, evaluator, config, counter, root_color
                )
        for arm in survivors:
            _score_arm(arm, config.lam)
        if round_idx < rounds - 1 and len(survivors) > 1:
            survivors.sort(key=lambda arm: arm.score, reverse=True)
            keep = math.ceil(len(survivors) / 2)
            for arm in survivors[keep:]:
                arm.eliminated_round = round_idx
            survivors = survivors[:keep]

    for arm in arms:
        _score_arm(arm, config.lam)
        # Survivors are already scored; this pass only fills backed_value /
        # score on eliminated arms so their debug rows are informative.

    best = max(survivors, key=lambda arm: arm.score)
    if best.score == float("-inf"):
        # Budget starvation: fall back to the highest-prior candidate.
        best = arms[0]

    rows = [
        {
            "move_uci": arm.move.uci(),
            "policy_log_prob": arm.root_log_prior,
            "evals_spent": arm.evals_spent,
            "max_depth": arm.max_depth_reached,
            "backed_value": arm.backed_value,
            "search_score": None if arm.score == float("-inf") else arm.score,
            "eliminated_round": arm.eliminated_round,
        }
        for arm in arms
    ]
    return best.local_idx, rows
