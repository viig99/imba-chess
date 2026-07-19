"""Move-selection strategies for eval play, decoupled from the model.

Strategies consume a PositionEvaluator: `handle` is opaque (the eval script
uses a parent-linked cache node; tests use whatever they need), `extend`
derives the handle for the position after a move, and `evaluate` batch-scores
positions, returning the value-head scalar (side-to-move POV) plus the legal
moves that map to the move vocab and their log-softmax policy priors.

This module must stay torch-free so strategy unit tests need no model.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, NamedTuple, Optional, Protocol

import chess
import cozy_chess as cc

from imba_chess.eval import cozy_bridge


class PositionEval(NamedTuple):
    """One evaluated position: value head + legal moves under the vocab.

    `legal_moves` are cozy-chess `cc.Move` objects (Stage 3: `evaluate()`
    receives cozy boards and projects legal moves via cozy movegen);
    `legal_ucis` is index-aligned with `legal_moves`, computed once during
    projection (via `cozy_move_to_uci`, castling-aware) so search/rows never
    re-derive UCI strings from a move object.
    """

    value_stm: float
    legal_moves: list["cc.Move"]
    legal_ucis: list[str]
    legal_log_priors: list[float]


class PositionEvaluator(Protocol):
    """`handle` is opaque (a search-node handle); `evaluate` batch entries
    are `(handle, cozy_board)` pairs (cozy-chess Board -- the search tree
    below the root is cozy-only, Stage 3 Task 5: no python-chess board is
    built or carried per tree node). `extend` only needs the played move's
    UCI (vocab encoding); it does not need a board."""

    def extend(self, handle: Any, move_uci: str) -> Any: ...

    def evaluate(
        self, batch: list[tuple[Any, "cc.Board"]]
    ) -> list[PositionEval]: ...


class EvalRequest(NamedTuple):
    """A batch of (handle, cozy_board) pairs a stepwise generator wants scored.

    Yielded by the `*_stepwise` generators in place of a synchronous
    `evaluator.evaluate(batch)` call; the driver sends back the matching
    `list[PositionEval]` via `gen.send(...)`.
    """

    batch: list[tuple[Any, "cc.Board"]]


def _drive(gen: Generator[EvalRequest, list[PositionEval], Any], evaluator: PositionEvaluator) -> Any:
    """Run a stepwise search generator to completion synchronously.

    This is the sync API's entire implementation: pump the generator,
    answering each EvalRequest with evaluator.evaluate(batch), until it
    returns. A future G-game scheduler drives the same generators by
    interleaving evaluate() calls across games instead.
    """
    try:
        request = next(gen)
        while True:
            request = gen.send(evaluator.evaluate(request.batch))
    except StopIteration as stop:
        return stop.value


@dataclass(frozen=True)
class HalvingConfig:
    budget: int = 256
    top_m: int = 16
    rounds: int = 0  # 0 = auto ceil(log2(num_arms))
    refutation_top_r: int = 2
    expand_top: int = 3
    max_depth: int = 4
    lam: float = 0.05
    gumbel_root_sampling: bool = False


def _auto_rounds(num_arms: int) -> int:
    return max(1, math.ceil(math.log2(max(2, num_arms))))


def terminal_value_for_color(
    board: chess.Board, *, color: chess.Color, cozy_board: "cc.Board | None" = None
) -> Optional[float]:
    """Public shim for external callers (tests/harness). The search tree
    itself (below this module's public select_* entry points) is cozy-only:
    it calls cozy_bridge.terminal_value_native directly with per-node
    hash_history, never through this function. This shim reconstructs the
    hash_history a fresh call needs from `board`'s own move stack via
    _root_hash_seed -- see that function's docstring for the contract.
    """
    if cozy_board is None:
        cozy_board = cozy_bridge.board_to_cozy(board)
    return cozy_bridge.terminal_value_native(
        cozy_board,
        color_is_stm=(color == board.turn),
        hash_history=_root_hash_seed(board),
    )


def select_greedy(legal_log_priors: list[float]) -> int:
    return max(range(len(legal_log_priors)), key=legal_log_priors.__getitem__)


def _forcing_index_set(
    legal_moves: list,
    cozy_board: "cc.Board",
    *,
    board: Optional[chess.Board] = None,
    legal_ucis: Optional[list[str]] = None,
) -> set[int]:
    """Indices of forcing moves (promotion/capture/check), one cozy board per node.

    `legal_moves` is python-chess Move objects at the root (`board` -- the
    real root python-chess board -- is required there for the capture test)
    or cozy-chess Move objects at tree nodes (legal_ucis required there,
    aligned with legal_moves via PositionEval.legal_ucis; the tree carries
    no python-chess board at all as of Task 5, so capture detection there is
    cozy-native via cozy_bridge.is_capture_cozy). Check-detection uses
    whichever cozy move representation is already at hand -- the root path
    lazily translates via py_move_to_cozy only when the promotion/capture
    fast checks don't already resolve the move as forcing.
    """
    forcing: set[int] = set()
    for idx, move in enumerate(legal_moves):
        if isinstance(move, chess.Move):
            assert board is not None
            is_capture, cozy_move = board.is_capture(move), None
        else:
            assert legal_ucis is not None
            is_capture, cozy_move = cozy_bridge.is_capture_cozy(cozy_board, move), move
        if move.promotion is not None or is_capture:
            forcing.add(idx)
        elif cozy_bridge.gives_check(
            cozy_board, cozy_move if cozy_move is not None else cozy_bridge.py_move_to_cozy(board, move)
        ):
            forcing.add(idx)
    return forcing


def _prior_order(legal_log_priors: list[float]) -> list[int]:
    return sorted(
        range(len(legal_log_priors)),
        key=legal_log_priors.__getitem__,
        reverse=True,
    )


def _gumbel_top_k_order(legal_log_priors: list[float], *, rng: random.Random) -> list[int]:
    """Sample move indices without replacement via the Gumbel-Top-k trick.

    Adds i.i.d. Gumbel(0) noise to each move's log-prior and orders by the
    perturbed score. This is an unbiased sample-without-replacement from the
    policy distribution (Danihelka et al., ICLR 2022) -- unlike a plain
    top-k-by-prior cut, which can permanently and systematically exclude a
    genuinely good but low-prior move from ever being searched (their
    Example 1 constructs exactly this failure: a deterministic top-2 cut
    that misses the only good action and scores worse than the raw prior).
    """
    def gumbel_noise() -> float:
        u = max(rng.random(), 1e-12)
        return -math.log(-math.log(u))

    scored = [(log_prior + gumbel_noise(), idx) for idx, log_prior in enumerate(legal_log_priors)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [idx for _, idx in scored]


def _search_copy(board: chess.Board) -> chess.Board:
    # Bounded copy: only enough move-stack history for draw-claim detection
    # (bounded by halfmove_clock) is needed; copying a late-game full stack
    # is ~150x slower for no benefit. Used only by _root_hash_seed (the
    # cozy-only tree below the root carries no python-chess board at all).
    return board.copy(stack=board.halfmove_clock)


def _root_hash_seed(board: chess.Board) -> tuple[int, ...]:
    """repetition_hash() of the (up to) `halfmove_clock` positions PRIOR to
    each of the last `n = min(board.halfmove_clock, len(board.move_stack))`
    played moves, oldest first, current position excluded -- the exact
    hash_history contract cozy_bridge.terminal_value_native expects (see its
    docstring), reconstructed from a bare python-chess board's move stack.
    This is what seeds a fresh tree walk (_cozy_push then folds in one more
    hash per non-zeroing ply as the tree descends). Empty stack -> empty
    tuple (matches pre-Task-5 stackless behavior: no history, no claim).

    Bounded-copy + pop/replay rather than re-walking board.move_stack in
    place, so the passed-in `board` is never mutated.
    """
    twin = _search_copy(board)
    n = len(twin.move_stack)
    if n == 0:
        return ()
    moves = [twin.pop() for _ in range(n)]
    moves.reverse()  # chronological order, oldest first
    cozy = cozy_bridge.board_to_cozy(twin)
    history = [cozy_bridge.repetition_hash(cozy)]
    for move in moves[:-1]:  # skip the last move: it reaches the CURRENT position, excluded
        cozy = copy.copy(cozy)
        cozy.play(cozy_bridge.py_move_to_cozy(twin, move))
        twin.push(move)
        history.append(cozy_bridge.repetition_hash(cozy))
    return tuple(history)


def _cozy_push(
    cozy_board: "cc.Board", cozy_move: "cc.Move", hash_history: tuple[int, ...]
) -> tuple["cc.Board", tuple[int, ...]]:
    """Copy-and-play one tree edge, threading the repetition hash_history.

    Every tree edge goes through here. child_history resets to empty on a
    zeroing move (capture/pawn move -- child.halfmove_clock == 0) and
    otherwise carries the parent's history forward with the parent's own
    repetition_hash appended -- see cozy_bridge.terminal_value_native's
    docstring for why reset-on-zeroing-only is a sufficient contract.
    """
    child = copy.copy(cozy_board)
    child.play(cozy_move)
    if child.halfmove_clock == 0:
        child_history: tuple[int, ...] = ()
    else:
        child_history = hash_history + (cozy_bridge.repetition_hash(cozy_board),)
    return child, child_history


@dataclass
class _RootCandidate:
    """One top-k root move expanded one ply deep.

    terminal_value is the exact game result (root POV) when the move ends the
    game, else None; board1_eval is the value/prior evaluation of the position
    after the move. worst_reply_value / reply_candidates are filled by
    value_search_d2 only. hash_history1 is cozy1's repetition hash_history
    (per the _cozy_push contract), threaded into any further push from cozy1.
    """

    local_idx: int
    move: chess.Move
    cozy1: "cc.Board"
    hash_history1: tuple[int, ...]
    log_prior: float
    terminal_value: Optional[float]
    handle1: Any = None
    board1_eval: Optional[PositionEval] = None
    worst_reply_value: Optional[float] = None
    reply_candidates: list[dict[str, Any]] = field(default_factory=list)


def _expand_root_candidates_stepwise(
    *,
    extend: Callable[[Any, str], Any],
    root_handle: Any,
    board: chess.Board,
    cozy_root: "cc.Board",
    root_hash_seed: tuple[int, ...],
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
) -> Generator[EvalRequest, list[PositionEval], tuple[list[_RootCandidate], Optional[int]]]:
    """Build the top-k prior root candidates and batch-evaluate their boards.

    Stepwise generator core shared by select_value_rerank and
    select_value_search_d2 (via `yield from`). Returns (candidates,
    mate_index) as its StopIteration.value. When a candidate move delivers
    checkmate no other move can score higher: mate_index is set, no eval
    request is made, and the partially built candidates list must be
    ignored. Terminal boards never appear as training tokens, so they carry
    the exact game result instead of going through the value head.
    """
    candidates: list[_RootCandidate] = []
    batch: list[tuple[Any, "cc.Board"]] = []
    batch_to_candidate: list[int] = []
    for local_idx in _prior_order(legal_log_priors)[: min(top_k, len(legal_moves))]:
        move = legal_moves[local_idx]
        cozy_move = cozy_bridge.py_move_to_cozy(board, move)
        cozy1, hash_history1 = _cozy_push(cozy_root, cozy_move, root_hash_seed)
        # One ply past the root: side to move at cozy1 is the opponent, so
        # root-POV color is never the side to move here (color_is_stm=False).
        terminal_value = cozy_bridge.terminal_value_native(
            cozy1, color_is_stm=False, hash_history=hash_history1
        )
        if terminal_value is not None and terminal_value >= 1.0:
            return candidates, local_idx
        candidate = _RootCandidate(
            local_idx=local_idx,
            move=move,
            cozy1=cozy1,
            hash_history1=hash_history1,
            log_prior=float(legal_log_priors[local_idx]),
            terminal_value=terminal_value,
        )
        candidates.append(candidate)
        if terminal_value is not None:
            continue
        candidate.handle1 = extend(root_handle, move.uci())
        batch.append((candidate.handle1, cozy1))
        batch_to_candidate.append(len(candidates) - 1)

    if batch:
        position_evals = yield EvalRequest(batch=batch)
        for cand_idx, position_eval in zip(batch_to_candidate, position_evals):
            candidates[cand_idx].board1_eval = position_eval
    return candidates, None


def _rerank_stepwise(
    *,
    extend: Callable[[Any, str], Any],
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
    lam: float,
) -> Generator[EvalRequest, list[PositionEval], tuple[int, list[dict[str, Any]]]]:
    root_cozy = cozy_bridge.board_to_cozy(board)
    candidates, mate_index = yield from _expand_root_candidates_stepwise(
        extend=extend,
        root_handle=root_handle,
        board=board,
        cozy_root=root_cozy,
        root_hash_seed=_root_hash_seed(board),
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        top_k=top_k,
    )
    if mate_index is not None:
        return mate_index, [
            {
                "move_uci": legal_moves[mate_index].uci(),
                "policy_logit": float(legal_log_priors[mate_index]),
                "policy_log_prob": float(legal_log_priors[mate_index]),
                "value_next": 1.0,
                "terminal": True,
                "rerank_score": 1.0,
            }
        ]

    chosen_index = candidates[0].local_idx
    best_score = float("-inf")
    rerank_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.terminal_value is not None:
            value_root = float(candidate.terminal_value)
        else:
            # Side-to-move at board1 is the opponent; negate to root POV.
            value_root = -float(candidate.board1_eval.value_stm)
        # Value-dominant score with a small log-prob policy prior as tiebreak.
        rerank_score = value_root + (lam * candidate.log_prior)
        rerank_rows.append(
            {
                "move_uci": candidate.move.uci(),
                "policy_logit": candidate.log_prior,
                "policy_log_prob": candidate.log_prior,
                "value_next": value_root,
                "terminal": candidate.terminal_value is not None,
                "rerank_score": rerank_score,
            }
        )
        if rerank_score > best_score:
            best_score = rerank_score
            chosen_index = candidate.local_idx

    return chosen_index, rerank_rows


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
    return _drive(
        _rerank_stepwise(
            extend=evaluator.extend,
            root_handle=root_handle,
            board=board,
            legal_moves=legal_moves,
            legal_log_priors=legal_log_priors,
            top_k=top_k,
            lam=lam,
        ),
        evaluator,
    )


def _d2_stepwise(
    *,
    extend: Callable[[Any, str], Any],
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
    lam: float,
) -> Generator[EvalRequest, list[PositionEval], tuple[int, list[dict[str, Any]]]]:
    root_cozy = cozy_bridge.board_to_cozy(board)
    candidates, mate_index = yield from _expand_root_candidates_stepwise(
        extend=extend,
        root_handle=root_handle,
        board=board,
        cozy_root=root_cozy,
        root_hash_seed=_root_hash_seed(board),
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        top_k=top_k,
    )
    if mate_index is not None:
        return mate_index, [
            {
                "move_uci": legal_moves[mate_index].uci(),
                "policy_logit": float(legal_log_priors[mate_index]),
                "policy_log_prob": float(legal_log_priors[mate_index]),
                "worst_reply_value": 1.0,
                "best_reply_uci": None,
                "search_score": 1.0,
            }
        ]

    board2_batch: list[tuple[Any, "cc.Board"]] = []
    board2_meta: list[tuple[_RootCandidate, int]] = []
    for candidate in candidates:
        if candidate.terminal_value is not None or candidate.board1_eval is None:
            continue
        board1_eval = candidate.board1_eval
        if not board1_eval.legal_moves:
            candidate.worst_reply_value = -float(board1_eval.value_stm)
            continue

        opp_indices = _prior_order(board1_eval.legal_log_priors)[
            : min(top_k, len(board1_eval.legal_moves))
        ]
        # Always consider forcing replies (captures/checks/promotions): the
        # tactical refutation is often a low-probability move under a
        # human-imitation policy, so policy top-k alone misses it.
        opp_seen = set(opp_indices)
        opp_forcing = _forcing_index_set(
            board1_eval.legal_moves, candidate.cozy1, legal_ucis=board1_eval.legal_ucis
        )
        for opp_idx in range(len(board1_eval.legal_moves)):
            if opp_idx in opp_seen:
                continue
            if opp_idx in opp_forcing:
                opp_indices.append(opp_idx)
                opp_seen.add(opp_idx)

        for opp_local_idx in opp_indices:
            opp_uci = board1_eval.legal_ucis[opp_local_idx]
            opp_move_cozy = board1_eval.legal_moves[opp_local_idx]
            cozy2, hash_history2 = _cozy_push(
                candidate.cozy1, opp_move_cozy, candidate.hash_history1
            )
            # Two plies past the root: side to move at cozy2 is root_color
            # again, so root-POV color IS the side to move (color_is_stm=True).
            terminal_value = cozy_bridge.terminal_value_native(
                cozy2, color_is_stm=True, hash_history=hash_history2
            )
            candidate.reply_candidates.append(
                {
                    "move_uci": opp_uci,
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
                (extend(candidate.handle1, opp_uci), cozy2)
            )
            board2_meta.append((candidate, len(candidate.reply_candidates) - 1))

    if board2_batch:
        board2_evals = yield EvalRequest(batch=board2_batch)
        for (candidate, reply_idx), position_eval in zip(
            board2_meta, board2_evals
        ):
            # Side-to-move at board2 is the root color again: POV matches root.
            candidate.reply_candidates[reply_idx]["value_after_reply"] = float(
                position_eval.value_stm
            )

    chosen_index = candidates[0].local_idx
    best_score = float("-inf")
    search_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.terminal_value is not None:
            worst_reply_value = float(candidate.terminal_value)
            best_reply_uci = None
        elif candidate.worst_reply_value is not None:
            worst_reply_value = float(candidate.worst_reply_value)
            best_reply_uci = None
        else:
            evaluated_replies = [
                row
                for row in candidate.reply_candidates
                if row.get("value_after_reply") is not None
            ]
            if not evaluated_replies:
                worst_reply_value = (
                    -float(candidate.board1_eval.value_stm)
                    if candidate.board1_eval is not None
                    else 0.0
                )
                best_reply_uci = None
            else:
                best_reply = min(
                    evaluated_replies, key=lambda row: float(row["value_after_reply"])
                )
                worst_reply_value = float(best_reply["value_after_reply"])
                best_reply_uci = str(best_reply["move_uci"])

        # Value-dominant score with a small log-prob policy prior as tiebreak.
        search_score = worst_reply_value + (lam * candidate.log_prior)
        search_rows.append(
            {
                "move_uci": candidate.move.uci(),
                "policy_logit": candidate.log_prior,
                "policy_log_prob": candidate.log_prior,
                "worst_reply_value": worst_reply_value,
                "best_reply_uci": best_reply_uci,
                "search_score": search_score,
            }
        )
        if search_score > best_score:
            best_score = search_score
            chosen_index = candidate.local_idx

    return chosen_index, search_rows


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
    return _drive(
        _d2_stepwise(
            extend=evaluator.extend,
            root_handle=root_handle,
            board=board,
            legal_moves=legal_moves,
            legal_log_priors=legal_log_priors,
            top_k=top_k,
            lam=lam,
        ),
        evaluator,
    )


@dataclass
class _TreeNode:
    cozy_board: "cc.Board"
    hash_history: tuple[int, ...]  # repetition_hash history per the _cozy_push contract
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
    extend: Callable[[Any, str], Any],
    config: HalvingConfig,
    counter: "itertools.count",
    root_color: chess.Color,
) -> None:
    if node.depth >= config.max_depth or not position_eval.legal_moves:
        return
    node_stm_is_white = node.cozy_board.side_to_move() == cc.Color.White
    opponent_to_move = node_stm_is_white != root_color
    order = _prior_order(position_eval.legal_log_priors)
    if opponent_to_move:
        forcing = _forcing_index_set(
            position_eval.legal_moves, node.cozy_board, legal_ucis=position_eval.legal_ucis
        )
        # Refutation floor: top-r replies by prior plus ALL forcing replies.
        picks = list(order[: config.refutation_top_r])
        seen = set(picks)
        for idx in range(len(position_eval.legal_moves)):
            if idx not in seen and idx in forcing:
                picks.append(idx)
                seen.add(idx)
    else:
        forcing = set()
        picks = list(order[: config.expand_top])

    for idx in picks:
        move_uci = position_eval.legal_ucis[idx]
        move_cozy = position_eval.legal_moves[idx]
        child_cozy, child_history = _cozy_push(node.cozy_board, move_cozy, node.hash_history)
        # Forcing replies inherit the parent's priority (no decay for their
        # own low prior): a refutation must compete at the plausibility of
        # the line it refutes, not of the reply itself.
        floor_pick = opponent_to_move and idx in forcing
        child_prior = node.path_log_prior + (
            0.0 if floor_pick else position_eval.legal_log_priors[idx]
        )
        child = _TreeNode(
            cozy_board=child_cozy,
            hash_history=child_history,
            handle=None,
            depth=node.depth + 1,
            path_log_prior=child_prior,
        )
        # color IS the child's own side to move by construction: color_is_stm
        # is trivially True (terminal_value_stm is side-to-move POV, per
        # _TreeNode's docstring).
        terminal_stm = cozy_bridge.terminal_value_native(
            child_cozy, color_is_stm=True, hash_history=child_history
        )
        if terminal_stm is not None:
            child.terminal_value_stm = terminal_stm
            node.children.append(child)
            continue
        child.handle = extend(node.handle, move_uci)
        node.children.append(child)
        heapq.heappush(arm.frontier, (-child.path_log_prior, next(counter), child))


def _halving_stepwise(
    *,
    extend: Callable[[Any, str], Any],
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    config: HalvingConfig,
    rng: Optional[random.Random] = None,
) -> Generator[EvalRequest, list[PositionEval], tuple[int, list[dict[str, Any]]]]:
    """Stepwise generator core of select_value_search_halving; see its docstring.

    Precondition: legal_moves is non-empty (the caller projects legal moves
    and raises before dispatch when none map to the vocab).

    rng is only consulted when config.gumbel_root_sampling is set; live
    Stockfish-eval play should leave it False (today's validated,
    deterministic top-m-by-prior behavior) and only rollout generation for
    future policy distillation should opt in -- see HalvingConfig.
    """
    root_color = board.turn
    root_cozy = cozy_bridge.board_to_cozy(board)
    root_hash_seed = _root_hash_seed(board)
    if config.gumbel_root_sampling:
        order = _gumbel_top_k_order(legal_log_priors, rng=rng if rng is not None else random.Random())
    else:
        order = _prior_order(legal_log_priors)
    picks = list(order[: min(config.top_m, len(order))])
    seen = set(picks)
    forcing = _forcing_index_set(legal_moves, root_cozy, board=board)
    for idx in range(len(legal_moves)):
        if idx not in seen and idx in forcing:
            picks.append(idx)
            seen.add(idx)

    counter = itertools.count()
    arms: list[_Arm] = []
    for idx in picks:
        move = legal_moves[idx]
        cozy_move = cozy_bridge.py_move_to_cozy(board, move)
        cozy1, hash_history1 = _cozy_push(root_cozy, cozy_move, root_hash_seed)
        # One ply past the root: side to move at cozy1 is the opponent, so
        # root-POV color is never the side to move here (color_is_stm=False).
        terminal_root = cozy_bridge.terminal_value_native(
            cozy1, color_is_stm=False, hash_history=hash_history1
        )
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
                cozy_board=cozy1,
                hash_history=hash_history1,
                handle=extend(root_handle, move.uci()),
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
            evals = yield EvalRequest(batch=[(node.handle, node.cozy_board) for _, node in wave])
            spent += len(wave)
            for (arm, node), position_eval in zip(wave, evals):
                node.value_stm = float(position_eval.value_stm)
                arm.evals_spent += 1
                arm.max_depth_reached = max(arm.max_depth_reached, node.depth)
                _push_children(
                    arm, node, position_eval, extend, config, counter, root_color
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


def select_value_search_halving(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    config: HalvingConfig,
    rng: Optional[random.Random] = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Pick a root move by sequential halving over value-backed subtrees.

    See _halving_stepwise for the algorithm; this is a thin synchronous
    driver around its generator core.
    """
    return _drive(
        _halving_stepwise(
            extend=evaluator.extend,
            root_handle=root_handle,
            board=board,
            legal_moves=legal_moves,
            legal_log_priors=legal_log_priors,
            config=config,
            rng=rng,
        ),
        evaluator,
    )
