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
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, NamedTuple, Optional, Protocol

import chess
import imba_chess_native as cc

from imba_chess.eval import cozy_bridge


class PositionEval(NamedTuple):
    """One evaluated position: value head + legal moves under the vocab.

    `legal_moves` are cozy-chess `cc.Move` objects (Stage 3: `evaluate()`
    receives cozy boards and projects legal moves via cozy movegen);
    `legal_ucis` is index-aligned with `legal_moves`, computed once during
    projection (via `cozy_move_to_uci`, castling-aware) so search/rows never
    re-derive UCI strings from a move object.

    `legal_forcing` is index-aligned too: True where the move is a promotion,
    capture, or check. It comes back from the same native projection call that
    produced the moves, because the search's refutation floor needs exactly
    that predicate over exactly that (board, moves) pair -- re-deriving it
    afterwards cost ~19.8us and two FFI crossings per move on every
    opponent-to-move node.

    `legal_ids` is the vocabulary id per move, also index-aligned. The
    projector computes it to gather logits and used to drop it, so `extend()`
    re-derived the same integer from the UCI string -- a hash and a dict lookup
    per created child, 270k times per 4-game run.

    Deliberately no defaults: an omitted `legal_forcing` would read as "no move
    is forcing", silently emptying the refutation floor and changing what the
    search explores. Better a TypeError at construction.
    """

    value_stm: float
    legal_moves: list["cc.Move"]
    legal_ucis: list[str]
    legal_log_priors: list[float]
    legal_forcing: list[bool]
    legal_ids: list[int]
    # Optional for scalar-only evaluators; never infer draw mass from a scalar.
    wdl: tuple[float, float, float] | None = None  # loss, draw, win; side to move


class PositionEvaluator(Protocol):
    """`handle` is opaque (a search-node handle); `evaluate` batch entries
    are `(handle, cozy_board)` pairs (cozy-chess Board -- the search tree
    below the root is cozy-only, Stage 3 Task 5: no python-chess board is
    built or carried per tree node). `extend` only needs the played move's
    UCI (vocab encoding); it does not need a board."""

    def extend(
        self, handle: Any, move_uci: str, move_vocab_id: int | None = None
    ) -> Any: ...

    def evaluate(self, batch: list[tuple[Any, "cc.Board"]]) -> list[PositionEval]: ...


class EvalRequest(NamedTuple):
    """A batch of (handle, cozy_board) pairs a stepwise generator wants scored.

    Yielded by the `*_stepwise` generators in place of a synchronous
    `evaluator.evaluate(batch)` call; the driver sends back the matching
    `list[PositionEval]` via `gen.send(...)`.
    """

    batch: list[tuple[Any, "cc.Board"]]


def _drive(
    gen: Generator[EvalRequest, list[PositionEval], Any], evaluator: PositionEvaluator
) -> Any:
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
    budget: int = 2048
    top_m: int = 16
    rounds: int = 0  # 0 = auto ceil(log2(num_arms))
    refutation_top_r: int = 4
    expand_top: int = 3
    max_depth: int = 4
    lam: float = 0.05


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


def _forcing_index_set_root(
    legal_moves: list[chess.Move],
    cozy_board: "cc.Board",
    *,
    board: chess.Board,
) -> set[int]:
    """Indices of forcing moves (promotion/capture/check) at the root.

    `legal_moves` is python-chess Move objects; `board` -- the real root
    python-chess board -- is required for the capture test. Check-detection
    lazily translates each move via py_move_to_cozy only when the
    promotion/capture fast checks don't already resolve it as forcing.
    """
    forcing: set[int] = set()
    for idx, move in enumerate(legal_moves):
        is_capture = board.is_capture(move)
        if move.promotion is not None or is_capture:
            forcing.add(idx)
        elif cozy_bridge.gives_check(
            cozy_board, cozy_bridge.py_move_to_cozy(board, move)
        ):
            forcing.add(idx)
    return forcing


def _prior_order(legal_log_priors: list[float]) -> list[int]:
    return sorted(
        range(len(legal_log_priors)),
        key=legal_log_priors.__getitem__,
        reverse=True,
    )


def _search_copy(board: chess.Board) -> chess.Board:
    # Bounded copy: only enough move-stack history for draw-claim detection
    # (bounded by halfmove_clock) is needed; copying a late-game full stack
    # is ~150x slower for no benefit. Used only by _root_hash_seed (the
    # cozy-only tree below the root carries no python-chess board at all).
    return board.copy(stack=board.halfmove_clock)


def _root_hash_seed(board: chess.Board) -> list[int]:
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
    for move in moves[
        :-1
    ]:  # skip the last move: it reaches the CURRENT position, excluded
        cozy = copy.copy(cozy)
        cozy.play(cozy_bridge.py_move_to_cozy(twin, move))
        twin.push(move)
        history.append(cozy_bridge.repetition_hash(cozy))
    return history


@dataclass
class _TreeNode:
    cozy_board: "cc.Board"
    hash_history: list[
        int
    ]  # repetition_hash history per the push_and_classify contract
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
        forcing = {idx for idx, flag in enumerate(position_eval.legal_forcing) if flag}
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
        # color IS the child's own side to move by construction, so
        # color_is_stm is trivially True (terminal_value_stm is side-to-move
        # POV, per _TreeNode's docstring).
        child_cozy, child_history, terminal_stm = cc.push_and_classify(
            node.cozy_board, move_cozy, node.hash_history, True
        )
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
        if terminal_stm is not None:
            child.terminal_value_stm = terminal_stm
            node.children.append(child)
            continue
        child.handle = extend(node.handle, move_uci, position_eval.legal_ids[idx])
        node.children.append(child)
        heapq.heappush(arm.frontier, (-child.path_log_prior, next(counter), child))


def merge_search_stats(target: dict[str, int], fragment: dict[str, int]) -> None:
    """Pool counts over arms/games; maximum depths are maxima, not sums."""
    for key, value in fragment.items():
        if key.startswith("max_"):
            target[key] = max(target.get(key, 0), value)
        else:
            target[key] = target.get(key, 0) + value


def summarize_search_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for row in rows:
        merge_search_stats(stats, row.get("search_stats", {}))
    return stats


def _arm_search_stats(arm: _Arm, config: HalvingConfig) -> dict[str, int]:
    """Measure the final tree once, including eliminated arms."""
    stats = dict.fromkeys(("evals_spent", "horizon_evals", "max_depth"), 0)
    stack = [arm.root_node] if arm.root_node is not None else []
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if node.value_stm is not None:
            stats["evals_spent"] += 1
            stats["horizon_evals"] += int(node.depth == config.max_depth)
            stats["max_depth"] = max(stats["max_depth"], node.depth)
    return stats


def _halving_stepwise(
    *,
    extend: Callable[[Any, str], Any],
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    config: HalvingConfig,
) -> Generator[EvalRequest, list[PositionEval], tuple[int, list[dict[str, Any]]]]:
    """Stepwise generator core of select_value_search_halving; see its docstring.

    Precondition: legal_moves is non-empty (the caller projects legal moves
    and raises before dispatch when none map to the vocab).
    """
    root_color = board.turn
    root_cozy = cozy_bridge.board_to_cozy(board)
    root_hash_seed = _root_hash_seed(board)
    order = _prior_order(legal_log_priors)
    picks = list(order[: min(config.top_m, len(order))])
    seen = set(picks)
    forcing = _forcing_index_set_root(legal_moves, root_cozy, board=board)
    for idx in range(len(legal_moves)):
        if idx not in seen and idx in forcing:
            picks.append(idx)
            seen.add(idx)

    counter = itertools.count()
    arms: list[_Arm] = []
    for idx in picks:
        move = legal_moves[idx]
        cozy_move = cozy_bridge.py_move_to_cozy(board, move)
        # One ply past the root: side to move at cozy1 is the opponent, so
        # root-POV color is never the side to move here (color_is_stm=False).
        cozy1, hash_history1, terminal_root = cc.push_and_classify(
            root_cozy, cozy_move, root_hash_seed, False
        )
        if terminal_root is not None and terminal_root >= 1.0:
            # Immediate win (checkmate delivered): no other move can score higher.
            terminal_arm = _Arm(
                idx, move, float(legal_log_priors[idx]), None, terminal_root
            )
            return idx, [
                {
                    "move_uci": move.uci(),
                    "policy_log_prob": float(legal_log_priors[idx]),
                    "evals_spent": 0,
                    "max_depth": 0,
                    "backed_value": 1.0,
                    "search_score": 1.0,
                    "eliminated_round": None,
                    "search_stats": _arm_search_stats(terminal_arm, config),
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
            evals = yield EvalRequest(
                batch=[(node.handle, node.cozy_board) for _, node in wave]
            )
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
        best = max(arms, key=lambda arm: arm.root_log_prior)

    rows = [
        {
            "move_uci": arm.move.uci(),
            "policy_log_prob": arm.root_log_prior,
            "evals_spent": arm.evals_spent,
            "max_depth": arm.max_depth_reached,
            "backed_value": arm.backed_value,
            "search_score": None if arm.score == float("-inf") else arm.score,
            "eliminated_round": arm.eliminated_round,
            "search_stats": _arm_search_stats(arm, config),
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
        ),
        evaluator,
    )
