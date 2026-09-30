"""Sequential Gumbel AlphaZero, with exact chess transitions and no torch.

Algorithm reference: google-deepmind/mctx at
88f92056a420c2673bed282f5a0c00211f126e78 (Apache-2.0), policies.py,
action_selection.py, qtransforms.py and seq_halving.py. The schedule below
is adapted from DeepMind's Apache-2.0 implementation, Copyright 2021
DeepMind Technologies Limited. No JAX/mctx runtime dependency.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable
import chess
import imba_chess_native as cc
from . import cozy_bridge
from .search import EvalRequest, PositionEval, _drive, _root_hash_seed

REFERENCE_REVISION = "88f92056a420c2673bed282f5a0c00211f126e78"


@dataclass(frozen=True)
class GumbelConfig:
    simulations: int = 128
    top_m: int = 16
    max_depth: int = 32
    maxvisit_init: float = 50.0
    value_scale: float = 0.1
    epsilon: float = 1e-08
    # Halving-style tactics, all off by default (then the search is unchanged).
    # root_forcing: every forcing root move (capture, check, promotion) joins the
    #   top_m candidates of the root sequential halving.
    # forcing_floor: at opponent-to-move interior nodes, unvisited forcing
    #   replies are visited first (highest prior first), so refutations are found.
    # minimax_weight: edge Q = (1 - w) * mean + w * negamax over the realized
    #   subtree, so one strong refutation counts fully instead of being averaged.
    # own_forcing_floor: the same one-visit floor for our own forcing moves at our
    #   interior nodes (the root already has root_forcing).
    # visit_cap: caps the visit term of the completed-Q scale, (maxvisit_init +
    #   min(max visits, cap)) * value_scale, so budgets above the cap search deeper
    #   without trusting Q more (0 = uncapped).
    # own_width: at our own interior nodes, only the top-k moves by prior are
    #   selectable (0 = no cap); the root keeps its own candidate set.
    root_forcing: bool = False
    forcing_floor: bool = False
    minimax_weight: float = 0.0
    own_width: int = 0
    own_forcing_floor: bool = False
    visit_cap: int = 0

    def __post_init__(self):
        if min(self.simulations, self.top_m, self.max_depth) < 1:
            raise ValueError("simulations, top_m and max_depth must be positive")
        if any(type(x) is not bool for x in (self.root_forcing, self.forcing_floor, self.own_forcing_floor)):
            raise ValueError("root_forcing, forcing_floor and own_forcing_floor must be booleans")
        if type(self.visit_cap) is not int or self.visit_cap < 0:
            raise ValueError("visit_cap must be a nonnegative integer")
        if type(self.own_width) is not int or self.own_width < 0:
            raise ValueError("own_width must be a nonnegative integer")
        if not math.isfinite(self.minimax_weight) or not 0 <= self.minimax_weight <= 1:
            raise ValueError("minimax_weight must be in [0, 1]")
        if (
            not all(
                (
                    math.isfinite(x)
                    for x in (self.maxvisit_init, self.value_scale, self.epsilon)
                )
            )
            or self.epsilon <= 0
            or min(self.maxvisit_init, self.value_scale) < 0
        ):
            raise ValueError("invalid Q transform constants")


def considered_visits(candidates: int, simulations: int) -> tuple[int, ...]:
    if candidates <= 1:
        return tuple(range(simulations))
    rounds = math.ceil(math.log2(candidates))
    visits = [0] * candidates
    sequence = []
    count = candidates
    while len(sequence) < simulations:
        for _ in range(max(1, simulations // (rounds * count))):
            sequence.extend(visits[:count])
            for i in range(count):
                visits[i] += 1
        count = max(2, count // 2)
    return tuple(sequence[:simulations])


def softmax(logits):
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    total = math.fsum(weights)
    return [x / total for x in weights]


def completed_q(value, priors, visits, qvalues, config: GumbelConfig, prior_probs=None):
    """Python mirror of the native completed-Q transform (used for policy targets)."""
    probs = (
        [max(x, 1.1754943508222875e-38) for x in softmax(priors)]
        if prior_probs is None
        else prior_probs
    )
    visited_mass = math.fsum((p for p, n in zip(probs, visits) if n))
    weighted = (
        math.fsum(
            (p * q / visited_mass for p, q, n in zip(probs, qvalues, visits) if n)
        )
        if visited_mass
        else 0.0
    )
    count = sum(visits)
    mixed = (value + count * weighted) / (count + 1)
    values = [q if n else mixed for q, n in zip(qvalues, visits)]
    low, high = (min(values), max(values))
    top = max(visits)
    if config.visit_cap:
        top = min(top, config.visit_cap)
    scale = (config.maxvisit_init + top) * config.value_scale
    denominator = max(high - low, config.epsilon)
    return [scale * (q - low) / denominator for q in values]


@dataclass(frozen=True)
class GumbelResult:
    move_uci: str
    move_id: int
    legal_ids: list[int]
    policy: list[float]
    root_value: float
    root_wdl: tuple[float, float, float] | None
    visits: list[int]
    qvalues: list[float]
    simulations: int
    neural_evaluations: int
    terminal_hits: int
    depth_cutoffs: int
    maximum_depth: int
    root_log_priors: list[float] | None = None
    # Mean backed-up leaf WDL over the exact simulations, root-player POV.
    # Distinct from root_wdl (the raw network prediction). Measurement only:
    # neither this distribution nor the auxiliary head affects tree selection.
    search_wdl: tuple[float, float, float] | None = None


@dataclass
class _Node:
    board: Any
    history: list[int]
    handle: Any
    value: float | None = None
    evaluation: PositionEval | None = None
    visits: list[int] | None = None
    sums: list[float] | None = None
    means: list[float] | None = None
    prior_probs: list[float] = field(default_factory=list)
    children: dict[int, Any] = field(default_factory=dict)
    stats: Any = None

    def initialize(self, evaluation, config=None, role="root"):
        """role: "root", "opponent" (opponent to move) or "own" (root player, not root)."""
        n = len(evaluation.legal_ids)
        if not n or not all(
            (
                len(x) == n
                for x in (
                    evaluation.legal_moves,
                    evaluation.legal_ucis,
                    evaluation.legal_log_priors,
                )
            )
        ):
            raise ValueError("nonterminal evaluation has invalid legal projection")
        if (
            not math.isfinite(evaluation.value_stm)
            or abs(evaluation.value_stm) > 1.000001
            or (not all((math.isfinite(x) for x in evaluation.legal_log_priors)))
        ):
            raise ValueError("nonfinite or invalid network evaluation")
        self.evaluation = evaluation
        priors = list(evaluation.legal_log_priors)
        if evaluation.wdl is not None:
            wdl = evaluation.wdl
            if (len(wdl) != 3 or any(not math.isfinite(p) or p < 0 for p in wdl)
                    or abs(sum(wdl) - 1) > 1e-5
                    or abs(wdl[2] - wdl[0] - evaluation.value_stm) > 1e-5):
                raise ValueError("invalid or inconsistent evaluation WDL")
        self.value = evaluation.value_stm
        self.priors = priors
        self.prior_probs = [max(p, 1.1754943508222875e-38) for p in softmax(priors)]
        if len(evaluation.legal_forcing) != n:
            raise ValueError("nonterminal evaluation has invalid legal projection")
        if config is None or config.minimax_weight == 0:
            self.stats = cc.NodeStats(self.value, priors, self.prior_probs)
        else:
            self.stats = cc.NodeStats(self.value, priors, self.prior_probs, config.minimax_weight)
        if config is not None and (config.forcing_floor or config.own_forcing_floor):
            self.stats.set_forcing(list(evaluation.legal_forcing))
        if config is not None and config.own_width and role == "own" and n > config.own_width:
            keep = set(sorted(range(n), key=lambda i: (-priors[i], i))[: config.own_width])
            self.stats.set_candidates([i in keep for i in range(n)])

    def qs(self):
        return self.means


def gumbel_stepwise(
    *,
    board: chess.Board,
    extend: Callable,
    root_handle=None,
    config=GumbelConfig(),
    rng=None,
    noise=None,
    root_eval: PositionEval | None = None,
    root_wdl=None,
    should_stop: Callable[[], bool] = lambda: False,
):
    """Yield at most one new neural leaf per simulation; retain history per path.

    A supplied root_eval is counted as one neural evaluation too. Terminal
    roots are not playable and raise ValueError. Cancellation raises InterruptedError.
    """
    root = _Node(cozy_bridge.board_to_cozy(board), _root_hash_seed(board), root_handle)
    if (
        cozy_bridge.terminal_value_native(
            root.board, color_is_stm=True, hash_history=root.history
        )
        is not None
    ):
        raise ValueError("cannot search terminal root")
    if root_eval is None:
        (root_eval,) = yield EvalRequest([(root_handle, root.board)])
    root.initialize(root_eval, config)
    n = len(root_eval.legal_ids)
    if noise is None:
        rng = rng or random.Random()
        noise = [-math.log(-math.log(max(rng.random(), 1e-12))) for _ in range(n)]
    if len(noise) != n or not all((math.isfinite(x) for x in noise)):
        raise ValueError("noise must contain one finite value per legal action")
    root.stats.set_noise(noise)
    priors = root_eval.legal_log_priors
    max_prior = max(priors)
    evaluations, terminal_hits, cutoffs, deepest = (1, 0, 0, 0)
    wdl_sums = [0.0, 0.0, 0.0]
    wdl_available = True

    considered = min(config.top_m, n, config.simulations)
    if config.root_forcing:
        # Root halving over the usual top_m (by noise + prior) plus every forcing move.
        order = sorted(range(n), key=lambda i: (-(noise[i] + priors[i]), i))
        candidates = set(order[: min(config.top_m, n)])
        candidates.update(i for i, forcing in enumerate(root_eval.legal_forcing) if forcing)
        root.stats.set_candidates([i in candidates for i in range(n)])
        considered = min(len(candidates), config.simulations)

    def root_action(visit):
        return root.stats.root(
            visit, config.maxvisit_init, config.value_scale, config.epsilon, config.visit_cap
        )

    for visit in considered_visits(considered, config.simulations):
        if should_stop():
            raise InterruptedError("search cancelled")
        node, action, depth, path = (root, root_action(visit), 0, [])
        while True:
            path.append((node, action))
            depth += 1
            deepest = max(deepest, depth)
            if action not in node.children:
                ev = node.evaluation
                child_board, history, terminal = cc.push_and_classify(
                    node.board, ev.legal_moves[action], node.history, True
                )
                child = _Node(child_board, history, None, value=terminal)
                node.children[action] = child
                if terminal is None:
                    child.handle = extend(
                        node.handle, ev.legal_ucis[action], ev.legal_ids[action]
                    )
                    (evaluation,) = yield EvalRequest([(child.handle, child.board)])
                    child.initialize(evaluation, config, "opponent" if depth % 2 else "own")
                    evaluations += 1
                leaf = child
                if terminal is not None:
                    terminal_hits += 1
                elif depth == config.max_depth:
                    cutoffs += 1
                break
            node = node.children[action]
            if node.evaluation is None:
                terminal_hits += 1
                leaf = node
                break
            if depth == config.max_depth:
                cutoffs += 1
                leaf = node
                break
            floor = config.forcing_floor if depth % 2 == 1 else config.own_forcing_floor
            action = node.stats.interior(
                config.maxvisit_init, config.value_scale, config.epsilon, floor, config.visit_cap
            )
            continue
        cc.gumbel_backup([(n.stats, a) for n, a in path], leaf.value)
        if leaf.evaluation is None:
            leaf_wdl = (float(leaf.value == -1), float(leaf.value == 0), float(leaf.value == 1))
        else:
            leaf_wdl = leaf.evaluation.wdl
        if leaf_wdl is None:
            wdl_available = False
        else:
            # Every chess ply swaps players; draws retain their perspective.
            root_pov = leaf_wdl[::-1] if depth % 2 else leaf_wdl
            for i, probability in enumerate(root_pov):
                wdl_sums[i] += probability
    root.visits, root.sums, root.means = root.stats.snapshot()
    action = root_action(max(root.visits))
    q = completed_q(
        root.value, priors, root.visits, root.qs(), config, root.prior_probs
    )
    return GumbelResult(
        root_eval.legal_ucis[action],
        root_eval.legal_ids[action],
        list(root_eval.legal_ids),
        softmax([p + v for p, v in zip(priors, q)]),
        root.value,
        root_wdl,
        root.visits,
        root.qs(),
        config.simulations,
        evaluations,
        terminal_hits,
        cutoffs,
        deepest,
        list(priors),
        tuple(p / config.simulations for p in wdl_sums) if wdl_available else None,
    )


def select_gumbel(*, evaluator, **kwargs):
    return _drive(gumbel_stepwise(extend=evaluator.extend, **kwargs), evaluator)
