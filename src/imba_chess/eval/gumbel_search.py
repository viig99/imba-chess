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
    epsilon: float = 1e-8

    def __post_init__(self):
        if min(self.simulations, self.top_m, self.max_depth) < 1:
            raise ValueError("simulations, top_m and max_depth must be positive")
        if (
            not all(
                math.isfinite(x)
                for x in (self.maxvisit_init, self.value_scale, self.epsilon)
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
    # Clamp to float32 tiny just as the reference does, including underflowed priors.
    probs = (
        [max(x, 1.1754943508222875e-38) for x in softmax(priors)]
        if prior_probs is None
        else prior_probs
    )
    visited_mass = math.fsum(p for p, n in zip(probs, visits) if n)
    weighted = (
        math.fsum(p * q / visited_mass for p, q, n in zip(probs, qvalues, visits) if n)
        if visited_mass
        else 0.0
    )
    count = sum(visits)
    mixed = (value + count * weighted) / (count + 1)
    values = [q if n else mixed for q, n in zip(qvalues, visits)]
    low, high = min(values), max(values)
    scale = (config.maxvisit_init + max(visits)) * config.value_scale
    denominator = max(high - low, config.epsilon)
    return [scale * (q - low) / denominator for q in values]


def interior_action(value, priors, visits, qvalues, config, prior_probs=None):
    transformed = completed_q(value, priors, visits, qvalues, config, prior_probs)
    policy = softmax([p + q for p, q in zip(priors, transformed)])
    denominator = 1 + sum(visits)
    return max(range(len(priors)), key=lambda i: policy[i] - visits[i] / denominator)


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


@dataclass
class _Node:
    board: Any
    history: list[int]
    handle: Any
    value: float | None = None
    evaluation: PositionEval | None = None
    visits: list[int] = field(default_factory=list)
    sums: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    prior_probs: list[float] = field(default_factory=list)
    children: dict[int, Any] = field(default_factory=dict)

    def initialize(self, evaluation):
        n = len(evaluation.legal_ids)
        if not n or not all(
            len(x) == n
            for x in (
                evaluation.legal_moves,
                evaluation.legal_ucis,
                evaluation.legal_log_priors,
            )
        ):
            raise ValueError("nonterminal evaluation has invalid legal projection")
        if (
            not math.isfinite(evaluation.value_stm)
            or abs(evaluation.value_stm) > 1.000001
            or not all(math.isfinite(x) for x in evaluation.legal_log_priors)
        ):
            raise ValueError("nonfinite or invalid network evaluation")
        self.evaluation = evaluation
        self.value = evaluation.value_stm
        self.visits = [0] * n
        self.sums = [0.0] * n
        self.means = [0.0] * n
        self.prior_probs = [
            max(p, 1.1754943508222875e-38) for p in softmax(evaluation.legal_log_priors)
        ]

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
    root.initialize(root_eval)
    n = len(root.visits)
    if noise is None:
        rng = rng or random.Random()
        noise = [-math.log(-math.log(max(rng.random(), 1e-12))) for _ in range(n)]
    if len(noise) != n or not all(math.isfinite(x) for x in noise):
        raise ValueError("noise must contain one finite value per legal action")
    priors = root_eval.legal_log_priors
    max_prior = max(priors)
    evaluations, terminal_hits, cutoffs, deepest = 1, 0, 0, 0

    def root_action(visit):
        q = completed_q(
            root.value, priors, root.visits, root.qs(), config, root.prior_probs
        )
        return max(
            (i for i in range(n) if root.visits[i] == visit),
            key=lambda i: max(-1e9, noise[i] + priors[i] - max_prior + q[i]),
        )

    for visit in considered_visits(
        min(config.top_m, n, config.simulations), config.simulations
    ):
        if should_stop():
            raise InterruptedError("search cancelled")
        node, action, depth, path = root, root_action(visit), 0, []
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
                    child.initialize(evaluation)
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
            action = interior_action(
                node.value,
                node.evaluation.legal_log_priors,
                node.visits,
                node.qs(),
                config,
                node.prior_probs,
            )
        value = leaf.value
        for parent, edge in reversed(path):
            value = -value
            parent.visits[edge] += 1
            parent.sums[edge] += value
            parent.means[edge] = parent.sums[edge] / parent.visits[edge]
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
    )


def select_gumbel(*, evaluator, **kwargs):
    return _drive(gumbel_stepwise(extend=evaluator.extend, **kwargs), evaluator)
