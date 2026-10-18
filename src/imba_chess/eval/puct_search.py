"""Batched PUCT family on the same chess transitions and leaf decoder as Gumbel.

Lc0-style means PUCT with a visit-dependent coefficient and reduced parent FPU;
this is not a port of the complete Lc0 engine. Variance scaling regularizes the
parent's empirical playout variance toward a fixed prior. PUCT-V implements
Weichart, arXiv:2512.21648 (Eq. 22 in v1, Eq. 17 in v3), with empirical
action-return variance; its log term uses ln(N + 1) rather than the paper's log N.
"""

from dataclasses import dataclass
import math
import random

import chess
import imba_chess_native as cc

from . import cozy_bridge
from .gumbel_search import GumbelResult, _Node, policy_wdl
from .search import EvalRequest, _drive, _root_hash_seed


PUCT_ALGORITHMS = ("puct", "puct_variance", "puct_v")


@dataclass(frozen=True)
class PuctConfig:
    algorithm: str = "puct"
    simulations: int = 256
    max_depth: int = 32
    cpuct: float = 1.75
    cpuct_base: float = 38739.0
    cpuct_factor: float = 3.89
    fpu_reduction: float = 0.33
    variance_prior: float = 0.25
    variance_prior_weight: float = 8.0
    # PUCT-V's c2 accounts for our reward range of 2 rather than 1.
    variance_c1: float = math.sqrt(2)
    variance_c2: float = 6.0
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    target_policy: str = "visits"
    rpo_regularization: float = 1.0
    # With root noise on (self-play), game plies below this play a visit-proportional
    # sample instead of the most-visited move: AlphaZero's temperature 1 for the opening.
    # Without it, games from the initial board collapse onto one repeated line.
    sample_plies: int = 0

    def __post_init__(self):
        if self.algorithm not in PUCT_ALGORITHMS:
            raise ValueError("unknown PUCT variant")
        if any(type(v) is not int or v < 1 for v in (self.simulations, self.max_depth)):
            raise ValueError("simulations and max_depth must be positive integers")
        if type(self.sample_plies) is not int:
            raise ValueError("sample_plies must be an integer")
        for name, value in vars(self).items():
            if name in ("algorithm", "target_policy"):
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid PUCT parameter {name}")
        if min(self.cpuct_base, self.variance_prior, self.variance_prior_weight, self.dirichlet_alpha) <= 0:
            raise ValueError("PUCT prior scales must be positive")
        if self.dirichlet_fraction > 1:
            raise ValueError("dirichlet_fraction must be in [0,1]")
        if self.target_policy not in ("visits", "rpo") or self.rpo_regularization <= 0:
            raise ValueError("invalid PUCT target policy or RPO regularization")


def rpo_policy(priors, qvalues, regularization):
    """Exact root reverse-KL RPO target, Grill et al. (ICML 2020), Eq. 7.

    Maximize q.y - lambda KL(prior || y), lambda as in their Eq. 4. Its solution is
    y[a] = lambda prior[a] / (alpha - q[a]); find alpha by normalization.
    This changes only the learning target, not tree traversal or the played move.
    """
    if (not priors or len(priors) != len(qvalues) or not math.isfinite(regularization)
            or regularization <= 0 or any(not math.isfinite(q) for q in qvalues)
            or any(not math.isfinite(p) or p <= 0 for p in priors)):
        raise ValueError("invalid RPO inputs")
    total = math.fsum(priors)
    priors = [p / total for p in priors]
    maximum = max(qvalues)
    if max(qvalues) == min(qvalues):
        return priors
    low, high = maximum, maximum + regularization
    for _ in range(64):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        mass = math.fsum(regularization * p / (middle - q) for p, q in zip(priors, qvalues))
        if mass > 1:
            low = middle
        else:
            high = middle
    weights = [regularization * p / max(high - q, 1e-15) for p, q in zip(priors, qvalues)]
    total = math.fsum(weights)
    return [w / total for w in weights]


def puct_stepwise(*, board, extend, root_handle=None, config=PuctConfig(), rng=None,
                  noise=None, root_eval=None, root_wdl=None,
                  should_stop=lambda: False, on_played=None):
    root = _Node(cozy_bridge.board_to_cozy(board), _root_hash_seed(board), root_handle)
    if cozy_bridge.terminal_value_native(root.board, color_is_stm=True, hash_history=root.history) is not None:
        raise ValueError("cannot search terminal root")
    if root_eval is None:
        (root_eval,) = yield EvalRequest([(root_handle, root.board)])
    mode = PUCT_ALGORITHMS.index(config.algorithm)

    def initialize(node, evaluation, selection_probs=None):
        node.initialize(evaluation)
        node.stats.configure_puct(mode, config.cpuct, config.cpuct_base, config.cpuct_factor,
                                  config.fpu_reduction, config.variance_c1, config.variance_c2,
                                  config.variance_prior, config.variance_prior_weight, selection_probs)

    initialize(root, root_eval)
    n = len(root_eval.legal_ids)
    if noise is None:
        rng = rng or random.Random()
        eta = [rng.gammavariate(config.dirichlet_alpha, 1.0) for _ in range(n)]
        total = math.fsum(eta)
        eta = [v / total for v in eta]
    elif noise == 0.0 or (len(noise) == n and not any(noise)):
        eta = None
    else:
        raise ValueError("PUCT noise must be omitted or explicitly zero")
    if eta is not None and config.dirichlet_fraction:
        fraction = config.dirichlet_fraction
        mixed = [(1 - fraction) * p + fraction * e for p, e in zip(root.stats.probs, eta)]
        root.stats.configure_puct(mode, config.cpuct, config.cpuct_base, config.cpuct_factor,
                                  config.fpu_reduction, config.variance_c1, config.variance_c2,
                                  config.variance_prior, config.variance_prior_weight, mixed)
    evaluations, terminal_hits, cutoffs, deepest = 1, 0, 0, 0
    wdl_sums, wdl_available = [0.0, 0.0, 0.0], True
    edge_wdl_sums = [[0.0, 0.0, 0.0] for _ in range(n)]
    for _ in range(config.simulations):
        if should_stop():
            raise InterruptedError("search cancelled")
        node, depth, path = root, 0, []
        while True:
            action = node.stats.select_puct()
            path.append((node, action))
            depth += 1
            deepest = max(deepest, depth)
            if action not in node.children:
                ev = node.evaluation
                child_board, history, terminal = cc.push_and_classify(node.board, ev.move_at(action), node.history, True)
                child = _Node(child_board, history, None, value=terminal)
                node.children[action] = child
                if terminal is None:
                    child.handle = extend(node.handle, ev.uci_at(action), ev.id_at(action))
                    (evaluation,) = yield EvalRequest([(child.handle, child.board)])
                    initialize(child, evaluation)
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
        cc.gumbel_backup([(parent.stats, a) for parent, a in path], leaf.value)
        leaf_wdl = ((float(leaf.value == -1), float(leaf.value == 0), float(leaf.value == 1))
                    if leaf.evaluation is None else leaf.evaluation.wdl)
        if leaf_wdl is None:
            wdl_available = False
        else:
            root_pov = leaf_wdl[::-1] if depth % 2 else leaf_wdl
            edge = edge_wdl_sums[path[0][1]]
            for i, probability in enumerate(root_pov):
                wdl_sums[i] += probability
                edge[i] += probability
    visits, _, qvalues = root.stats.snapshot()
    # Visits define both the played move and the untempered training policy. Root
    # exploration affects visits, as in AlphaZero; with noise off the move is deterministic.
    if noise is None and board.ply() < config.sample_plies:
        action = rng.choices(range(n), weights=visits)[0]
    else:
        action = max(range(n), key=lambda i: (visits[i], qvalues[i], root_eval.legal_log_priors[i], -i))
    if on_played is not None:
        on_played(root.children.get(action))
    policy = [v / config.simulations for v in visits]
    if config.target_policy == "rpo":
        probs = root.stats.probs
        mass = math.fsum(p for p, v in zip(probs, visits) if v)
        weighted_q = math.fsum(p * q / mass for p, q, v in zip(probs, qvalues, visits) if v)
        mixed = (root.value + config.simulations * weighted_q) / (config.simulations + 1)
        completed = [q if v else mixed for q, v in zip(qvalues, visits)]
        c = config.cpuct + config.cpuct_factor * math.log((config.simulations + config.cpuct_base + 1) / config.cpuct_base)
        regularization = config.rpo_regularization * max(c, 1e-6) * math.sqrt(config.simulations) / (config.simulations + n)
        policy = rpo_policy(probs, completed, regularization)
    root_wdl = root_wdl if root_wdl is not None else root_eval.wdl
    # A visit policy's WDL is search_wdl; an RPO target's differs.
    improved_wdl = (policy_wdl(policy, visits, edge_wdl_sums, root.stats.probs, root_wdl)
                    if wdl_available else None)
    return GumbelResult(root_eval.legal_ucis[action], root_eval.legal_ids[action],
                        list(root_eval.legal_ids), policy,
                        root.value, root_wdl,
                        visits, qvalues, config.simulations, evaluations, terminal_hits, cutoffs,
                        deepest, list(root_eval.legal_log_priors),
                        tuple(p / config.simulations for p in wdl_sums) if wdl_available else None,
                        improved_wdl)


def select_puct(*, evaluator, **kwargs):
    return _drive(puct_stepwise(extend=evaluator.extend, **kwargs), evaluator)
