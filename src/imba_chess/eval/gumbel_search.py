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
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple
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
    value_scale: float = 0.5
    epsilon: float = 1e-08
    # rescale_values (mctx semantics): True min-max normalizes completed Q per node
    #   to [0, 1] (epsilon floors the range); False uses raw Q in [-1, 1]. Min-max
    #   stretches near-ties into noise and squashes real gaps when one move blunders;
    #   raw Q at value_scale 0.5 keeps the mean target sharpness of min-max at 0.1.
    rescale_values: bool = False
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
        if any(type(x) is not bool for x in (self.root_forcing, self.forcing_floor, self.own_forcing_floor, self.rescale_values)):
            raise ValueError("root_forcing, forcing_floor, own_forcing_floor and rescale_values must be booleans")
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
    top = max(visits)
    if config.visit_cap:
        top = min(top, config.visit_cap)
    scale = (config.maxvisit_init + top) * config.value_scale
    if config.rescale_values:
        low, high = (min(values), max(values))
        denominator = max(high - low, config.epsilon)
        values = [(q - low) / denominator for q in values]
    return [scale * q for q in values]


def policy_wdl(policy, visits, edge_wdl_sums, prior_probs, root_wdl):
    """Root-player WDL of a root policy: sum_a policy[a] * WDL_a.

    WDL_a is the mean leaf WDL of the simulations that began with move a. An
    unvisited move takes the WDL form of completed Q's mixed value,
    (root_wdl + N * prior-weighted visited mean) / (N + 1), so W - L of the result
    is sum_a policy[a] * completed_q[a] (raw Q, no minimax blend). For a visit
    policy it equals the all-simulation average, search_wdl.
    """
    visited = [i for i, n in enumerate(visits) if n]
    if not visited or root_wdl is None:
        return None
    means = [[x / n for x in sums] if n else None for sums, n in zip(edge_wdl_sums, visits)]
    mass = math.fsum(prior_probs[i] for i in visited)
    count = sum(visits)
    mixed = [(root_wdl[k] + count * math.fsum(prior_probs[i] * means[i][k] for i in visited) / mass)
             / (count + 1) for k in range(3)]
    return tuple(math.fsum(p * (mean if mean is not None else mixed)[k] for p, mean in zip(policy, means))
                 for k in range(3))


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
    # Distinct from root_wdl (the raw network prediction). Neither WDL field
    # affects tree selection, but both can be training labels: [learning]
    # value_search_source picks one for value_search_mix and the auxiliary TD
    # targets. Under Gumbel this average includes the simulations halving spends
    # on moves it discards (~75% at 256 x 16), so as a label it is no better than
    # the raw network (golden-set MAE 0.166, bias -0.08 against Stockfish).
    search_wdl: tuple[float, float, float] | None = None
    # Root-player WDL of the returned policy (see policy_wdl): the value label of
    # the improved policy (golden-set MAE 0.138, bias -0.004); the recipe trains
    # on it. For a PUCT visit target it equals search_wdl.
    improved_wdl: tuple[float, float, float] | None = None


class PlayedChild(NamedTuple):
    """The played root child: its search handle and evaluation (both None
    for a terminal child)."""

    handle: Any
    evaluation: Any


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
    on_played: Callable | None = None,
    transition: Callable | None = None,
    observer=None,
):
    """Yield at most one new neural leaf per simulation; retain history per path.

    The tree and every simulation live in native code (cc.GumbelTree); this
    generator only scores the leaves it asks for. on_played, if given,
    receives the played action's PlayedChild (or None). transition replaces
    chess with a synthetic environment (see GumbelTree; tests only). observer,
    for measurement only, gets observer.root(root_eval, noise, priors,
    considered) once and observer.leaf(...) after every simulation (see
    GumbelTree.set_observer).

    A supplied root_eval is counted as one neural evaluation too. Terminal
    roots are not playable and raise ValueError. Cancellation raises InterruptedError.
    """
    root_board, history = cozy_bridge.board_to_cozy(board), _root_hash_seed(board)
    if transition is None and cc.gumbel_root_is_terminal(root_board, list(history)):
        raise ValueError("cannot search terminal root")
    if root_eval is None:
        (root_eval,) = yield EvalRequest([(root_handle, root_board)])
    tree = cc.GumbelTree(config, root_board, list(history), root_handle, root_eval, transition)
    n = len(tree)
    if noise is None:
        rng = rng or random.Random()
        noise = [-math.log(-math.log(max(rng.random(), 1e-12))) for _ in range(n)]
    tree.set_root_noise(list(noise))
    if observer is not None:
        observer.root(root_eval, list(noise), list(root_eval.legal_log_priors), tree.considered)
        tree.set_observer(observer.leaf)
    while (leaf := tree.next_leaf(should_stop)) is not None:
        parent, uci, move_id, child_board = leaf
        handle = extend(parent, uci, move_id)
        (evaluation,) = yield EvalRequest([(handle, child_board)])
        tree.attach(handle, evaluation)
    (
        visits, qs, probs, action, evaluations, terminal_hits, cutoffs, deepest,
        wdl_sums, edge_wdl_sums, wdl_available, played,
    ) = tree.finish()
    if on_played is not None:
        on_played(None if played is None else PlayedChild(*played))
    priors = root_eval.legal_log_priors
    root_value = root_eval.value_stm
    q = completed_q(root_value, priors, visits, qs, config, probs)
    policy = softmax([p + v for p, v in zip(priors, q)])
    improved_wdl = policy_wdl(
        policy, visits, edge_wdl_sums, probs,
        root_wdl if root_wdl is not None else root_eval.wdl,
    ) if wdl_available else None
    return GumbelResult(
        root_eval.legal_ucis[action],
        root_eval.legal_ids[action],
        list(root_eval.legal_ids),
        policy,
        root_value,
        root_wdl,
        visits,
        qs,
        config.simulations,
        evaluations,
        terminal_hits,
        cutoffs,
        deepest,
        list(priors),
        tuple(p / config.simulations for p in wdl_sums) if wdl_available else None,
        improved_wdl,
    )


def select_gumbel(*, evaluator, **kwargs):
    return _drive(gumbel_stepwise(extend=evaluator.extend, **kwargs), evaluator)
