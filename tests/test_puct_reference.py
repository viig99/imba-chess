"""End-to-end check of the native PUCT family against an independent reference.

The reference is written from the papers rather than from gumbel.rs: Lc0's
cpuct(N) and FPU reduction, parent-variance scaling, PUCT-V (Weichart,
arXiv:2512.21648 v1 Eq. 22 / v3 Eq. 17), and the RPO target of Grill et al.
(2020) Eqs. 4 and 7 with completed Q. lc0_parent_q / paper_log_n select the exact
Lc0 parent Q (own evaluation counted as a visit) and the paper's log N.
FakeEvaluator's constant value and uniform priors cannot tell formulas apart, so leaves get hashed values and peaked priors.
"""

from dataclasses import replace
import hashlib
import math
import random

import chess
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval.puct_search import PUCT_ALGORITHMS, PuctConfig, select_puct
from imba_chess.eval.search import PositionEval
from tests.test_gumbel_search import FakeEvaluator


FENS = (
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # Qxf7#
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",  # back-rank Ra8#
    "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP1QPPP/R3K2R w KQkq - 0 9",
)


def unit(key, salt):
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / 16**12


class HashEvaluator(FakeEvaluator):
    """Deterministic per-path value in (-0.9, 0.9), consistent WDL, peaked priors."""

    def evaluate(self, batch):
        result = []
        for handle, board in batch:
            self.calls.append(handle)
            ids, moves, ucis, forcing, _ = cozy_bridge.project_legal_moves(board, self.vocab)
            key = "/".join(handle or ())
            value = 1.8 * unit(key, "v") - 0.9
            logits = [4.0 * unit(f"{key}|{u}", "p") for u in ucis]
            top = max(logits)
            norm = top + math.log(sum(math.exp(x - top) for x in logits))
            draw = 0.3 * (1 - abs(value))
            wdl = ((1 - value - draw) / 2, draw, (1 + value - draw) / 2)  # loss, draw, win
            result.append(PositionEval(value, moves, ucis, [x - norm for x in logits],
                                       forcing, ids, wdl=wdl))
        return result


class Node:
    def __init__(self, board, evaluation=None, terminal=None):
        self.board, self.evaluation, self.terminal, self.children = board, evaluation, terminal, {}
        if evaluation is not None:
            logs = evaluation.legal_log_priors
            top = max(logs)
            weights = [math.exp(x - top) for x in logs]
            total = sum(weights)
            self.probs = [max(w / total, 1.1754943508222875e-38) for w in weights]
            n = len(logs)
            self.visits, self.sums, self.returns, self.node_returns = [0] * n, [0.0] * n, [[] for _ in range(n)], []


def terminal_value(board):
    """Side-to-move value; enough for the shallow trees of these positions."""
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    return None


def select(node, cfg):
    probs, visits, sums = node.probs, node.visits, node.sums
    total = sum(visits)
    if total == 0:
        return max(range(len(probs)), key=lambda i: (probs[i], -i))
    parent_q = (node.evaluation.value_stm + sum(sums)) / (total + 1) if cfg.lc0_parent_q else sum(sums) / total
    fpu = parent_q - cfg.fpu_reduction * math.sqrt(sum(p for p, n in zip(probs, visits) if n))
    # AlphaZero / Lc0: c(N) = c_init + c_factor log((N + base) / base), N = 1 + child visits.
    c = cfg.cpuct + cfg.cpuct_factor * math.log((total + 1 + cfg.cpuct_base) / cfg.cpuct_base)
    if cfg.algorithm == "puct_variance":
        x = node.node_returns
        mean = sum(x) / len(x)
        posterior = (sum((v - mean) ** 2 for v in x) + cfg.variance_prior_weight * cfg.variance_prior) / (
            len(x) + cfg.variance_prior_weight)
        c *= math.sqrt(posterior / cfg.variance_prior)
    scores = []
    for i, p in enumerate(probs):
        n = visits[i]
        q = sums[i] / n if n else fpu
        if cfg.algorithm == "puct_v":
            r = node.returns[i]
            sigma = math.sqrt(sum((v - q) ** 2 for v in r) / n) if n else 0.0
            u = (cfg.variance_c1 * p * sigma * math.sqrt(total)
                 + cfg.variance_c2 * p * math.log(total if cfg.paper_log_n else total + 1)) / (1 + n)
        else:
            u = c * p * math.sqrt(total) / (1 + n)
        scores.append(q + u)
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def rpo(priors, q, lam):
    """Grill et al. Prop. 4: bisection for alpha on [max(q + lam p), max(q) + lam]."""
    low, high = max(qi + lam * p for qi, p in zip(q, priors)), max(q) + lam
    for _ in range(200):
        middle = (low + high) / 2
        if sum(lam * p / (middle - qi) for p, qi in zip(priors, q)) > 1:
            low = middle
        else:
            high = middle
    y = [lam * p / (high - qi) for p, qi in zip(priors, q)]
    return [v / sum(y) for v in y]


def reference(board, cfg, eta):
    evaluate = HashEvaluator().evaluate
    (root_eval,) = evaluate([((), cozy_bridge.board_to_cozy(board))])
    root = Node(board.copy(), root_eval)
    clean = list(root.probs)
    if eta is not None:
        root.probs = [(1 - cfg.dirichlet_fraction) * p + cfg.dirichlet_fraction * e
                      for p, e in zip(root.probs, eta)]
    wdl_sums = [0.0, 0.0, 0.0]
    for _ in range(cfg.simulations):
        node, path, handle = root, [], ()
        while True:
            action = select(node, cfg)
            path.append((node, action))
            uci = node.evaluation.legal_ucis[action]
            if action not in node.children:
                child_board = node.board.copy()
                child_board.push_uci(uci)
                terminal = terminal_value(child_board)
                if terminal is None:
                    (evaluation,) = evaluate([(handle + (uci,), cozy_bridge.board_to_cozy(child_board))])
                    node.children[action] = Node(child_board, evaluation)
                else:
                    node.children[action] = Node(child_board, terminal=terminal)
                leaf = node.children[action]
                break
            node, handle = node.children[action], handle + (uci,)
            if node.evaluation is None or len(path) == cfg.max_depth:
                leaf = node
                break
        if leaf.evaluation is None:
            value = leaf.terminal
            wdl = (float(value == -1), float(value == 0), float(value == 1))
        else:
            value, wdl = leaf.evaluation.value_stm, leaf.evaluation.wdl
        for i, probability in enumerate(wdl[::-1] if len(path) % 2 else wdl):
            wdl_sums[i] += probability
        # Negamax: the parent of a leaf sees -value, alternating upward.
        for k, (parent, action) in enumerate(reversed(path)):
            r = -value if k % 2 == 0 else value
            parent.visits[action] += 1
            parent.sums[action] += r
            parent.returns[action].append(r)
            parent.node_returns.append(r)
    n = len(root.probs)
    q = [s / v if v else 0.0 for s, v in zip(root.sums, root.visits)]
    move = max(range(n), key=lambda i: (root.visits[i], q[i], root_eval.legal_log_priors[i], -i))
    policy = [v / cfg.simulations for v in root.visits]
    if cfg.target_policy == "rpo":
        # Unvisited actions take v_mix (Danihelka et al. 2022) under the clean prior.
        seen = [i for i in range(n) if root.visits[i]]
        mass = sum(clean[i] for i in seen)
        mixed = (root_eval.value_stm + cfg.simulations * sum(clean[i] * q[i] for i in seen) / mass) / (
            cfg.simulations + 1)
        completed = [q[i] if root.visits[i] else mixed for i in range(n)]
        c = cfg.cpuct + cfg.cpuct_factor * math.log((cfg.simulations + 1 + cfg.cpuct_base) / cfg.cpuct_base)
        policy = rpo(clean, completed, cfg.rpo_regularization * c * math.sqrt(cfg.simulations) / (n + cfg.simulations))
    return root_eval.legal_ucis[move], root.visits, q, policy, tuple(x / cfg.simulations for x in wdl_sums)


@pytest.mark.parametrize("algorithm", PUCT_ALGORITHMS)
@pytest.mark.parametrize("target", ["visits", "rpo"])
@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("exact", [False, True])
def test_native_search_matches_independent_reference(algorithm, target, noisy, fen, exact):
    board = chess.Board(fen)
    for simulations, depth in ((7, 3), (64, 4), (256, 6)):
        cfg = PuctConfig(algorithm=algorithm, simulations=simulations, max_depth=depth,
                         target_policy=target, rpo_regularization=0.25,
                         variance_c2=3.0 if algorithm == "puct_v" else 6.0,
                         lc0_parent_q=exact, paper_log_n=exact)
        eta, kwargs = None, dict(noise=0.0)
        if noisy:
            seed = f"{fen}:{simulations}"
            kwargs = dict(rng=random.Random(seed))
            rng = random.Random(seed)
            eta = [rng.gammavariate(cfg.dirichlet_alpha, 1.0) for _ in board.legal_moves]
            eta = [v / math.fsum(eta) for v in eta]
        got = select_puct(evaluator=HashEvaluator(), board=board, config=cfg, **kwargs)
        move, visits, q, policy, search_wdl = reference(board, cfg, eta)
        assert got.move_uci == move
        assert list(got.visits) == visits
        assert got.qvalues == pytest.approx(q, abs=1e-9)
        assert got.policy == pytest.approx(policy, abs=1e-7)
        assert got.search_wdl == pytest.approx(search_wdl, abs=1e-9)


def test_reference_discriminates_formulas():
    """Guard the guard: the hashed evaluator separates the three traversals."""
    board = chess.Board(FENS[3])
    visits = [reference(board, PuctConfig(algorithm=a, simulations=128, max_depth=6), None)[1]
              for a in PUCT_ALGORITHMS]
    assert visits[0] != visits[1] != visits[2] != visits[0]
    for algorithm, flag in (("puct", "lc0_parent_q"), ("puct_variance", "lc0_parent_q"),
                            ("puct_v", "lc0_parent_q"), ("puct_v", "paper_log_n")):
        cfg = PuctConfig(algorithm=algorithm, simulations=128, max_depth=6)
        assert reference(board, cfg, None)[1] != reference(board, replace(cfg, **{flag: True}), None)[1]


def test_paper_exact_options_keep_existing_run_identity():
    from imba_chess.self_play.config import load_config

    config = load_config("config/self_play_puct_variance.toml")
    # Without late sampling, the identifier of this config before the options existed
    # (main at 97b0fed), so the stopped 2026-10-18 run stays resumable.
    assert config.search.late_temperature
    config = replace(config, search=replace(config.search, late_temperature=0.0, late_value_cutoff=0.0))
    assert config.identifier == "92da5238fd7d246647ccb526323df6e80e3474668208c40097a3bf20c5a7f28d"
    assert replace(config, search=replace(config.search, lc0_parent_q=True)).identifier != config.identifier
    with pytest.raises(ValueError):
        PuctConfig(lc0_parent_q=1)
