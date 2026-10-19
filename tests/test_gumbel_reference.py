"""End-to-end check of the native Gumbel search against an independent reference.

The reference transcribes mctx (Danihelka et al. 2022, "Policy improvement by
planning with Gumbel") rather than gumbel.rs: the sequential-halving schedule,
root selection max(-1e9, g + logits + sigma(qhat)) among moves at the considered
visit count, interior selection softmax(logits + sigma(qhat)) - N / (1 + sum N),
completed Q with v_mix, the final action and the improved policy. Hashed leaf
values and peaked priors (from test_puct_reference) make the formulas separable;
the mutant test checks that they do.
"""

import math
import random

import chess
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval.gumbel_search import GumbelConfig, select_gumbel
from tests.test_puct_reference import FENS, HashEvaluator


TINY = 1.1754943508222875e-38  # float32 tiny, mctx's prior clamp


def mctx_sequence(m, n):
    """mctx seq_halving.get_sequence_of_considered_visits."""
    if m <= 1:
        return list(range(n))
    log2max = int(math.ceil(math.log2(m)))
    sequence, visits, considered = [], [0] * m, m
    while len(sequence) < n:
        for _ in range(max(1, int(n / (log2max * considered)))):
            sequence.extend(visits[:considered])
            for i in range(considered):
                visits[i] += 1
        considered = max(2, considered // 2)
    return sequence[:n]


def softmax(x):
    top = max(x)
    e = [math.exp(v - top) for v in x]
    return [v / sum(e) for v in e]


class Node:
    def __init__(self, board, evaluation=None, terminal=None):
        self.board, self.evaluation, self.terminal, self.children = board, evaluation, terminal, {}
        if evaluation is not None:
            n = len(evaluation.legal_ucis)
            self.logits = list(evaluation.legal_log_priors)
            self.probs = [max(p, TINY) for p in softmax(self.logits)]
            self.visits, self.sums = [0] * n, [0.0] * n
            self.edge_wdl = [[0.0] * 3 for _ in range(n)]


def sigma_qhat(node, cfg):
    """mctx qtransform_completed_by_mix_value."""
    visits = node.visits
    q = [s / n if n else 0.0 for s, n in zip(node.sums, visits)]
    total = sum(visits)
    seen = [i for i, n in enumerate(visits) if n]
    weighted = sum(node.probs[i] * q[i] for i in seen) / sum(node.probs[i] for i in seen) if seen else 0.0
    mixed = (node.evaluation.value_stm + total * weighted) / (total + 1)
    completed = [q[i] if visits[i] else mixed for i in range(len(visits))]
    if cfg.rescale_values:
        low, high = min(completed), max(completed)
        completed = [(v - low) / max(high - low, cfg.epsilon) for v in completed]
    return [(cfg.maxvisit_init + max(visits)) * cfg.value_scale * v for v in completed]


def root_pick(node, noise, considered_visit, cfg, transform=sigma_qhat):
    sigma = transform(node, cfg)
    top = max(node.logits)
    best, action = -math.inf, None
    for i, n in enumerate(node.visits):
        if n == considered_visit:
            score = max(-1e9, noise[i] + node.logits[i] - top + sigma[i])
            if score > best:
                best, action = score, i
    return action


def interior_pick(node, cfg, transform=sigma_qhat):
    policy = softmax([l + s for l, s in zip(node.logits, transform(node, cfg))])
    total = sum(node.visits)
    scores = [p - n / (1 + total) for p, n in zip(policy, node.visits)]
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def terminal_value(board):
    """Side-to-move value under the game's own adjudication (claimable draws end it)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)


def reference(board, cfg, noise, schedule=mctx_sequence, interior=interior_pick, transform=sigma_qhat):
    evaluate = HashEvaluator().evaluate
    (root_eval,) = evaluate([((), cozy_bridge.board_to_cozy(board))])
    root = Node(board.copy(), root_eval)
    n = len(root.visits)
    sequence = schedule(min(cfg.top_m, n), cfg.simulations)
    wdl_sums = [0.0] * 3
    for k in range(cfg.simulations):
        node, path, handle = root, [], ()
        action = root_pick(root, noise, sequence[k], cfg, transform)
        while True:
            path.append((node, action))
            uci = node.evaluation.legal_ucis[action]
            if action not in node.children:
                child_board = node.board.copy()
                child_board.push_uci(uci)
                value = terminal_value(child_board)
                if value is None:
                    (evaluation,) = evaluate([(handle + (uci,), cozy_bridge.board_to_cozy(child_board))])
                    node.children[action] = Node(child_board, evaluation)
                else:
                    node.children[action] = Node(child_board, terminal=value)
                leaf = node.children[action]
                break
            node, handle = node.children[action], handle + (uci,)
            if node.evaluation is None or len(path) == cfg.max_depth:
                leaf = node
                break
            action = interior(node, cfg, transform)
        if leaf.evaluation is None:
            value = leaf.terminal
            wdl = (float(value == -1), float(value == 0), float(value == 1))  # loss, draw, win
        else:
            value, wdl = leaf.evaluation.value_stm, leaf.evaluation.wdl
        for i, probability in enumerate(wdl[::-1] if len(path) % 2 else wdl):
            wdl_sums[i] += probability
            root.edge_wdl[path[0][1]][i] += probability
        # Negamax: the parent of a leaf sees -value, alternating upward.
        for j, (parent, edge) in enumerate(reversed(path)):
            parent.visits[edge] += 1
            parent.sums[edge] += -value if j % 2 == 0 else value
    move = root_pick(root, noise, max(root.visits), cfg, transform)
    policy = softmax([l + s for l, s in zip(root.logits, transform(root, cfg))])
    # improved_wdl: policy-weighted per-move WDL; unvisited moves take v_mix's WDL form.
    seen = [i for i in range(n) if root.visits[i]]
    mass, total = sum(root.probs[i] for i in seen), sum(root.visits)
    means = [[x / root.visits[i] for x in root.edge_wdl[i]] if root.visits[i] else None for i in range(n)]
    mixed = [(root_eval.wdl[c] + total * sum(root.probs[i] * means[i][c] for i in seen) / mass) / (total + 1)
             for c in range(3)]
    improved = [sum(policy[i] * (means[i] or mixed)[c] for i in range(n)) for c in range(3)]
    return dict(move=root_eval.legal_ucis[move], visits=root.visits,
                q=[s / v if v else 0.0 for s, v in zip(root.sums, root.visits)], policy=policy,
                search_wdl=[x / cfg.simulations for x in wdl_sums], improved_wdl=improved)


def gumbel_noise(board, seed):
    rng = random.Random(seed)
    return [-math.log(-math.log(max(rng.random(), 1e-12))) for _ in board.legal_moves]


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("fen", FENS)
def test_native_search_matches_independent_reference(fen, noisy, rescale):
    board = chess.Board(fen)
    # Budgets below top_m, uneven halving rounds and non-power-of-two top_m.
    for simulations, top_m, depth in ((1, 1, 3), (7, 4, 3), (16, 16, 4), (64, 8, 6),
                                      (100, 5, 6), (256, 16, 8), (300, 32, 6)):
        cfg = GumbelConfig(simulations=simulations, top_m=top_m, max_depth=depth,
                           rescale_values=rescale, value_scale=0.1 if rescale else 0.5)
        noise = gumbel_noise(board, f"{fen}:{simulations}:{top_m}") if noisy else [0.0] * board.legal_moves.count()
        got = select_gumbel(evaluator=HashEvaluator(), board=board, config=cfg, noise=noise)
        ref = reference(board, cfg, noise)
        assert got.move_uci == ref["move"]
        assert list(got.visits) == ref["visits"]
        assert got.qvalues == pytest.approx(ref["q"], abs=1e-9)
        assert got.policy == pytest.approx(ref["policy"], abs=1e-9)
        assert got.search_wdl == pytest.approx(ref["search_wdl"], abs=1e-9)
        assert got.improved_wdl == pytest.approx(ref["improved_wdl"], abs=1e-9)


def test_reference_detects_formula_mutants():
    """Guard the guard: plausible wrong formulas must change the search's root Q."""

    def no_visit_term(node, cfg, transform):
        policy = softmax([l + s for l, s in zip(node.logits, transform(node, cfg))])
        return max(range(len(policy)), key=lambda i: (policy[i], -i))

    def raw_value_fill(node, cfg):
        scale = (cfg.maxvisit_init + max(node.visits)) * cfg.value_scale
        return [scale * node.evaluation.value_stm if n == 0 else s
                for s, n in zip(sigma_qhat(node, cfg), node.visits)]

    mutants = dict(interior=dict(interior=no_visit_term), transform=dict(transform=raw_value_fill),
                   schedule=dict(schedule=lambda m, n: [i // m for i in range(n)]))
    # Quiet roots exercise every rule; mate-in-1 roots can bypass interior choice.
    quiet = [chess.Board(FENS[0]), chess.Board(FENS[3])]
    configs = [GumbelConfig(simulations=64, top_m=8, max_depth=6),
               GumbelConfig(simulations=256, top_m=16, max_depth=8)]
    for name, mutant in mutants.items():
        for board in quiet:
            for cfg in configs:
                noise = gumbel_noise(board, board.fen())
                got = select_gumbel(evaluator=HashEvaluator(), board=board, config=cfg, noise=noise)
                ref = reference(board, cfg, noise, **mutant)
                # Halving fixes most root visit counts; the subtrees' Q values carry the change.
                assert max(abs(a - b) for a, b in zip(got.qvalues, ref["q"])) > 1e-6, (
                    name, board.fen(), cfg.simulations)
