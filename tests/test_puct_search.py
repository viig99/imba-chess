import math
import random

import chess
import imba_chess_native as cc
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval.puct_search import PuctConfig, PUCT_ALGORITHMS, select_puct
from imba_chess.self_play.config import SelfPlayConfig, search_config
from tests.test_gumbel_search import FakeEvaluator


@pytest.mark.parametrize("algorithm", PUCT_ALGORITHMS)
@pytest.mark.parametrize("budget", [1, 2, 17, 64])
def test_batched_search_contract(algorithm, budget):
    ev = FakeEvaluator(0.25)
    result = select_puct(evaluator=ev, board=chess.Board(), noise=0.0,
                         config=PuctConfig(algorithm=algorithm, simulations=budget, max_depth=3))
    assert sum(result.visits) == result.simulations == budget
    assert result.neural_evaluations == len(ev.calls)
    assert result.maximum_depth <= 3
    assert result.policy == pytest.approx([n / budget for n in result.visits])
    assert result.move_uci in {m.uci() for m in chess.Board().legal_moves}


def native_stats(mode=0, probs=(0.6, 0.4)):
    stats = cc.NodeStats.from_evaluation(0.1, [math.log(p) for p in probs])
    stats.configure_puct(mode, 1.75, 38739.0, 3.89, 0.33, math.sqrt(2), 6.0, 0.25, 8.0)
    return stats


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_native_selection_matches_scalar_reference(mode):
    stats = native_stats(mode)
    values = [(0, 0.2), (0, -0.4), (1, 0.8), (0, 0.3), (1, -0.6)]
    observations = [[], []]
    for edge, value in values:
        cc.gumbel_backup([(stats, edge)], -value)
        observations[edge].append(value)
        visits, sums, means = stats.snapshot()
        count = sum(visits)
        parent_q = sum(sums) / count
        fpu = parent_q - 0.33 * math.sqrt(sum(p for p, n in zip((0.6, 0.4), visits) if n))
        c = 1.75 + 3.89 * math.log((count + 38739 + 1) / 38739)
        flat = [v for group in observations for v in group]
        if mode == 1:
            avg = sum(flat) / len(flat)
            m2 = sum((v - avg) ** 2 for v in flat)
            c *= math.sqrt(((m2 + 8 * 0.25) / (count + 8)) / 0.25)
        scores = []
        for i, p in enumerate((0.6, 0.4)):
            n = visits[i]
            q = means[i] if n else fpu
            if mode == 2:
                variance = sum((v - means[i]) ** 2 for v in observations[i]) / n if n else 0
                bonus = p * (math.sqrt(2 * variance * count) + 6 * math.log(count + 1)) / (1 + n)
            else:
                bonus = c * p * math.sqrt(count) / (1 + n)
            scores.append(q + bonus)
        assert stats.select_puct() == max(range(2), key=scores.__getitem__)
    if mode:
        m2, n, avg, total_m2 = stats.variance_snapshot()
        assert n == len(flat)
        assert avg == pytest.approx(sum(flat) / n)
        assert total_m2 == pytest.approx(sum((v - avg) ** 2 for v in flat))
        assert m2 == pytest.approx([sum((v - means[i]) ** 2 for v in group) for i, group in enumerate(observations)])
    else:
        assert stats.variance_snapshot()[0] == []


@pytest.mark.parametrize("algorithm", PUCT_ALGORITHMS)
def test_terminal_backup_and_wdl(algorithm):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    ev = FakeEvaluator()
    root = ev.evaluate([(None, cozy_bridge.board_to_cozy(board))])[0]
    priors = [0 if uci == "f7g7" else -100 for uci in root.legal_ucis]
    root = root._replace(legal_log_priors=priors, wdl=(0, 1, 0))
    result = select_puct(evaluator=ev, board=board, root_eval=root, noise=0.0,
                         config=PuctConfig(algorithm=algorithm, simulations=16))
    assert result.move_uci == "f7g7"
    assert result.terminal_hits == 16
    assert result.neural_evaluations == 1
    assert result.search_wdl == (0, 0, 1)


def test_seeded_exploration_keeps_clean_priors():
    def run(seed):
        return select_puct(evaluator=FakeEvaluator(), board=chess.Board(), rng=random.Random(seed),
                           config=PuctConfig(simulations=32, max_depth=2))
    a, b, c = run(1), run(1), run(2)
    assert a == b
    assert a.root_log_priors == c.root_log_priors
    assert a.visits != c.visits


def test_configuration_preserves_gumbel_default():
    from imba_chess.eval.gumbel_search import GumbelConfig

    assert search_config() == GumbelConfig()
    assert search_config(algorithm="gumbel") == GumbelConfig()
    assert isinstance(search_config(algorithm="puct_v"), PuctConfig)
    assert SelfPlayConfig(search=PuctConfig()).identifier != SelfPlayConfig().identifier
    with pytest.raises(ValueError):
        PuctConfig(algorithm="typo")
    with pytest.raises(ValueError):
        PuctConfig(variance_prior=0)


def test_rpo_is_the_reverse_kl_optimizer_and_preserves_equal_q_priors():
    from imba_chess.eval.puct_search import rpo_policy

    priors, q, lam = [0.2, 0.3, 0.5], [-0.6, 0.1, 0.8], 0.2
    policy = rpo_policy(priors, q, lam)
    assert sum(policy) == pytest.approx(1)
    # KKT stationarity: Q + lambda * P / policy is the same for each action.
    alphas = [v + lam * p / y for p, v, y in zip(priors, q, policy)]
    assert max(alphas) - min(alphas) < 1e-12
    assert policy[2] > priors[2]
    assert rpo_policy(priors, [0, 0, 0], lam) == priors


def test_rpo_target_does_not_change_tree_or_played_move():
    from dataclasses import replace

    cfg = PuctConfig(simulations=32, max_depth=3)
    def run(config):
        return select_puct(evaluator=FakeEvaluator(0.2), board=chess.Board(), noise=0.0, config=config)
    a, b = run(cfg), run(replace(cfg, target_policy="rpo"))
    assert a.visits == b.visits
    assert a.qvalues == b.qvalues
    assert a.move_uci == b.move_uci
    assert sum(b.policy) == pytest.approx(1)
    assert b.policy != a.policy
