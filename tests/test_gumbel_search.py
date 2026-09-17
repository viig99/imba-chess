import json
import math
from pathlib import Path
import random

import chess
import pytest

from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval import cozy_bridge
from imba_chess.eval.gumbel_search import (
    GumbelConfig,
    completed_q,
    considered_visits,
    select_gumbel,
    REFERENCE_REVISION,
)
from imba_chess.eval.search import PositionEval


class FakeEvaluator:
    def __init__(self, value=0.0):
        self.value = value
        self.calls = []
        self.vocab = MoveVocab.load("artifacts/move_vocab_static_uci.json")

    def extend(self, handle, uci, move_vocab_id=None):
        return (handle or ()) + (uci,)

    def evaluate(self, batch):
        result = []
        for handle, board in batch:
            assert handle not in self.calls
            self.calls.append(handle)
            ids, moves, ucis, forcing, total = cozy_bridge.project_legal_moves(
                board, self.vocab
            )
            assert total == len(ids)
            result.append(
                PositionEval(
                    self.value,
                    moves,
                    ucis,
                    [-math.log(len(ids))] * len(ids),
                    forcing,
                    ids,
                )
            )
        return result


@pytest.mark.parametrize("budget", [1, 2, 3, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("candidates", [1, 3, 8, 16, 64])
def test_exact_budgets(budget, candidates):
    evaluator = FakeEvaluator(0.25)
    result = select_gumbel(
        evaluator=evaluator,
        board=chess.Board(),
        rng=random.Random(42),
        config=GumbelConfig(simulations=budget, top_m=candidates, max_depth=2),
    )
    assert sum(result.visits) == result.simulations == budget
    assert result.neural_evaluations == len(evaluator.calls)
    assert result.maximum_depth <= 2
    assert sum(result.policy) == pytest.approx(1)
    assert len(result.policy) == 20
    assert sum(n > 0 for n in result.visits) <= min(candidates, budget, 20)


def test_schedule_odd_and_tiny():
    assert considered_visits(3, 8) == (0, 0, 0, 1, 1, 2, 2, 3)
    assert considered_visits(1, 3) == (0, 1, 2)
    assert considered_visits(3, 2) == (0, 0)


def test_mate_and_terminal_revisits():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    ev = FakeEvaluator()
    root = ev.evaluate([(None, cozy_bridge.board_to_cozy(board))])[0]
    noise = [100 if uci == "f7g7" else 0 for uci in root.legal_ucis]
    result = select_gumbel(
        evaluator=ev,
        board=board,
        root_eval=root,
        noise=noise,
        config=GumbelConfig(simulations=16, top_m=1),
    )
    assert result.move_uci == "f7g7"
    assert result.qvalues[root.legal_ucis.index("f7g7")] == 1
    assert result.terminal_hits == 16
    assert result.neural_evaluations == 1


def test_depth_reuse_and_signs():
    ev = FakeEvaluator(0.5)
    result = select_gumbel(
        evaluator=ev,
        board=chess.Board(),
        noise=[0] * 20,
        config=GumbelConfig(simulations=8, top_m=1, max_depth=1),
    )
    assert result.qvalues[0] == -0.5
    assert result.neural_evaluations == 2
    assert result.depth_cutoffs == 8
    ev = FakeEvaluator(0.5)
    result = select_gumbel(
        evaluator=ev,
        board=chess.Board(),
        noise=[0] * 20,
        config=GumbelConfig(simulations=2, top_m=1, max_depth=2),
    )
    assert result.qvalues[0] == 0  # (-.5 + .5) / 2


def test_noise_not_added_to_target():
    a = select_gumbel(
        evaluator=FakeEvaluator(),
        board=chess.Board(),
        noise=[0] * 20,
        config=GumbelConfig(simulations=1),
    )
    b = select_gumbel(
        evaluator=FakeEvaluator(),
        board=chess.Board(),
        noise=list(range(20)),
        config=GumbelConfig(simulations=1),
    )
    assert a.move_id != b.move_id
    assert a.policy == b.policy == pytest.approx([0.05] * 20)


def test_completed_q_and_deficit():
    cfg = GumbelConfig()
    assert completed_q(0.8, [0, 0], [0, 0], [0, 0], cfg) == [0, 0]
    assert completed_q(0.2, [0, 0, 0], [2, 0, 1], [1, 0, -1], cfg) == pytest.approx(
        [5.2, 2.73, 0]
    )
    assert all(
        math.isfinite(v) for v in completed_q(0, [0, -10000], [0, 1], [0, 1], cfg)
    )
    assert interior_action(0, [0, 0], [1, 0], [0, 0], cfg) == 1


def test_terminal_takeover_repetition():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push_uci(uci)
    with pytest.raises(ValueError, match="terminal"):
        select_gumbel(evaluator=FakeEvaluator(), board=board)


def test_upstream_q_fixtures():
    fixture = json.loads(Path("tests/fixtures/gumbel/qtransform.json").read_text())
    assert fixture["revision"] == REFERENCE_REVISION
    for row in fixture["cases"]:
        assert completed_q(
            row["value"], row["priors"], row["visits"], row["qvalues"], GumbelConfig()
        ) == pytest.approx(row["expected"], abs=2e-5)


def test_fixed_noise_upstream_search_fixtures():
    fixture = json.loads(Path("tests/fixtures/gumbel/search.json").read_text())
    assert fixture["revision"] == REFERENCE_REVISION
    for row in fixture["cases"]:
        evaluator = FakeEvaluator()
        root = evaluator.evaluate([(None, cozy_bridge.board_to_cozy(chess.Board()))])[0]
        root = root._replace(
            value_stm=fixture["root_value"], legal_log_priors=fixture["priors"]
        )
        base_evaluate = evaluator.evaluate

        def evaluate(batch):
            result = base_evaluate(batch)
            return [
                ev._replace(
                    value_stm=-fixture["returns"][root.legal_ucis.index(handle[0])]
                )
                for (handle, _), ev in zip(batch, result)
            ]

        evaluator.evaluate = evaluate
        result = select_gumbel(
            evaluator=evaluator,
            board=chess.Board(),
            root_eval=root,
            noise=fixture["noise"],
            config=GumbelConfig(
                simulations=row["budget"], top_m=row["top_m"], max_depth=1
            ),
        )
        assert result.visits == row["visits"]
        assert result.move_uci == root.legal_ucis[row["action"]]
        assert result.policy == pytest.approx(row["policy"], abs=2e-6)


def interior_action(value, priors, visits, qvalues, config, prior_probs=None):
    transformed = completed_q(value, priors, visits, qvalues, config, prior_probs)
    logits = [p + q for p, q in zip(priors, transformed)]
    weights = [math.exp(x - max(logits)) for x in logits]
    policy = [x / math.fsum(weights) for x in weights]
    denominator = 1 + sum(visits)
    return max(range(len(priors)), key=lambda i: policy[i] - visits[i] / denominator)
