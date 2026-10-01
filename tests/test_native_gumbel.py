"""Native selections must preserve Python's compensated arithmetic and ties."""

import math
import random

import chess
import imba_chess_native as cc
import pytest

from imba_chess.eval.gumbel_search import (
    GumbelConfig,
    completed_q,
    select_gumbel,
    softmax,
)
from tests.test_gumbel_search import FakeEvaluator


@pytest.mark.parametrize("rescale", [True, False])
def test_native_q_matches_python_exactly(rescale):
    from imba_chess_native.imba_chess_native import _gumbel_completed_q

    rng = random.Random(721)
    for case in range(1200):
        n = rng.choice([1, 2, 3, 20, 67, 218])
        priors = [rng.uniform(-10000 if case % 5 == 0 else -30, 0) for _ in range(n)]
        visits = [rng.choice([0, 0, 1, 2, 16, 128]) for _ in range(n)]
        qs = [rng.uniform(-1, 1) for _ in range(n)]
        if case % 7 == 0:
            qs = [1 - rng.random() * 1e-12 for _ in range(n)]
        if case % 11 == 0:
            priors, qs, visits = [0.0] * n, [0.0] * n, [0] * n
        value = rng.uniform(-1, 1)
        probs = [max(p, 1.1754943508222875e-38) for p in softmax(priors)]
        cfg = GumbelConfig(
            value_scale=rng.choice([0.0, 0.1, 1.0]), epsilon=rng.choice([1e-8, 1e-14]), rescale_values=rescale)
        constants = (cfg.maxvisit_init, cfg.value_scale, cfg.epsilon)
        expected = completed_q(value, priors, visits, qs, cfg, probs)
        assert _gumbel_completed_q(value, visits, qs, probs, *constants, rescale_values=rescale) == expected


def test_raw_q_is_default_and_skips_min_max():
    from imba_chess_native.imba_chess_native import _gumbel_completed_q

    cfg = GumbelConfig()
    assert (cfg.rescale_values, cfg.value_scale) == (False, 0.5)
    # Visited Q stays raw (no min-max), scaled by (maxvisit_init + max visits) * value_scale;
    # the unvisited move takes the prior-weighted mixed value.
    visits, qs, probs = [3, 1, 0], [0.12, 0.10, 0.0], [0.5, 0.3, 0.2]
    mixed = (0.0 + 4 * (0.5 * 0.12 + 0.3 * 0.10) / 0.8) / 5
    expected = [53 * 0.5 * q for q in (0.12, 0.10, mixed)]
    assert completed_q(0.0, [0.0] * 3, visits, qs, cfg, probs) == pytest.approx(expected)
    assert _gumbel_completed_q(0.0, visits, qs, probs, 50.0, 0.5, 1e-8) == pytest.approx(expected)
    # A near-tie keeps its small gap instead of being stretched to the full range.
    rescaled = completed_q(0.0, [0.0] * 3, visits, qs, GumbelConfig(rescale_values=True), probs)
    assert expected[0] - expected[1] < 1 < rescaled[0] - rescaled[1]
    with pytest.raises(ValueError):
        GumbelConfig(rescale_values=1)


@pytest.mark.parametrize("budget", [1, 3, 128])
@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"])
def test_native_search_keeps_rng_visits_targets_and_counters(fen, budget):
    import json
    from dataclasses import asdict
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/gumbel/native_search_baseline.json"
        ).read_text()
    )
    expected = next(
        row["result"]
        for row in fixture["cases"]
        if row["fen"] == fen and row["budget"] == budget
    )
    result = select_gumbel(
        evaluator=FakeEvaluator(0.25),
        board=chess.Board(fen),
        config=GumbelConfig(simulations=budget, rescale_values=True, value_scale=0.1),
        rng=random.Random(42),
    )
    actual = json.loads(json.dumps(asdict(result)))
    # The historical fixture predates the measurement-only WDL field.
    # Its backup semantics are covered by test_auxiliary_value; all original
    # search actions, targets, visits and counters must remain identical.
    actual.pop("search_wdl")
    assert actual == expected


def test_native_rejects_malformed_inputs_and_ineligible_root():
    with pytest.raises(ValueError):
        cc.NodeStats(0.0, [0.0], [])
    node = cc.NodeStats(0.0, [0.0], [1.0])
    node.set_noise([0.0])
    with pytest.raises(ValueError, match="eligible"):
        node.root(1, 50.0, 0.1, 1e-8, rescale_values=True)
    with pytest.raises(ValueError):
        cc.NodeStats(math.nan, [0.0], [1.0])


def test_native_node_priors_match_python_softmax_exactly():
    rng = random.Random(1311)
    for case in range(3000):
        n = rng.choice([1, 2, 3, 20, 67, 218])
        scale = rng.choice([0.01, 2.0, 30.0, 800.0])
        logits = [rng.gauss(0, scale) for _ in range(n)]
        if case % 9 == 0:
            logits = [rng.choice([-1.5, 0.0, 3.25]) for _ in range(n)]  # ties
        m = max(logits)
        lse = m + math.log(math.fsum(math.exp(x - m) for x in logits))
        priors = [x - lse for x in logits]
        stats = cc.NodeStats.from_evaluation(0.25, priors)
        assert stats.probs == [max(p, 1.1754943508222875e-38) for p in softmax(priors)]



INVALID_EVALUATION = "nonfinite or invalid network evaluation"
INVALID_WDL = "invalid or inconsistent evaluation WDL"
INVALID_PROJECTION = "nonterminal evaluation has invalid legal projection"


@pytest.mark.parametrize(
    "value, priors, wdl, forcing, expected",
    [
        (0.0, [0.0], None, [False], None),
        (0.2, [0.0, -1.0], (0.2, 0.4, 0.4), [False, True], None),
        (math.nan, [0.0], None, [False], INVALID_EVALUATION),
        (1.00001, [0.0], None, [False], INVALID_EVALUATION),
        (0.0, [0.0, math.inf], None, [False, False], INVALID_EVALUATION),
        (0.0, [0.0], (0.5, 0.5), [False], INVALID_WDL),
        (0.0, [0.0], (0.5, math.nan, 0.5), [False], INVALID_WDL),
        (0.0, [0.0], (-0.1, 0.6, 0.5), [False], INVALID_WDL),
        (0.0, [0.0], (0.4, 0.4, 0.4), [False], INVALID_WDL),
        (0.3, [0.0], (0.2, 0.6, 0.2), [False], INVALID_WDL),
        (0.0, [0.0], None, [], INVALID_PROJECTION),
        # Evaluation errors take precedence over a malformed forcing mask.
        (math.nan, [0.0], None, [], INVALID_EVALUATION),
        (0.0, [0.0], (0.4, 0.4, 0.4), [], INVALID_WDL),
    ],
)
def test_node_initialize_validates_evaluations(value, priors, wdl, forcing, expected):
    from imba_chess.eval.gumbel_search import _Node
    from imba_chess.eval.search import PositionEval

    n = len(priors)
    evaluation = PositionEval(value, [None] * n, ["a"] * n, priors, forcing, list(range(n)), wdl=wdl)
    node = _Node(None, [], None)
    if expected is None:
        node.initialize(evaluation, GumbelConfig(forcing_floor=True, minimax_weight=0.5, rescale_values=True, value_scale=0.1))
        assert node.value == value and node.priors == priors
        assert node.stats.probs == [max(p, 1.1754943508222875e-38) for p in softmax(priors)]
    else:
        with pytest.raises(ValueError, match=expected):
            node.initialize(evaluation)
