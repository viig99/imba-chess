"""Native selections must preserve Python's compensated arithmetic and ties."""

import math
import random

import chess
import imba_chess_native as cc
import pytest

from imba_chess.eval.gumbel_search import (
    GumbelConfig,
    completed_q,
    interior_action,
    select_gumbel,
    softmax,
)
from tests.test_gumbel_search import FakeEvaluator


def test_native_q_and_fused_selections_match_python_exactly():
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
            value_scale=rng.choice([0.0, 0.1, 1.0]), epsilon=rng.choice([1e-8, 1e-14])
        )
        constants = (cfg.maxvisit_init, cfg.value_scale, cfg.epsilon)
        expected = completed_q(value, priors, visits, qs, cfg, probs)
        assert _gumbel_completed_q(value, visits, qs, probs, *constants) == expected
        assert cc.gumbel_interior_action(
            value, priors, visits, qs, probs, *constants
        ) == interior_action(value, priors, visits, qs, cfg, probs)
        noise = [0.0 if case % 11 == 0 else rng.uniform(-1e12, 10) for _ in range(n)]
        eligible = rng.choice(visits)
        max_prior = max(priors)
        action = max(
            (i for i in range(n) if visits[i] == eligible),
            key=lambda i: max(-1e9, noise[i] + priors[i] - max_prior + expected[i]),
        )
        assert (
            cc.gumbel_root_action(
                value, priors, visits, qs, probs, *constants, noise, eligible, max_prior
            )
            == action
        )


@pytest.mark.parametrize("budget", [1, 3, 128])
@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"])
def test_native_search_keeps_rng_visits_targets_and_counters(fen, budget):
    results = [
        select_gumbel(
            evaluator=FakeEvaluator(0.25),
            board=chess.Board(fen),
            config=GumbelConfig(simulations=budget),
            rng=random.Random(42),
            native_selection=native,
        )
        for native in (False, True)
    ]
    assert results[0] == results[1]


def test_native_rejects_malformed_inputs_and_ineligible_root():
    with pytest.raises(ValueError):
        cc.gumbel_interior_action(0.0, [0.0], [], [], [], 50.0, 0.1, 1e-8)
    with pytest.raises(ValueError, match="eligible"):
        cc.gumbel_root_action(
            0.0, [0.0], [0], [0.0], [1.0], 50.0, 0.1, 1e-8, [0.0], 1, 0.0
        )
    with pytest.raises(ValueError):
        cc.gumbel_interior_action(math.nan, [0.0], [0], [0.0], [1.0], 50.0, 0.1, 1e-8)
