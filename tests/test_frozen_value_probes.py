import numpy as np
import pytest
import torch

from scripts.audit_frozen_value_probes import (
    fit_linear, game_weights, grouped_summary, predict, soft_ce, split_games, target_ldw,
)


def test_engine_target_order_and_both_player_perspectives():
    assert target_ldw([700, 200, 100], True, True) == [.1, .2, .7]
    assert target_ldw([700, 200, 100], False, True) == [.7, .2, .1]
    assert target_ldw([0, 0, 1000], False, False) == [1., 0., 0.]
    with pytest.raises(ValueError):
        target_ldw([1, 2, 3], True, True)


def test_nested_game_splits_never_separate_branches_or_neighbors():
    rows = [dict(source=f'game{i}', arm=arm, ply=ply)
            for i in range(42) for arm in ('bad', 'good') for ply in range(4)]
    plans = split_games(rows)
    assert plans == split_games(rows)
    seen = set()
    for plan in plans:
        train, val, test = map(set, (plan['train'], plan['validation'], plan['test']))
        assert not train & val and not train & test and not val & test
        assert len(train | val | test) == 42
        assert not seen & test
        seen |= test
    assert len(seen) == 42


def test_each_source_game_has_equal_training_weight():
    rows = [dict(source='a'), dict(source='a'), dict(source='b')]
    assert game_weights(rows, [0, 1, 2]).tolist() == [.25, .25, .5]
    assert game_weights(rows, [0, 1]).tolist() == [.5, .5]


def test_probe_recovers_known_heldout_signal_and_preprocessing_uses_only_training():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(120, 4, generator=rng, dtype=torch.float64)
    w = torch.tensor([[1., -.3, .2, 0], [0, .5, 0, .2], [-1, -.2, -.2, -.2]], dtype=torch.float64)
    y = (x @ w.T).softmax(-1)
    fitted = fit_linear(x[:100], y[:100], torch.full((100,), .01, dtype=torch.float64), .001)
    assert torch.allclose(fitted['mean'], x[:100].mean(0))
    assert (predict(fitted, x[100:]).softmax(-1) - y[100:]).abs().mean() < .01
    assert fitted['diagnostics']['gradient_max'] < 1e-4


def test_soft_target_cross_entropy_and_game_cluster_summary():
    p = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    y = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    assert soft_ce(p.log(), y).item() == pytest.approx(-np.log(.5))
    rows = [dict(source='a')] * 9 + [dict(source='b')]
    out = grouped_summary(np.array([0.] * 9 + [1.]), rows)
    assert out['mean'] == .5  # equal games, not equal rows
    assert out['position_mean'] == .1
    assert out['ci95'] == [0., 1.]
