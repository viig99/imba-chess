"""Audit checks using normalized priors, including a known FP32 edge case."""
import math

import pytest
import torch

from imba_chess.eval.gumbel_search import softmax
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.dataset import policy_weights


def test_surprise_kl_with_valid_normalized_distributions():
    targets = [dict(policy=[.5, .5], root_log_priors=[math.log(.5)] * 2),
               dict(policy=[1., 0.], root_log_priors=[-2., math.log1p(-math.exp(-2.))])]
    for t in targets:
        assert sum(math.exp(x) for x in t['root_log_priors']) == pytest.approx(1.)
    result = policy_weights(targets, learning=LearningConfig(policy_surprise_enabled=True))
    assert result['policy_surprise'] == pytest.approx([0., 2.])
    assert result['policy_training_weight'] == pytest.approx([.5, 1.5])


def test_unchanged_policy_is_not_reweighted_by_fp32_roundoff():
    targets = []
    for n in [1, 2, 3, 4, 5, 8, 12, 16, 20, 30, 32, 40, 50, 60]:
        logs = torch.zeros(n, dtype=torch.float32).log_softmax(-1).tolist()
        # This is precisely the production target when completed Q is constant.
        targets.append(dict(policy=softmax(logs), root_log_priors=logs))
    result = policy_weights(targets, learning=LearningConfig(policy_surprise_enabled=True))
    assert result['policy_training_weight'] == pytest.approx([1.] * len(targets), abs=1e-6)


def test_surprise_is_invariant_to_actor_logit_offset():
    logs = [math.log(.2), math.log(.8)]
    target = [.75, .25]
    expected = .75 * math.log(.75 / .2) + .25 * math.log(.25 / .8)
    for offset in [0., -2., 1000.]:
        result = policy_weights([dict(policy=target, root_log_priors=[x + offset for x in logs])])
        assert result['policy_surprise'][0] == pytest.approx(expected)


def test_tiny_positive_surprise_is_not_lost_with_the_roundoff_floor():
    target = [.5 + 1e-5, .5 - 1e-5]
    result = policy_weights([dict(policy=target, root_log_priors=[math.log(.5)] * 2)])
    assert result['policy_surprise'][0] == pytest.approx(2e-10, rel=1e-5, abs=1e-15)
