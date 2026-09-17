import random

import pytest
from imba_chess_native import NodeStats, gumbel_backup as backup

from imba_chess.eval.gumbel_search import (
    softmax,
    completed_q,
    interior_action,
    GumbelConfig,
)


def test_native_stats_selections_and_backup_match_reference_exactly():
    rng = random.Random(815)
    constants = (50.0, 0.1, 1e-8)
    for n in (1, 2, 20, 67, 218):
        priors = [rng.uniform(-40, 0) for _ in range(n)]
        probs = [max(p, 1.1754943508222875e-38) for p in softmax(priors)]
        value = rng.uniform(-1, 1)
        noise = [rng.uniform(-10, 10) for _ in range(n)]
        nodes = [NodeStats(value, priors, probs) for _ in range(4)]
        for node in nodes:
            node.set_noise(noise)
        visits, sums, means = (
            [[0] * n for _ in nodes],
            [[0.0] * n for _ in nodes],
            [[0.0] * n for _ in nodes],
        )
        for _ in range(256):
            edges = [rng.randrange(n) for _ in nodes]
            leaf = rng.uniform(-1, 1)
            backup(list(zip(nodes, edges)), leaf)
            for i in reversed(range(len(nodes))):
                leaf = -leaf
                edge = edges[i]
                visits[i][edge] += 1
                sums[i][edge] += leaf
                means[i][edge] = sums[i][edge] / visits[i][edge]
                node = nodes[i]
                assert node.snapshot() == (visits[i], sums[i], means[i])
                assert node.interior(*constants) == interior_action(
                    value, priors, visits[i], means[i], GumbelConfig(), probs
                )
                eligible = visits[i][rng.randrange(n)]
                q = completed_q(
                    value, priors, visits[i], means[i], GumbelConfig(), probs
                )
                expected = max(
                    (j for j in range(n) if visits[i][j] == eligible),
                    key=lambda j: max(-1e9, noise[j] + priors[j] - max(priors) + q[j]),
                )
                assert node.root(eligible, *constants) == expected


def test_native_stats_ties_and_atomic_validation():
    node = NodeStats(0.0, [0.0, 0.0], [0.5, 0.5])
    other = NodeStats(0.0, [0.0], [1.0])
    node.set_noise([0.0, 0.0])
    assert node.interior(50, 0.1, 1e-8) == 0
    assert node.root(0, 50, 0.1, 1e-8) == 0
    before = node.snapshot()
    for path in ([(node, 0), (other, 2)], [(node, 0), (node, 1)]):
        with pytest.raises(ValueError):
            backup(path, 0.5)
        assert node.snapshot() == before
