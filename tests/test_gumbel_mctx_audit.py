"""Multi-depth goldens from actual DeepMind mctx, with production native stats.

Only environment transitions and neural evaluations are synthetic. Sequential
halving, interior selection, backups, depth cutoff and targets use production code.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from imba_chess.eval.gumbel_search import GumbelConfig, REFERENCE_REVISION, select_gumbel
from imba_chess.eval.search import PositionEval

FIXTURE = Path(__file__).parent / 'fixtures/gumbel/multidepth_mctx.json'


def run_synthetic(*, k, seed, budget, top_m, depth, terminal, scale, noise):
    def push(board, action, history, color):
        node, d = board
        child = node * k + action + 1
        outcome = -1. if terminal and d + 1 >= 2 and child % 11 == 0 else None
        return (child, d + 1), history, outcome

    class Evaluator:
        def extend(self, handle, uci, move_vocab_id=None):
            return None

        def evaluate(self, batch):
            out = []
            for _, (node, _) in batch:
                logits = [((node + 3) * (a + 7) * 17 + seed * 11) % 113 / 16. - 56 / 16. for a in range(k)]
                value = ((node * 37 + seed * 13) % 101 - 50) / 64.
                out.append(PositionEval(value, list(range(k)), [str(i) for i in range(k)],
                                        logits, [False] * k, list(range(k))))
            return out

    with (
        patch('imba_chess.eval.gumbel_search.cozy_bridge.board_to_cozy', return_value=(1, 0)),
        patch('imba_chess.eval.gumbel_search._root_hash_seed', return_value=[]),
        patch('imba_chess.eval.gumbel_search.cozy_bridge.terminal_value_native', return_value=None),
        patch('imba_chess.eval.gumbel_search.cc.push_and_classify', side_effect=push),
    ):
        return select_gumbel(evaluator=Evaluator(), board=None, noise=noise,
                             config=GumbelConfig(simulations=budget, top_m=top_m,
                                                 max_depth=depth, value_scale=scale))


@pytest.mark.parametrize('case', json.loads(FIXTURE.read_text())['cases'])
def test_multidepth_matches_mctx(case):
    assert json.loads(FIXTURE.read_text())['revision'] == REFERENCE_REVISION
    actual = run_synthetic(**case['inputs'])
    expected = case['expected']
    assert actual.move_id == expected['action']
    assert actual.visits == expected['visits']
    assert actual.qvalues == pytest.approx(expected['qvalues'], abs=1e-10)
    assert actual.policy == pytest.approx(expected['policy'], abs=1e-10)


def test_sparse_q_range_differs_from_upstream_with_masked_action_padding():
    # Upstream includes unvisited masked actions' mixed values in its Q extrema.
    # Production uses only legal moves: do not silently assume these are equal.
    from imba_chess.eval.gumbel_search import completed_q, softmax
    cfg = GumbelConfig(value_scale=1.)
    compact = completed_q(-.5, [0., 0.], [1, 1], [.1, .1001], cfg)
    padded = completed_q(-.5, [0., 0., -10000.], [1, 1, 0], [.1, .1001, 0.], cfg)
    assert compact == pytest.approx([0., 51.])
    # Values obtained by directly executing pinned mctx qtransforms.py.
    assert padded == pytest.approx([50.97450849716762, 51., 0.])
    assert softmax(compact)[1] > .999999
    assert softmax(padded[:2])[1] == pytest.approx(.5063725306304215)
