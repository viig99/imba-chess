"""Observation must preserve the real search and WDL perspectives."""
import math
from pathlib import Path
import random
import runpy

import chess
import pytest

from imba_chess.eval import cozy_bridge, gumbel_search, inference_runtime
from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.eval.search import _drive
from tests.test_gumbel_search import FakeEvaluator

STUDY = runpy.run_path('scripts/measure_search_wdl.py')


class WdlEvaluator(FakeEvaluator):
    def evaluate(self, batch):
        rows = super().evaluate(batch)
        output = []
        for row, (handle, _) in zip(rows, batch):
            # Include asymmetric WDL and a different value at each depth/action.
            code = sum(ord(x) for x in str(handle)) % 13
            win = .1 + .04 * code
            loss = .8 - win
            wdl = (loss, .2, win)
            output.append(row._replace(value_stm=win-loss, wdl=wdl))
        return output


def measured_search(evaluator, board, config, root=None, noise=None):
    if root is None:
        root = evaluator.evaluate([(None,cozy_bridge.board_to_cozy(board))])[0]
    with STUDY['capture_leaf_wdl']() as captured:
        result = _drive(inference_runtime.gumbel_stepwise(board=board, extend=evaluator.extend,
            config=config, root_eval=root, root_wdl=root.wdl, noise=noise, rng=random.Random(42)), evaluator)
        observation = captured.pop(id(result))
    return result, observation


@pytest.mark.parametrize('forcing,floor,weight', [(False,False,0), (True,False,.5), (True,True,.5), (True,True,1)])
def test_observer_preserves_search_and_counts_every_leaf(forcing, floor, weight):
    board = chess.Board('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3')
    cfg = GumbelConfig(simulations=64, top_m=4, max_depth=5, root_forcing=forcing,
                       forcing_floor=floor, minimax_weight=weight)
    plain_eval = WdlEvaluator()
    root = plain_eval.evaluate([(None,cozy_bridge.board_to_cozy(board))])[0]
    expected = gumbel_search.select_gumbel(evaluator=plain_eval, board=board, root_eval=root,
        root_wdl=root.wdl, rng=random.Random(42), config=cfg)
    evaluator = WdlEvaluator()
    actual, observation = measured_search(evaluator, board, cfg)
    assert actual == expected
    assert evaluator.calls == plain_eval.calls
    assert observation['labels']['A'] == actual.search_wdl
    assert observation['visits'] == actual.visits
    assert observation['simulations'] == 64
    assert observation['odd_depth_simulations'] > 0
    if not forcing:
        assert observation['forcing_added_share'] == 0
    assert inference_runtime.gumbel_stepwise is gumbel_search.gumbel_stepwise


@pytest.mark.parametrize('move,expected', [('f7g7',(0,0,1)), ('f7e6',(0,1,0))])
def test_terminal_leaf_flips_from_opponent_loss_to_root_win(move, expected):
    board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    evaluator = WdlEvaluator()
    root = evaluator.evaluate([(None,cozy_bridge.board_to_cozy(board))])[0]
    noise = [100 if uci==move else 0 for uci in root.legal_ucis]
    result, observed = measured_search(evaluator,board,GumbelConfig(simulations=16,top_m=1),root,noise)
    assert result.move_uci == move
    assert observed['terminal_simulations'] == 16
    assert observed['labels']['A'] == observed['labels']['B'] == observed['labels']['C'] == expected
    assert observed['outside_chosen_share'] == 0


def test_hand_computed_B_C_and_exploration_shares():
    # Two first-round visits go to opposite root edges. Final selection chooses
    # edge1. All-simulation mean = (.35,.2,.45); chosen mean=(.1,.2,.7).
    cfg = GumbelConfig(simulations=2,top_m=1,root_forcing=True)
    observer = STUDY['LeafObserver']({'config':cfg})
    observer.samples=2; observer.counts=[1,1,0]
    observer.total=[.7,.4,.9]
    observer.edge_sums=[[.6,.2,.2],[.1,.2,.7],[0,0,0]]
    observer.metadata=dict(forcing_only=[1], root_candidates=2)
    result = type('Result',(),dict(simulations=2,visits=[1,1,0],search_wdl=(.35,.2,.45),
        move_id=11,legal_ids=[10,11,12],move_uci='b2b3',policy=[.2,.3,.5],root_wdl=(.2,.3,.5)))()
    row=observer.finish(result)
    assert row['labels']['B'] == pytest.approx((.1,.2,.7))
    assert row['labels']['C'] == pytest.approx((.3,.2,.5))
    assert row['improved_policy_visited_mass'] == .5
    assert row['outside_chosen_share'] == row['forcing_added_share'] == .5
    assert row['forcing_added_candidates'] == 1


def test_forcing_only_candidates_exclude_noise_prior_top_m():
    board=chess.Board('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3')
    evaluator=WdlEvaluator(); root=evaluator.evaluate([(None,cozy_bridge.board_to_cozy(board))])[0]
    noise=[0]*len(root.legal_ids)
    capture=root.legal_ucis.index('f3e5')
    noise[capture]=-100
    _, row=measured_search(evaluator,board,GumbelConfig(simulations=32,top_m=1,max_depth=2,root_forcing=True),root,noise)
    assert capture in row['forcing_only'] and capture not in row['base_top_m']
    assert row['forcing_added_share'] > 0
    assert row['root_candidates'] > 1


def test_sampler_preserves_exact_prefix_and_ply(monkeypatch):
    from types import SimpleNamespace
    from imba_chess.self_play import seeds
    b=chess.Board(); prefix=[]; rng=random.Random(42)
    for _ in range(96):
        moves=list(b.legal_moves); rng.shuffle(moves)
        for move in moves:
            test=b.copy(); test.push(move)
            if not test.is_game_over(claim_draw=True):
                b=test; prefix.append(move.uci()); break
        else:
            pytest.fail('fixture ran out of playable moves')
    monkeypatch.setattr(seeds,'load_seeds',lambda *args,**kwargs:[SimpleNamespace(
        source_id='source',seed_id='seed',prefix_moves=prefix)])
    rows=STUDY['position_sample'](Path('artifacts/corpus/v4_self_play_seeds_4096.json'),per_phase=2)
    assert rows and len({r['position_id'] for r in rows}) == len(rows)
    for row in rows:
        assert row['ply'] == len(row['prefix_moves'])
        b=chess.Board()
        for move in row['prefix_moves']: b.push_uci(move)
        assert b.fen() == row['fen']
