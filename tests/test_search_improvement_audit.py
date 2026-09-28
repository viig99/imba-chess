import copy
import json
import math
from types import SimpleNamespace

import chess
import chess.engine
import pytest
import torch

from scripts import audit_search_improvement as audit
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.dataset import policy_weights
from test_gumbel_mctx_audit import run_synthetic
from test_self_play import mate_game

VOCAB=MoveVocab.load('artifacts/move_vocab_static_uci.json')


def test_greedy_legal_argmax_ignores_value_and_illegal_logits():
    runtime=SimpleNamespace(move_vocab=VOCAB,encoder=BoardStateEncoder(),executors={'root_eval':None})
    greedy=audit.GreedyRuntime(runtime)
    board=chess.Board()
    history=_SequenceHistory(move_vocab=VOCAB,board_state_encoder=runtime.encoder)
    gen=greedy.search(board=board,history=history,actor_id='greedy',game_id='g')
    next(gen)
    logits=torch.zeros(1,len(VOCAB.token_to_id))
    logits[0,VOCAB.encode('e2e4')]=10
    logits[0,VOCAB.encode('e7e5')]=100
    with pytest.raises(StopIteration) as stopped:
        gen.send((('greedy','g'),dict(logits=logits)))  # No value output at all.
    assert stopped.value.value.move_uci=='e2e4'
    assert stopped.value.value.simulations==0


def test_distinct_players_route_same_checkpoint():
    ids=[audit.player_id('same','greedy',0),audit.player_id('same','gumbel-zero',32),audit.player_id('same','gumbel-zero',200)]
    assert len(set(ids))==3
    players={actor:SimpleNamespace(executors={'root_eval':lambda ps,a=actor:[(p[0],a) for p in ps]}) for actor in ids}
    payloads=[((actor,str(i)),{}) for i,actor in enumerate(reversed(ids))]
    actual=audit.routed_executors(players)['root_eval'](payloads)
    assert [v for _,v in actual]==list(reversed(ids))


@pytest.mark.parametrize('budget',[32,200])
def test_exact_search_budget(budget):
    result=run_synthetic(k=20,seed=42,budget=budget,top_m=16,depth=32,terminal=False,scale=.1,noise=[0.]*20)
    assert result.simulations==budget
    assert sum(result.visits)==budget


def test_resume_rejects_identity_changes_and_preserves_completed(tmp_path):
    path=tmp_path/'state.json'
    state=audit.resume(path,{'budget':32})
    state['results']['g']={'status':'completed','outcome_white':1}
    audit.atomic_json(path,state)
    assert audit.resume(path,{'budget':32})==state
    with pytest.raises(ValueError,match='incompatible'):
        audit.resume(path,{'budget':200})


def test_pair_scores_both_colors_and_unlabeled_limits():
    rows={'a':dict(pair=0,candidate_white=True,status='completed',outcome_white=1),
          'b':dict(pair=0,candidate_white=False,status='completed',outcome_white=-1),
          'c':dict(pair=1,candidate_white=False,status='unfinished',outcome_white=None,termination='game_limit')}
    result=audit.match_summary(rows,2)
    assert result['wins']==2 and result['draws']==0 and result['completed']==2
    assert result['paired_interval']['score']==1
    assert result['unlabeled']=={'c':'game_limit'}


@pytest.mark.parametrize('white',[True,False])
def test_stockfish_original_side_perspective_fresh_hash(white):
    class Engine:
        cleared=False
        def configure(self,options):
            assert options=={'Clear Hash':None}
            self.cleared=True
        def analyse(self,board,limit,root_moves):
            assert self.cleared and limit.nodes==100000 and len(root_moves)==1
            return dict(score=chess.engine.PovScore(chess.engine.Cp(120),chess.WHITE),
                        wdl=chess.engine.PovWdl(chess.engine.Wdl(700,200,100),chess.WHITE))
    board=chess.Board()
    if not white: board.push_uci('e2e4')
    value=audit.score_move(Engine(),board,next(iter(board.legal_moves)),100000)
    assert value['cp']==(120 if white else -120)
    assert value['expectation']==pytest.approx(.8 if white else .2)


def test_gain_alignment_stable_normalization_and_mates():
    scores={'a':dict(expectation=.2,cp=-100,mate=None),'b':dict(expectation=.8,cp=100,mate=None)}
    target=dict(policy=[.75,.25],root_log_priors=[1000+math.log(.25),1000+math.log(.75)],move_uci='b')
    value=audit.metrics(target,['b','a'],scores)
    assert value['target_gain']==pytest.approx(.3)
    assert value['target_cp_gain']==pytest.approx(100)
    scores['b'].update(cp=None,mate=2)
    assert audit.metrics(target,['b','a'],scores)['target_cp_gain'] is None


def test_full_game_weights_precede_segment_sampling():
    learning=LearningConfig(policy_surprise_enabled=True)
    game=dict(game_id='g',targets=[dict(policy=[1.,0.],root_log_priors=[-i-1.,0.]) for i in range(101)])
    samples=audit.sample_game(game,learning)
    expected=policy_weights(game['targets'],learning=learning)['policy_training_weight']
    assert len(samples)==10
    for j,(index,weight) in enumerate(samples):
        assert j*101//10<=index<(j+1)*101//10
        assert weight==expected[index]
    assert audit.sample_game(game,learning)==samples


def test_clustered_bootstrap_uses_games_and_weighted_ratio():
    rows=[dict(game_id=gid,weight=w,metrics=dict(target_gain=g,selected_gain=0,target_cp_gain=None))
          for gid,w,g in [('a',1,0)]*10+[('b',3,1)]*10]
    result=audit.target_summary(rows)
    assert result['target_gain']==.5
    assert result['weighted_target_gain']==.75
    assert result['game_bootstrap_95ci']['target_gain']==[0,1]
    assert result['games']==2 and result['cp_positions']==0


def test_complete_history_terminal_and_move_alignment():
    game=mate_game()
    boards=audit.validate_trajectory(game,VOCAB)
    assert len(boards[-1].move_stack)==3
    bad=copy.deepcopy(game)
    bad['outcome_white']=1
    with pytest.raises(ValueError,match='terminal'):
        audit.validate_trajectory(bad,VOCAB)
    bad=copy.deepcopy(game)
    bad['targets'][0]['legal_ids'][0]=VOCAB.encode('e7e5')
    with pytest.raises(ValueError):
        audit.validate_trajectory(bad,VOCAB)


def test_match_restart_skips_completed_games(tmp_path,monkeypatch):
    from imba_chess.self_play.config import SelfPlayConfig
    from imba_chess.self_play.seeds import Seed,source_split
    from test_self_play import ScriptRuntime
    source='source'
    while source_split(source)!='monitor': source+='x'
    seed=Seed('s',source,[],0,'monitor','c')
    class RoutedScript(ScriptRuntime):
        def search(self, **kwargs):
            gen=super().search(**kwargs)
            next(gen)
            from imba_chess.eval.batch_scheduler import WorkRequest
            yield WorkRequest('tick',((kwargs['actor_id'],kwargs['game_id']),None))
            try:
                gen.send(None)
            except StopIteration as stopped:
                return stopped.value
    runtime=RoutedScript()
    runtime.executors={'root_eval':lambda ps:[], 'tick':lambda ps:[None]*len(ps)}
    runtime.clear_caches=lambda:None
    monkeypatch.setattr(audit,'load_runtime',lambda *args:(runtime,128))
    # Exercise collector, persistence, and scheduler with scripted mate play.
    monkeypatch.setattr(audit,'GreedyRuntime',lambda r:r)
    checkpoint=tmp_path/'ckpt'; checkpoint.write_text('frozen')
    args=SimpleNamespace(checkpoint=checkpoint,device='cuda',budgets=[32],output=tmp_path,pairs=1)
    audit.run_match(args,SelfPlayConfig(),[seed])
    first=json.loads((tmp_path/'match-n32.json').read_text())
    assert first['summary']['completed']==2
    def forbidden(**kwargs): raise AssertionError('replayed completed game')
    runtime.search=forbidden
    audit.run_match(args,SelfPlayConfig(),[seed])
    second=json.loads((tmp_path/'match-n32.json').read_text())
    assert second['results']==first['results']


def test_fallback_stops_at_ten_and_resume_preserves_frozen_games(tmp_path,monkeypatch):
    from imba_chess.self_play.config import SelfPlayConfig
    runtime=SimpleNamespace(executors={},clear_caches=lambda:None)
    monkeypatch.setattr(audit,'load_runtime',lambda *args:(runtime,1024))
    monkeypatch.setattr(audit,'qualifies',lambda g,l:g['status']=='completed')
    monkeypatch.setattr(audit,'validate_trajectory',lambda *args:None)
    calls=[]
    def play(**kwargs):
        i=int(kwargs['seed'].seed_id)
        calls.append(i)
        if False: yield
        return dict(game_id=str(i),actor_id='actor54',status='unfinished' if i<27 else 'completed')
    monkeypatch.setattr(audit,'play_game',play)
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_text('actor54')
    args=SimpleNamespace(output=tmp_path,checkpoint=checkpoint,device='cuda',replay=None,remote_failure='refused')
    seeds=[SimpleNamespace(seed_id=str(i)) for i in range(45)]
    games=audit.acquire_games(args,SelfPlayConfig(),seeds,None)
    assert len(games)==10 and calls==list(range(25,37))
    assert audit.acquire_games(args,SelfPlayConfig(),seeds,None)==games
    assert calls==list(range(25,37))
