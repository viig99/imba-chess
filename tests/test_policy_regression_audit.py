import math
from types import SimpleNamespace
import pytest
import torch
from scripts import audit_policy_regression as a
from imba_chess.eval.batch_scheduler import WorkRequest
from imba_chess.eval.search import PositionEval


def test_composition_alignment_and_value_perspective():
    for v in (-.8,.8):
        p=PositionEval(.2,[],['a','b'],[-1.,-2.],[False,False],[1,2])
        value=p._replace(value_stm=v,legal_log_priors=[-2.,-1.])
        composed=a.compose_nodes([p],[value])[0]
        assert composed.value_stm==v and composed.legal_log_priors==p.legal_log_priors
        with pytest.raises(ValueError,match='alignment'):
            a.compose_nodes([p],[value._replace(legal_ids=[2,1])])


def test_mixed_root_preserves_cache_ownership():
    from dataclasses import dataclass
    @dataclass
    class Result:
        value:float
    caches=[object(),object()];observed=[]
    def runtime(i):
        def search(**kw):
            owner=('a','g')
            _,out=yield WorkRequest('root_eval',(owner,{}))
            assert out['kv_caches'] is caches[i]
            observed.append((out['logits'].item(),out['value_logits'].item()))
            return Result(1.)
        return SimpleNamespace(search=search,executors={},move_vocab=None,encoder=None,algorithm='gumbel')
    gen=a.MixedRuntime(runtime(0),runtime(1)).search(noise=0.)
    assert next(gen).kind=='policy_root_eval'
    assert gen.send((('a','g'),dict(logits=torch.tensor(3),value_logits=torch.tensor(4),kv_caches=caches[0]))).kind=='value_root_eval'
    with pytest.raises(StopIteration):
        gen.send((('a','g'),dict(logits=torch.tensor(8),value_logits=torch.tensor(9),kv_caches=caches[1])))
    assert observed==[(3,9),(3,9)]


def test_quality_and_kl_hand_calculation():
    s={'a':dict(expectation=.2,cp=-100),'b':dict(expectation=.8,cp=100)}
    q=a.quality([math.log(.8),math.log(.2)],[math.log(.2),math.log(.8)],['a','b'],s,[.25,.75])
    assert q['distribution']==pytest.approx(.36)
    assert q['greedy']==pytest.approx(.6)
    assert q['distribution_cp']==pytest.approx(120)
    assert q['kl_change']==pytest.approx(-.5*math.log(4))
    s['b']['cp']=None
    assert a.quality([0,0],[0,1],['a','b'],s)['distribution_cp'] is None


def test_joint_bootstrap_preserves_paired_arm_covariance():
    arms={a:{f'{i}:{w}':dict(pair=i,candidate_white=w,status='completed',outcome_white=outcome*(1 if w else -1))
             for i,outcome in enumerate([-1,1,-1,1]) for w in (True,False)} for a in ('00','10','01','11')}
    result=a.contrasts(arms)
    assert result['complete_joint_pairs']==4
    assert all(v==[0,0] for v in result['paired_bootstrap_95ci'].values())
    arms['11']['0:True']['status']='unfinished'
    assert a.contrasts(arms)['complete_joint_pairs']==3


def test_game_cluster_bootstrap():
    rows=[dict(game_id=g,quality=dict(distribution=v,greedy=v,distribution_cp=None,greedy_cp=None))
          for g,v in [('a',0)]*10+[('b',1)]*10]
    s=a.position_summary(rows)
    assert s['means']['distribution']==.5
    assert s['game_bootstrap_95ci']['distribution']==[0,1]


def test_historical_provenance_rejects_substitutes():
    state=dict(config_id='run',reuse_counts={'g':1})
    game=dict(game_id='g',actor_id='earlier',config_id='run',split='train')
    for changed in [dict(actor_id='actor54'),dict(config_id='wrong'),dict(game_id='unconsumed'),dict(split='monitor')]:
        assert not a.historical_qualifies(dict(game,**changed),state,'earlier','actor54','run')
    assert not a.historical_qualifies(game,state,'earlier','earlier','run')


def test_exact_512_budget():
    from test_gumbel_mctx_audit import run_synthetic
    result=run_synthetic(k=20,seed=42,budget=512,top_m=16,depth=32,terminal=False,scale=.1,noise=[0.]*20)
    assert result.simulations==512 and sum(result.visits)==512


def test_new_match_restart_preserves_games(tmp_path,monkeypatch):
    from imba_chess.self_play.config import SelfPlayConfig
    from imba_chess.self_play.seeds import Seed,source_split
    from test_self_play import ScriptRuntime
    class Runtime(ScriptRuntime):
        inference_rows={}
        waves={}
        def clear_caches(self):pass
        def search(self,**kw):
            gen=super().search(**kw);next(gen)
            yield WorkRequest('tick',((kw['actor_id'],kw['game_id']),None))
            try:gen.send(None)
            except StopIteration as stop:return stop.value
    r=Runtime();r.executors={'tick':lambda ps:[None]*len(ps)}
    monkeypatch.setattr(a,'load_runtime',lambda *args:(r,128))
    monkeypatch.setattr(a,'GreedyRuntime',lambda r:r)
    source='source'
    while source_split(source)!='monitor':source+='x'
    seed=Seed('s',source,[],0,'monitor','corpus')
    args=SimpleNamespace(baseline='base',checkpoint='actor',device='cuda',command='policy-match',output=tmp_path,stop_after_pairs=0)
    ident=dict(baseline='basehash',actor='actorhash')
    a.run_games(args,SelfPlayConfig(),[seed],ident)
    import json
    before=json.loads((tmp_path/'greedy.json').read_text())
    assert before['summary']['completed']==2
    def forbidden(**kw):raise AssertionError('completed game replayed')
    r.search=forbidden
    a.run_games(args,SelfPlayConfig(),[seed],ident)
    after=json.loads((tmp_path/'greedy.json').read_text())
    assert before['results']==after['results']
