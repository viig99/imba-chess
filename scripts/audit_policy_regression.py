"""Frozen policy/value diagnostics, separate from production evaluation."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import time

import chess
import torch
from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.composed_runtime import ComposedRuntime, compose_nodes
from imba_chess.eval.position_evaluator import _SequenceHistory, _project_legal_logits
from imba_chess.self_play.collector import play_game
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import file_hash, load_seeds, stable_hash
from scripts.audit_search_improvement import frozen, resume, GreedyRuntime, routed_executors, match_summary


class MixedRuntime(ComposedRuntime):
    """Zero-noise policy/value composition with this audit's recorded provenance.

    The composition itself lives in imba_chess.eval.composed_runtime; this
    subclass only pins the options block and the zero-noise requirement that
    the completed 2026-09-20 audit ran under.
    """
    allow_noise = False

    def __init__(self, policy, value):
        super().__init__(policy, value)
        self.options = dict(algorithm='gumbel-mixed', dtype='float32', tf32=False)

    def search(self, **kwargs):
        if kwargs.get('noise') != 0.0:
            raise ValueError('mixed diagnostic requires zero noise')
        return (yield from super().search(**kwargs))


def identity(args, cfg, seeds):
    sources = sorted(list(Path('src/imba_chess').rglob('*.py')) +
                     [Path(__file__), Path('scripts/audit_search_improvement.py'), Path('scripts/audit_search_scales.py')])
    return dict(revision=1, baseline=file_hash(args.baseline), actor=file_hash(args.checkpoint),
                config=cfg.identifier, search=asdict(cfg.search), noise=0, dtype='float32', tf32=False,
                seeds=[asdict(s) for s in seeds], seed=42,
                sources={str(p): file_hash(p) for p in sources})


def role_id(ident, p, v, algorithm):
    return f'policy={ident[p]}:value={ident[v]}:{algorithm}:n{512 if algorithm == "gumbel" else 0}'


def run_games(args, cfg, seeds, ident):
    baseline, maximum = load_runtime(cfg, args.baseline, args.device)
    actor, actor_max = load_runtime(cfg, args.checkpoint, args.device)
    if maximum != actor_max:
        raise ValueError('checkpoint context limits differ')
    greedy = args.command == 'policy-match'
    arms = ['greedy'] if greedy else args.arms
    for arm in arms:
        p, v = ('actor','actor') if greedy else tuple('actor' if x=='1' else 'baseline' for x in arm)
        nets = dict(baseline=baseline, actor=actor)
        candidate = GreedyRuntime(actor) if greedy else (nets[p] if p==v else MixedRuntime(nets[p],nets[v]))
        opponent = GreedyRuntime(baseline) if greedy else baseline
        cid = role_id(ident,p,v,'greedy' if greedy else 'gumbel')
        oid = role_id(ident,'baseline','baseline','greedy' if greedy else 'gumbel')
        path = args.output / f'{arm}.json'
        state = resume(path, dict(ident, arm=arm))
        start, previous = time.monotonic(), state['seconds']
        initial = {name:dict(net.inference_rows) for name,net in nets.items()}
        initial_waves = {name:{k:sum(v.values()) for k,v in net.waves.items()} for name,net in nets.items()}
        old_waves = state.get('executor_waves',{})
        counts = state.get('neural_rows',{})
        def save():
            state['seconds'] = previous + time.monotonic()-start
            state['summary'] = match_summary(state['results'],len(seeds))
            state['neural_rows'] = {name:{kind:counts.get(name,{}).get(kind,0)+n-initial[name].get(kind,0)
                for kind,n in net.inference_rows.items()} for name,net in nets.items()}
            state['executor_waves'] = {name:{k:old_waves.get(name,{}).get(k,0)+sum(v.values())-initial_waves[name].get(k,0) for k,v in net.waves.items()} for name,net in nets.items()}
            atomic_json(path,state)
        def factory():
            for pair,seed in enumerate(seeds[:args.stop_after_pairs or len(seeds)]):
                for white in (True,False):
                    key=f'{pair}:{int(white)}'
                    old=state['results'].get(key,{})
                    if old.get('status')=='completed' or old.get('termination') in ('context_limit','game_limit'):
                        continue
                    def actors(turn,white=white):
                        return (cid,candidate) if turn==white else (oid,opponent)
                    yield key,play_game(seed=seed,game_id=stable_hash(json.dumps(ident,sort_keys=True)+arm+key),
                        actor_id=cid,runtime=candidate,search_config=cfg.search,max_positions=maximum,
                        max_game_plies=cfg.collection.max_game_plies,run_seed=42,config_id=cfg.identifier,
                        actor_for_turn=actors,gumbel_noise=False)
        def done(key,game):
            pair,white=key.split(':')
            if not greedy and any(t['simulations'] != 512 for t in game.get('targets',[]) if len(t['legal_ids'])>1):
                raise ValueError('incorrect simulation budget')
            state['results'][key]=dict(game,pair=int(pair),candidate_white=bool(int(white)))
            save()
            print(arm,key,game['status'],game.get('outcome_white'),game.get('error',''),flush=True)
        try:
            BatchScheduler(game_factory=iter(factory()),executors=routed_executors({cid:candidate,oid:opponent}),
                concurrent_games=cfg.collection.concurrent_games,completion_order=True,on_game_done=done,
                on_game_error=lambda k,e:done(k,dict(status='unfinished',termination='error',error=repr(e)))).run()
        finally:
            save()
            baseline.clear_caches(); actor.clear_caches()
        print(json.dumps(state['summary']),flush=True)
    paths=[args.output/f'{a}.json' for a in ('00','10','01','11')]
    if all(p.exists() for p in paths):
        atomic_json(args.output/'contrasts.json',contrasts({a:json.loads(p.read_text())['results'] for a,p in zip(('00','10','01','11'),paths)}))


def contrasts(arms):
    pairs = sorted(set.intersection(*[set(r['pair'] for r in rows.values() if r['status']=='completed') for rows in arms.values()]))
    pairs=[i for i in pairs if all(sum(r['pair']==i and r['status']=='completed' for r in rows.values())==2 for rows in arms.values())]
    def compute(indices):
        scores={a:sum((r['outcome_white']*(1 if r['candidate_white'] else -1)+1)/2
                     for i in indices for r in rows.values() if r['pair']==i and r['status']=='completed')/(2*len(indices))
                for a,rows in arms.items()}
        return dict(policy=scores['10']-scores['00'],value=scores['01']-scores['00'],both=scores['11']-scores['00'],
                    interaction=scores['11']-scores['10']-scores['01']+scores['00'])
    if not pairs: return dict(complete_joint_pairs=0)
    means=compute(pairs); rng=random.Random(42)
    draws=[compute(rng.choices(pairs,k=len(pairs))) for _ in range(2000)]
    return dict(complete_joint_pairs=len(pairs),effects=means,
                paired_bootstrap_95ci={k:[sorted(d[k] for d in draws)[i] for i in (50,1950)] for k in means})


def quality(logs0, logs1, moves, scores, target=None):
    def normalize(logs):
        z=max(logs)+math.log(math.fsum(math.exp(x-max(logs)) for x in logs))
        return [x-z for x in logs]
    logs0,logs1=normalize(logs0),normalize(logs1)
    p0,p1=([math.exp(x) for x in logs] for logs in (logs0,logs1))
    m0,m1=(moves[max(range(len(moves)),key=p.__getitem__)] for p in (p0,p1))
    out=dict(distribution=sum((b-a)*scores[m]['expectation'] for a,b,m in zip(p0,p1,moves)),
             greedy=scores[m1]['expectation']-scores[m0]['expectation'],baseline_move=m0,actor_move=m1)
    finite=all(scores[m]['cp'] is not None for m in moves)
    out['distribution_cp']=sum((b-a)*scores[m]['cp'] for a,b,m in zip(p0,p1,moves)) if finite else None
    out['greedy_cp']=scores[m1]['cp']-scores[m0]['cp'] if finite else None
    if target is not None:
        target=[x/sum(target) for x in target]
        out['kl_baseline']=sum(t*(math.log(t)-l) for t,l in zip(target,logs0) if t>0)
        out['kl_actor']=sum(t*(math.log(t)-l) for t,l in zip(target,logs1) if t>0)
        out['kl_change']=out['kl_actor']-out['kl_baseline']
    return out


def position_summary(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[r['game_id']].append(r)
    keys=('distribution','greedy','distribution_cp','greedy_cp','kl_change')
    def means(rs):
        return {k:sum(r['quality'][k] for r in rs if r['quality'].get(k) is not None)/sum(r['quality'].get(k) is not None for r in rs)
                for k in keys if any(r['quality'].get(k) is not None for r in rs)}
    out=dict(positions=len(rows),games=len(grouped),means=means(rows),per_game={g:means(rs) for g,rs in grouped.items()})
    if not rows:return out
    rng=random.Random(42)
    draws=[means([r for g in rng.choices(list(grouped),k=len(grouped)) for r in grouped[g]]) for _ in range(2000)]
    out['game_bootstrap_95ci']={k:[sorted(d[k] for d in draws if k in d)[min(int(q*sum(k in d for d in draws)),sum(k in d for d in draws)-1)] for q in (.025,.975)] for k in out['means']}
    out['fractions']={k:dict(improved=sum(r['quality'][k]>1e-12 for r in rows)/len(rows),worsened=sum(r['quality'][k]<-1e-12 for r in rows)/len(rows),unchanged=sum(abs(r['quality'][k])<=1e-12 for r in rows)/len(rows)) for k in ('distribution','greedy')}
    out['cp_positions']=sum(r['quality']['distribution_cp'] is not None for r in rows)
    out['representative_gains']=sorted(rows,key=lambda r:r['quality']['distribution'],reverse=True)[:5]
    out['representative_losses']=sorted(rows,key=lambda r:r['quality']['distribution'])[:5]
    return out


def policy_logs(runtime,row):
    history=_SequenceHistory(move_vocab=runtime.move_vocab,board_state_encoder=runtime.encoder)
    board=chess.Board()
    for uci in row['history']:
        history.append_observed_position(board); history.record_played_move(uci); board.push_uci(uci)
    if board.fen()!=row['fen']:raise ValueError('history/FEN mismatch')
    batch=history.build_batch_for_current_position(board);batch['game_id']=[row['game_id']]
    _,output=runtime.executors['root_eval']([(('position',row['game_id']),batch)])[0]
    logits,moves,total,mapped=_project_legal_logits(logits=output['logits'][-1],board=board,move_vocab=runtime.move_vocab)
    if total!=mapped or set(row['moves'])!={m.uci() for m in moves}:raise ValueError('legal alignment mismatch')
    by_move=dict(zip([m.uci() for m in moves],torch.log_softmax(logits.float(),0).tolist()))
    return [by_move[m] for m in row['moves']]


def run_positions(args,cfg,ident):
    source=json.loads(args.audit.read_text())
    if len(source['results'])!=100 or len({r['game_id'] for r in source['results'].values()})!=10:
        raise ValueError('expected exact 100-position ten-game audit')
    state=resume(args.output/'positions.json',dict(ident,audit=file_hash(args.audit)))
    baseline,_=load_runtime(cfg,args.baseline,args.device);actor,_=load_runtime(cfg,args.checkpoint,args.device)
    start=time.monotonic()
    try:
        for key,row in source['results'].items():
            if key in state['results']:continue
            logs0,logs1=policy_logs(baseline,row),policy_logs(actor,row)
            state['results'][key]=dict(row,baseline_log_priors=logs0,actor_log_priors=logs1,
                quality=quality(logs0,logs1,row['moves'],row['scores']))
            atomic_json(args.output/'positions.json',state)
            print('positions',len(state['results']),flush=True)
            if args.stop_after_positions and len(state['results'])>=args.stop_after_positions:break
    finally:
        state['seconds']+=time.monotonic()-start
        state['summary']=position_summary(list(state['results'].values()))
        state['population']='Held-out monitor continuations generated by actor 54; not representative of all chess positions.'
        atomic_json(args.output/'positions.json',state)
        atomic_json(args.output/'position-summary.json',state['summary'])


def historical_qualifies(game, state, earlier_hash, actor_hash, config_id):
    from scripts.audit_search_improvement import qualifies
    from imba_chess.self_play.config import LearningConfig
    if earlier_hash == actor_hash or game.get('actor_id') != earlier_hash:
        return False
    if game.get('config_id') != config_id or state.get('config_id') != config_id:
        return False
    if game.get('split') != 'train' or state.get('reuse_counts',{}).get(game.get('game_id'),0) <= 0:
        return False
    return qualifies(game,LearningConfig(**state['learning_config']))


def run_transfer(args, ident):
    """Validate ancestry/exposure evidence before admitting any historical data."""
    if args.learner_state is None or args.run_state is None:
        raise ValueError('transfer requires --learner-state and --run-state')
    learner_path=frozen(args.learner_state,args.output/'inputs/learner-state.pt')
    run_path=frozen(args.run_state,args.output/'inputs/run-state.json')
    state=torch.load(learner_path,map_location='cpu',weights_only=False)
    actor=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    run=json.loads(run_path.read_text())
    equal=state['model'].keys()==actor['model'].keys() and all(torch.equal(t,actor['model'][k]) for k,t in state['model'].items())
    if not equal:raise ValueError('learner weights do not match frozen actor54')
    if run['actor_id']!=ident['actor'] or state['config_id']!=run['config_id'] or not state['learning_config']['detach_value_features']:
        raise ValueError('wrong-run learner evidence')
    counts=state['reuse_counts']
    evidence=dict(identity=ident,learner_hash=file_hash(learner_path),run_state_hash=file_hash(run_path),
        weights_match=True,config_id=state['config_id'],progress=state['progress'],
        exposure_counts=counts,positive_exposure_games=sum(n>0 for n in counts.values()),
        replay_shards=state['replay_shards'],remote_attempt=args.remote_evidence,
        status='blocked',missing=['Actual earlier detached-value replay trajectories with generating priors and complete histories',
        'Earlier generating checkpoint with at least ten qualifying consumed games and verified ancestry'])
    atomic_json(args.output/'transfer.json',evidence)
    print(json.dumps({k:evidence[k] for k in ('status','weights_match','positive_exposure_games','missing')}),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['policy-match','policy-positions','mixed-search','transfer'])
    for name in ('config','checkpoint','baseline','seeds','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--pairs',type=int,default=50)
    p.add_argument('--stop-after-pairs',type=int,default=0)
    p.add_argument('--stop-after-positions',type=int,default=0)
    p.add_argument('--arms',nargs='+',choices=['00','10','01','11'],default=['00','10','01','11'])
    p.add_argument('--audit',type=Path)
    p.add_argument('--learner-state',type=Path)
    p.add_argument('--run-state',type=Path)
    p.add_argument('--remote-evidence',default='')
    args=p.parse_args(argv)
    torch.set_num_threads(4)
    with run_lock(args.output):
        for name in ('config','checkpoint','baseline','seeds'):
            original=getattr(args,name)
            setattr(args,name,frozen(original,args.output/'inputs'/f'{name}{original.suffix}'))
        cfg=load_config(args.config)
        cfg=replace(cfg,search=replace(cfg.search,simulations=512,value_scale=.1,top_m=16,max_depth=32))
        frozen(cfg.base_config,args.output/'inputs/base-config.toml')
        seeds=load_seeds(args.seeds,'monitor')[:args.pairs]
        if len(seeds)!=args.pairs or len({s.source_id for s in seeds})!=len(seeds):raise ValueError('distinct opening shortfall')
        ident=identity(args,cfg,seeds)
        for source in ident['sources']:
            source_path=Path(source)
            relative=source_path.resolve().relative_to(Path.cwd())
            frozen(source_path,args.output/'source'/relative)
        existing=args.output/'manifest.json'
        if existing.exists() and json.loads(existing.read_text())!=ident:
            raise ValueError('incompatible frozen manifest')
        atomic_json(existing,ident)
        if args.command in ('policy-match','mixed-search'):run_games(args,cfg,seeds,ident)
        elif args.command=='policy-positions':
            args.audit=frozen(args.audit,args.output/'inputs/audit.json');run_positions(args,cfg,ident)
        else:
            run_transfer(args,ident)

if __name__=='__main__':main()
