"""Fixed-position scale screen; Stockfish scores are independent of neural search.

All legal moves receive an equal Stockfish node budget with a fresh hash table.
Scores are from the original side to move. WDL expectation is an engine proxy,
not an empirical model win rate. No training or remote state is modified.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import statistics
import time

import chess
import chess.engine
import torch
from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval import cozy_bridge
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.self_play.benchmarks import histories
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import load_seeds, file_hash


def score_move(engine, board, move, nodes):
    engine.configure({'Clear Hash': None})
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[move])
    score = info['score'].pov(board.turn)
    wdl = info['wdl'].pov(board.turn)
    return dict(cp=score.score(), mate=score.mate(), expectation=wdl.expectation(),
                wdl=list(wdl), depth=info.get('depth'), nodes=info.get('nodes'))


def metrics(result, moves, scores):
    logs = result['root_log_priors']
    maximum = max(logs)
    z = math.fsum(math.exp(x-maximum) for x in logs)
    normalized_logs = [(x-maximum)-math.log(z) for x in logs]
    prior = [math.exp(x) for x in normalized_logs]
    target = result['policy']
    values = [scores[m]['expectation'] for m in moves]
    best = max(values)
    prior_move = moves[max(range(len(prior)), key=prior.__getitem__)]
    selected = result['move_uci']
    # CP means are only comparable on positions with no mate-scored legal moves.
    cp_ok = all(scores[m]['cp'] is not None for m in moves)
    return dict(
        kl=math.fsum(p*(math.log(p)-lp) for p,lp in zip(target,normalized_logs) if p>0),
        entropy=-math.fsum(p*math.log(p) for p in target if p>0),
        target_gain=math.fsum((p-q)*v for p,q,v in zip(target,prior,values)),
        target_regret=best-math.fsum(p*v for p,v in zip(target,values)),
        selected_gain=scores[selected]['expectation']-scores[prior_move]['expectation'],
        selected_regret=best-scores[selected]['expectation'],
        changed_move=float(selected!=prior_move),
        target_cp_gain=math.fsum((p-q)*scores[m]['cp'] for p,q,m in zip(target,prior,moves)) if cp_ok else None,
        selected_cp_gain=scores[selected]['cp']-scores[prior_move]['cp'] if cp_ok else None,
    )


def summarize(rows):
    summary={}
    for scale in sorted({r['scale'] for r in rows}):
        chosen=[r for r in rows if r['scale']==scale]
        grouped={}
        for r in chosen: grouped.setdefault(r['seed_id'],[]).append(r['metrics'])
        position_means=[{k:statistics.mean(m[k] for m in ms if m[k] is not None)
                         for k in ms[0] if all(m[k] is not None for m in ms)} for ms in grouped.values()]
        out={k:statistics.mean(m[k] for m in position_means if k in m)
             for k in chosen[0]['metrics'] if any(k in m for m in position_means)}
        out['positions']=len(grouped)
        out['cp_positions']=sum('target_cp_gain' in m for m in position_means)
        for key in ('target_gain','selected_gain'):
            vals=[m[key] for m in position_means]; rng=random.Random(42)
            means=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(2000))
            out[key+'_95ci']=[means[50],means[1950]]
        summary[str(scale)]=out
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seeds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--positions',type=int,default=20)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--scales',default='0,0.003,0.01,0.03,0.1,0.3,1')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--stockfish',default='/usr/bin/stockfish')
    p.add_argument('--nodes-per-move',type=int,default=100000)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if min(args.positions,args.repeats,args.nodes_per_move)<1 or args.offset<0: p.error('invalid bounds')
    cfg=load_config(args.config)
    seeds=load_seeds(args.seeds,'monitor')[args.offset:args.offset+args.positions]
    if len(seeds)!=args.positions: p.error('not enough distinct monitor positions')
    scales=[float(x) for x in args.scales.split(',')]
    if any(not math.isfinite(s) or s<0 for s in scales) or len(set(scales))!=len(scales):
        p.error('scales must be distinct, finite and nonnegative')
    actor=file_hash(args.checkpoint)
    identity=dict(checkpoint=actor,config=cfg.identifier,seeds=[s.seed_id for s in seeds],
                  scales=scales,repeats=args.repeats,nodes=args.nodes_per_move,
                  stockfish=file_hash(args.stockfish),simulations=200,noise='matched training Gumbel',device=args.device)
    with run_lock(args.output):
        path=args.output/'results.json'
        data=json.loads(path.read_text()) if path.exists() else dict(identity=identity,stockfish={},search=[],timings={})
        if data['identity']!=identity: raise ValueError('different experiment identity')
        def save(): atomic_json(path,data)
        start=time.monotonic()
        with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
            engine.configure({'Threads':1,'Hash':64,'UCI_LimitStrength':False,'UCI_ShowWDL':True})
            data['engine']=engine.id
            for seed in seeds:
                if seed.seed_id in data['stockfish']: continue
                board=seed.board()
                scores={m.uci():score_move(engine,board,m,args.nodes_per_move) for m in sorted(board.legal_moves,key=lambda m:m.uci())}
                data['stockfish'][seed.seed_id]=dict(fen=board.fen(),prefix=seed.prefix_moves,scores=scores)
                save(); print('Stockfish',len(data['stockfish']),len(seeds),flush=True)
        data['timings']['stockfish_this_invocation']=time.monotonic()-start
        torch.set_num_threads(4)
        runtime,max_positions=load_runtime(cfg,args.checkpoint,args.device)
        done={(r['seed_id'],r['scale'],r['repeat']) for r in data['search']}
        def factory():
            for scale in scales:
                for repeat in range(args.repeats):
                    for seed,board,history in histories(seeds,runtime):
                        if (seed.seed_id,scale,repeat) in done: continue
                        if len(history.seq_token_id)+1+cfg.search.max_depth>max_positions: raise ValueError('context limit')
                        ids,_,ucis,_,_=cozy_bridge.project_legal_moves(cozy_bridge.board_to_cozy(board),runtime.move_vocab)
                        key=json.dumps([seed.seed_id,scale,repeat])
                        tasks[key]=(seed,scale,repeat,dict(zip(ids,ucis)))
                        yield key,runtime.search(board=board,history=history,actor_id=actor,game_id=key,
                            config=replace(cfg.search,simulations=200,value_scale=scale),
                            rng=random.Random(f'42:{seed.seed_id}:{repeat}'))
        def complete(key,result):
            seed,scale,repeat,mapping=tasks.pop(key)
            raw=asdict(result); moves=[mapping[i] for i in raw['legal_ids']]
            row=dict(seed_id=seed.seed_id,scale=scale,repeat=repeat,result=raw,moves=moves,
                     metrics=metrics(raw,moves,data['stockfish'][seed.seed_id]['scores']))
            data['search'].append(row); save()
            print('Search',len(data['search']),len(seeds)*len(scales)*args.repeats,flush=True)
        def error(key,exc): raise RuntimeError(key) from exc
        tasks={}; start=time.monotonic()
        BatchScheduler(game_factory=iter(factory()),executors=runtime.executors,concurrent_games=8,
                       on_game_done=complete,on_game_error=error,completion_order=True).run()
        data['timings']['search_this_invocation']=time.monotonic()-start
        data['summary']=summarize(data['search']); save()
        atomic_json(args.output/'summary.json',data['summary'])
        print(json.dumps(data['summary'],indent=2),flush=True)


if __name__=='__main__': main()
