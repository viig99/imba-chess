"""CUDA composition checks using real full histories and ordinary search."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.config import load_config
from imba_chess.self_play.seeds import load_seeds
from imba_chess.self_play.benchmarks import histories
from scripts.audit_policy_regression import MixedRuntime


def consume(gen,executors):
    response=None
    while True:
        try:r=gen.send(response)
        except StopIteration as stop:return stop.value
        response=executors[r.kind]([r.payload])[0]


def main():
    torch.set_num_threads(4)
    root=Path('artifacts/eval/search-improvement-2026-09-20')
    cfg=load_config(root/'config.toml')
    cfg=replace(cfg,search=replace(cfg.search,simulations=512))
    base,_=load_runtime(cfg,root/'ckpt34.pt','cuda')
    other,_=load_runtime(cfg,root/'ckpt34.pt','cuda')
    actor,_=load_runtime(cfg,root/'actor54.pt','cuda')
    actor_copy,_=load_runtime(cfg,root/'actor54.pt','cuda')
    seeds=load_seeds(root/'monitor-seeds.json','monitor')
    results=[]
    for _,board,history in histories(seeds[:1],base):
        for ply in range(2):
            kw=dict(board=board,history=history,actor_id='verify',game_id=f'g{ply}',config=cfg.search,noise=0.)
            ordinary=consume(base.search(**kw),base.executors)
            mixed=MixedRuntime(base,other)
            composed=consume(mixed.search(**kw),mixed.executors)
            assert asdict(ordinary)==asdict(composed),'ckpt34 same-source search mismatch'
            actor_ordinary=consume(actor.search(**kw),actor.executors)
            actor_mixed=MixedRuntime(actor,actor_copy)
            actor_composed=consume(actor_mixed.search(**kw),actor_mixed.executors)
            assert asdict(actor_ordinary)==asdict(actor_composed),'actor54 same-source search mismatch'
            for policy,value in ((base,actor),(actor,base)):
                runtime=MixedRuntime(policy,value)
                result=consume(runtime.search(**kw),runtime.executors)
                assert result.simulations==512
                results.append(dict(turn=board.turn,simulations=result.simulations,move=result.move_uci))
            history.append_observed_position(board)
            history.record_played_move(ordinary.move_uci)
            board.push_uci(ordinary.move_uci)
    print(json.dumps(dict(same_source_equivalence=['ckpt34','actor54'],both_colors=True,checks=results)))

if __name__=='__main__':main()
