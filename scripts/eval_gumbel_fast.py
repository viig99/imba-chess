"""Gumbel evaluation using the optimized self-play inference runtime."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import random
import statistics
import time
import traceback

import chess
import chess.engine
import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.gumbel_search import GumbelConfig, GumbelResult
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.self_play import collector
from imba_chess.self_play.config import SelfPlayConfig
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import Seed, source_split, stable_hash

ORIGINAL_GUMBEL = collector.gumbel_stepwise


def no_noise_gumbel(**kwargs):
    kwargs['noise'] = [0.0] * len(kwargs['root_eval'].legal_ids)
    return ORIGINAL_GUMBEL(**kwargs)


def status(out, phase, **extra):
    value = dict(phase=phase, timestamp=time.time(), **extra)
    atomic_json(out/'status.json', value)
    print(json.dumps(value), flush=True)


def runtime_for(model_runtime, mode):
    return collector.InferenceRuntime(
        model=model_runtime.model, move_vocab=model_runtime.move_vocab,
        encoder=model_runtime.encoder, device=model_runtime.device,
        root_batch_tokens=1024, one_query_per_game=True, cache_prefixes=True,
        batch_projection=True, batch_inputs=True, batch_suffix=True, decoder_mode=mode,
    )


def probe(runtime, positions, concurrency, simulations):
    results = {}
    config = GumbelConfig(simulations=simulations, top_m=16, max_depth=32)
    def jobs():
        for i, seed in enumerate(positions[:concurrency]):
            board = chess.Board()
            history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
            for move in seed['prefix_moves']:
                history.append_observed_position(board)
                history.record_played_move(move)
                board.push_uci(move)
            yield str(i), runtime.search(board=board, history=history, actor_id='actor111',
                                        game_id=str(i), config=config, rng=random.Random(42), should_stop=lambda: False)
    def fail(key, error):
        raise RuntimeError(f'Probe {key} failed') from error
    runtime.clear_caches()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        BatchScheduler(game_factory=iter(jobs()), executors=runtime.executors,
                       concurrent_games=concurrency, completion_order=True,
                       on_game_done=lambda key, result: results.__setitem__(key, result),
                       on_game_error=fail).run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    assert len(results) == concurrency and all(r.simulations == simulations for r in results.values())
    measurement = dict(concurrency=concurrency, seconds=elapsed, moves_per_second=concurrency/elapsed,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    runtime.clear_caches()
    return measurement, results


class StockfishPool:
    """One UCI engine per live game, with bounded parallel engine work."""
    def __init__(self, runtime, games, elo):
        self.vocab = runtime.move_vocab
        self.available, self.owners = [], {}
        self.executor = ThreadPoolExecutor(max_workers=min(games, 4))
        self.all_engines = []
        try:
            for _ in range(games):
                engine = chess.engine.SimpleEngine.popen_uci('/usr/bin/stockfish')
                self.all_engines.append(engine)
                engine.configure({'Threads':1,'Hash':64,'UCI_LimitStrength':True,'UCI_Elo':elo})
                self.available.append(engine)
        except BaseException:
            self.close()
            raise

    def execute(self, payloads):
        work = []
        for owner, board in payloads:
            if owner not in self.owners:
                if not self.available:
                    raise RuntimeError('Stockfish engine pool exhausted')
                self.owners[owner] = self.available.pop()
            work.append((owner, board, self.owners[owner]))
        def play(item):
            owner, board, engine = item
            move = engine.play(board, chess.engine.Limit(nodes=40000, time=5.0)).move
            if move is None: raise RuntimeError('Stockfish returned no move')
            return owner, move
        return list(self.executor.map(play, work))

    def search(self, *, board, actor_id, game_id, **kwargs):
        owner = (actor_id, game_id)
        identity, move = yield WorkRequest('stockfish', (owner, board.copy(stack=True)))
        assert identity == owner
        ids = [self.vocab.encode(m.uci()) for m in board.legal_moves]
        return GumbelResult(move.uci(), self.vocab.encode(move.uci()), ids, [1/len(ids)]*len(ids),
                            0.0, None, [0]*len(ids), [0.0]*len(ids), 0, 0, 0, 0, 0)

    def release(self, game_id):
        engine = self.owners.pop(('stockfish', game_id), None)
        if engine is not None: self.available.append(engine)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for engine in self.all_engines:
            engine.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--elo', type=int, default=2400)
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--label', default='actor111')
    parser.add_argument('--concurrency', type=int, choices=(8,16,24,32,48))
    parser.add_argument('--require-compiled', action='store_true')
    args = parser.parse_args()
    if args.simulations < 1 or args.games < 2 or args.games % 2:
        parser.error('Use positive simulations and a positive even game count')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    collector.gumbel_stepwise = no_noise_gumbel
    cfg = SelfPlayConfig(base_config=str(args.config), search=GumbelConfig(simulations=args.simulations, top_m=16, max_depth=32))
    status(out, 'loading', simulations=args.simulations)
    runtime, max_positions = load_runtime(cfg, args.checkpoint, 'cuda', decoder_mode='current')
    assert not runtime.model.training
    positions = sorted([s for s in json.loads(args.seeds.read_text())['seeds'] if s['split']=='monitor'],
                       key=lambda s: (-len(s['prefix_moves']),s['seed_id']))[:48]
    assert len(positions)==48
    atomic_json(out/'probe_positions.json', positions)
    status(out, 'checking_compiled_decoder')
    eager_measure, eager_results = probe(runtime, positions, 8, args.simulations)
    # Warm eager before obtaining its comparable timing.
    eager_measure, eager_results = probe(runtime, positions, 8, args.simulations)
    compiled = runtime_for(runtime, 'compiled')
    mode = 'compiled'
    try:
        cold, compiled_results = probe(compiled, positions, 8, args.simulations)
        compiled_measure, compiled_results = probe(compiled, positions, 8, args.simulations)
        errors=[]
        for key in eager_results:
            a,b=eager_results[key],compiled_results[key]
            assert a.legal_ids==b.legal_ids
            error=max(abs(x-y) for x,y in zip(a.policy,b.policy))
            errors.append(error)
            if a.move_uci!=b.move_uci or error>2e-4:
                raise RuntimeError(f'Compiled search differs from eager: move {a.move_uci}/{b.move_uci}, policy max error {error}')
        atomic_json(out/'compile_check.json',dict(passed=True, max_policy_error=max(errors),
                    eager=eager_measure, compiled=compiled_measure, cold=cold))
        runtime.clear_caches()
        runtime=compiled
    except Exception as error:
        atomic_json(out/'compile_check.json',dict(passed=False,error=str(error),traceback=traceback.format_exc()))
        if args.require_compiled:
            raise
        compiled.clear_caches()
        del compiled
        gc.collect(); torch.cuda.empty_cache()
        mode='current'
    rows=[]
    parameter_bytes=sum(p.numel()*p.element_size() for p in runtime.model.parameters())
    for concurrency in ((args.concurrency,) if args.concurrency else (8,16,24,32,48)):
        status(out,'concurrency_probe',concurrency=concurrency,decoder_mode=mode)
        try:
            probe(runtime,positions,concurrency,args.simulations)  # shape warmup
            trials=[probe(runtime,positions,concurrency,args.simulations)[0] for _ in range(2)]
            peak=max(t['peak_allocated_bytes'] for t in trials)
            # Conservative projection for long-game prefixes, plus headroom.
            # This is a capacity estimate, not a promise to allocate every byte.
            shortest=min(len(s['prefix_moves'])+2 for s in positions[:concurrency])
            projected=parameter_bytes+max(0,peak-parameter_bytes)*max_positions/shortest
            row=dict(concurrency=concurrency,decoder_mode=mode,trials=trials,
                     moves_per_second=statistics.median(t['moves_per_second'] for t in trials),
                     peak_allocated_bytes=peak,projected_long_context_bytes=projected,
                     eligible=projected<=7*1024**3 and max(t['peak_reserved_bytes'] for t in trials)<=7*1024**3)
            rows.append(row)
            atomic_json(out/'concurrency.json',rows)
            if not row['eligible']: break
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(concurrency=concurrency,eligible=False,error='CUDA out of memory during capacity probe'))
            runtime.clear_caches(); gc.collect(); torch.cuda.empty_cache()
            atomic_json(out/'concurrency.json',rows)
            break
    eligible=[r for r in rows if r['eligible']]
    if not eligible: raise RuntimeError('No concurrency passed the VRAM headroom check')
    selected=max(eligible,key=lambda r:r['moves_per_second'])
    concurrency=selected['concurrency']
    atomic_json(out/'selection.json',dict(selected=selected,simulations=args.simulations,decoder_mode=mode,
                 requested_concurrency=args.concurrency,
                 note='Fixed requested concurrency, capacity checked.' if args.concurrency else
                 'Fastest measured concurrency among candidates with long-context VRAM headroom. Not a latency-matched test.'))
    runtime.clear_caches(); gc.collect(); torch.cuda.empty_cache()
    pool=StockfishPool(runtime,concurrency,args.elo)
    records=[]
    model_seconds=[]
    begun=time.perf_counter()
    def position(result,elapsed):
        if result.simulations: model_seconds.append(elapsed)
    def jobs():
        for i in range(args.games):
            source=f'gumbel128-fast-6042-{i}'
            while source_split(source)!='monitor': source+='x'
            seed=Seed(stable_hash(source),source,[],0,'monitor','initial-board')
            game_id=f'game-{i}'
            def actors(turn,white=(i%2==0)):
                return (args.label,runtime) if turn==white else ('stockfish',pool)
            yield game_id, collector.play_game(seed=seed,game_id=game_id,actor_id=args.label,
                    runtime=runtime,search_config=cfg.search,max_positions=max_positions,
                    max_game_plies=512,run_seed=6042,actor_for_turn=actors,on_position=position)
    def done(key,game):
        pool.release(key)
        i=int(key.split('-')[-1])
        game.pop('targets',None)
        atomic_json(out/f'game-{i:03d}.json',game)
        if game['status']!='completed':
            raise RuntimeError(f'Game {i} incomplete: {game.get("termination")}: {game.get("error")}')
        score=(game['outcome_white']*(1 if i%2==0 else -1)+1)/2
        records.append(dict(game=i,model_color='white' if i%2==0 else 'black',score=score,plies=len(game['moves'])))
        wins=sum(r['score']==1 for r in records); draws=sum(r['score']==.5 for r in records)
        atomic_json(out/'game_records.json',records)
        status(out,'games',games_completed=len(records),games_requested=args.games,wins=wins,draws=draws,
               losses=len(records)-wins-draws,score=sum(r['score'] for r in records)/len(records),
               concurrency=concurrency,decoder_mode=mode,simulations=args.simulations,elo=args.elo,
               elapsed_seconds=time.perf_counter()-begun)
    def fail(key,error): raise RuntimeError(f'Game {key} failed') from error
    status(out,'games',games_completed=0,games_requested=args.games,concurrency=concurrency,
           decoder_mode=mode,simulations=args.simulations,elo=args.elo)
    try:
        with torch.inference_mode():
            BatchScheduler(game_factory=iter(jobs()),executors=dict(runtime.executors,stockfish=pool.execute),
                           concurrent_games=concurrency,completion_order=True,
                           on_game_done=done,on_game_error=fail).run()
    finally:
        runtime.clear_caches()
        pool.close()
    assert len(records)==args.games
    atomic_json(out/'summary.json',dict(label=args.label,checkpoint=str(args.checkpoint.resolve()),
               games=len(records),score=sum(r['score'] for r in records)/len(records),
               wins=sum(r['score']==1 for r in records),draws=sum(r['score']==.5 for r in records),
               losses=sum(r['score']==0 for r in records),concurrency=concurrency,decoder_mode=mode,
               simulations=args.simulations,elo=args.elo,model_moves=len(model_seconds),
               mean_model_move_seconds=statistics.mean(model_seconds),inference_mode=True,
               runtime_options=runtime.options,elapsed_seconds=time.perf_counter()-begun))
    status(out,'complete',games_completed=len(records),score=sum(r['score'] for r in records)/len(records))


if __name__=='__main__':
    main()
