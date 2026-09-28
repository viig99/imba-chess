"""Frozen-model diagnostics; never used for promotion or optimizer updates.

match: paired zero-noise search versus legal policy argmax.
target: complete-game stored-prior targets scored by unrestricted Stockfish.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import shutil
import time

import chess
import chess.engine
import torch

from imba_chess.config import load_repo_config
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.self_play_store import SelfPlayStore, atomic_json, validate_game
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.gumbel_search import GumbelResult
from imba_chess.eval.position_evaluator import _project_legal_logits
from imba_chess.self_play.collector import play_game, terminal_outcome
from imba_chess.self_play.config import load_config
from imba_chess.self_play.dataset import policy_weights
from imba_chess.self_play.evaluation import paired_interval
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import file_hash, load_seeds, stable_hash
from scripts.audit_search_scales import metrics, score_move

REVISION = 1


def frozen(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        tmp = destination.with_suffix(destination.suffix + '.tmp')
        shutil.copyfile(source, tmp)
        tmp.replace(destination)
    if file_hash(source) != file_hash(destination):
        raise ValueError(f'frozen input changed: {source}')
    return destination


def resume(path, identity):
    state = json.loads(path.read_text()) if path.exists() else dict(identity=identity, results={}, seconds=0)
    if state['identity'] != identity:
        raise ValueError('incompatible audit resume')
    atomic_json(path, state)
    return state


def player_id(checkpoint_hash, algorithm, budget):
    return f'{checkpoint_hash}:{algorithm}:n{budget}'


class GreedyRuntime:
    algorithm = 'greedy'

    def __init__(self, runtime):
        self.move_vocab, self.encoder = runtime.move_vocab, runtime.encoder
        self.executors = {'root_eval': runtime.executors['root_eval']}
        self.options = dict(algorithm='legal_policy_argmax', dtype='float32', tf32=False)

    def search(self, *, board, history, actor_id, game_id, **kwargs):
        owner = (actor_id, game_id)
        batch = history.build_batch_for_current_position(board)
        batch['game_id'] = [game_id]
        identity, output = yield WorkRequest('root_eval', (owner, batch))
        if identity != owner:
            raise ValueError('greedy inference owner mismatch')
        logits, moves, total, mapped = _project_legal_logits(
            logits=output['logits'][-1], board=board, move_vocab=self.move_vocab)
        if total != mapped or not moves:
            raise ValueError('incomplete legal vocabulary')
        logs = torch.log_softmax(logits.float(), 0).tolist()
        chosen = max(range(len(moves)), key=logs.__getitem__)
        ids = [self.move_vocab.encode(m.uci()) for m in moves]
        # Value output is deliberately not read, including for selection.
        return GumbelResult(moves[chosen].uci(), ids[chosen], ids,
                            [float(i == chosen) for i in range(len(ids))],
                            0., None, [0]*len(ids), [0.]*len(ids),
                            0, 1, 0, 0, 0, logs)


def routed_executors(players):
    def executor(kind):
        def run(payloads):
            groups = defaultdict(list)
            for i, payload in enumerate(payloads):
                groups[payload[0][0]].append((i, payload))
            output = [None]*len(payloads)
            for actor, entries in groups.items():
                values = players[actor].executors[kind]([p for _, p in entries])
                if len(values) != len(entries):
                    raise ValueError('inference result count mismatch')
                for (i, _), value in zip(entries, values):
                    output[i] = value
            return output
        return run
    return {kind: executor(kind) for p in players.values() for kind in p.executors}


def match_summary(results, pairs):
    completed = [r for r in results.values() if r['status'] == 'completed']
    scores = [(r['outcome_white']*(1 if r['candidate_white'] else -1)+1)/2 for r in completed]
    out = dict(planned=2*pairs, completed=len(completed), wins=scores.count(1),
               draws=scores.count(.5), losses=scores.count(0),
               score=sum(scores)/len(scores) if scores else None,
               unlabeled={k:r.get('termination') for k,r in results.items() if r['status'] != 'completed'})
    complete_pairs = [i for i in range(pairs) if sum(r['pair']==i for r in completed)==2]
    remap = {pair:i for i,pair in enumerate(complete_pairs)}
    out['completed_pairs'] = len(complete_pairs)
    if complete_pairs:
        out['paired_interval'] = paired_interval(
            [dict(r, pair=remap[r['pair']]) for r in completed if r['pair'] in remap],
            pairs=len(complete_pairs), seed=42)
    return out


def run_match(args, cfg, seeds):
    runtime, maximum = load_runtime(cfg, args.checkpoint, args.device)
    greedy = GreedyRuntime(runtime)
    checkpoint_hash = file_hash(args.checkpoint)
    for budget in args.budgets:
        config = replace(cfg, search=replace(cfg.search, simulations=budget, value_scale=.1, top_m=16, max_depth=32))
        search_id = player_id(checkpoint_hash, 'gumbel-zero', budget)
        greedy_id = player_id(checkpoint_hash, 'greedy', 0)
        identity = dict(revision=REVISION, checkpoint=checkpoint_hash, config=config.identifier,
                        seeds=[asdict(s) for s in seeds[:args.pairs]], players=[search_id, greedy_id],
                        search=asdict(config.search), noise=0, precision='float32', tf32=False, seed=42)
        path = args.output/f'match-n{budget}.json'
        state = resume(path, identity)
        start = time.monotonic()
        previous = state['seconds']
        def save():
            state['seconds'] = previous + time.monotonic()-start
            state['summary'] = match_summary(state['results'], args.pairs)
            atomic_json(path, state)
        def factory():
            for pair, seed in enumerate(seeds[:args.pairs]):
                for white in (True, False):
                    key = f'{pair}:{int(white)}'
                    old = state['results'].get(key, {})
                    if old.get('status') == 'completed' or old.get('termination') in ('context_limit','game_limit'):
                        continue
                    def actors(turn, white=white):
                        return (search_id, runtime) if turn == white else (greedy_id, greedy)
                    yield key, play_game(seed=seed, game_id=stable_hash(json.dumps(identity,sort_keys=True)+key),
                        actor_id=search_id, runtime=runtime, search_config=config.search,
                        max_positions=maximum, max_game_plies=config.collection.max_game_plies,
                        run_seed=42, config_id=config.identifier, actor_for_turn=actors, gumbel_noise=False)
        def done(key, game):
            pair, white = key.split(':')
            state['results'][key] = dict(game, pair=int(pair), candidate_white=bool(int(white)))
            save()
            print(f'n={budget} {key} {game["status"]} {game.get("outcome_white")}', flush=True)
        def error(key, exc):
            done(key, dict(status='unfinished',outcome_white=None,termination='error',error=repr(exc)))
        try:
            BatchScheduler(game_factory=iter(factory()), executors=routed_executors({search_id:runtime,greedy_id:greedy}),
                           concurrent_games=cfg.collection.concurrent_games, completion_order=True,
                           on_game_done=done, on_game_error=error).run()
        finally:
            save()
            runtime.clear_caches()
        print(json.dumps(state['summary']), flush=True)


def validate_trajectory(game, vocab):
    validate_game(game)
    board = chess.Board()
    for move in game['prefix_moves']:
        board.push_uci(move)
    boards = []
    for move, target in zip(game['moves'],game['targets']):
        moves = [vocab.decode(i) for i in target['legal_ids']]
        if set(moves) != {m.uci() for m in board.legal_moves}:
            raise ValueError('legal-move alignment mismatch')
        if target['move_uci'] != move or target['move_id'] != vocab.encode(move):
            raise ValueError('played-move alignment mismatch')
        boards.append(board.copy(stack=True))
        board.push_uci(move)
    if terminal_outcome(board) != (game['outcome_white'],game['termination']):
        raise ValueError('terminal outcome mismatch')
    return boards


def sample_game(game, learning, count=10):
    weights = policy_weights(game['targets'], learning=learning)
    eligible = [i for i,v in enumerate(weights['policy_surprise_eligible']) if v]
    if len(eligible)<count:
        raise ValueError('too few eligible positions')
    rng = random.Random(f'42:{game["game_id"]}')
    selected = [rng.choice(eligible[j*len(eligible)//count:(j+1)*len(eligible)//count]) for j in range(count)]
    return [(i,weights['policy_training_weight'][i]) for i in selected]


def qualifies(game, learning):
    if game.get('status') != 'completed':
        return False
    search = game.get('search_config',{})
    if search.get('simulations') != 200 or search.get('value_scale') != .1:
        return False
    validate_game(game)
    return sum(policy_weights(game['targets'],learning=learning)['policy_surprise_eligible']) >= 10


def target_summary(rows):
    if not rows:
        return dict(games=0,positions=0)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['game_id']].append(row)
    def aggregate(chosen):
        gains = [r['metrics']['target_gain'] for r in chosen]
        mass = sum(r['weight'] for r in chosen)
        cp = [r for r in chosen if r['metrics']['target_cp_gain'] is not None]
        return dict(target_gain=sum(gains)/len(gains),
                    weighted_target_gain=sum(r['weight']*r['metrics']['target_gain'] for r in chosen)/mass,
                    played_minus_actor_greedy=sum(r['metrics']['selected_gain'] for r in chosen)/len(chosen),
                    positive=sum(g>1e-12 for g in gains)/len(gains),
                    negative=sum(g< -1e-12 for g in gains)/len(gains),
                    unchanged=sum(abs(g)<=1e-12 for g in gains)/len(gains),
                    cp_positions=len(cp),
                    target_cp_gain=sum(r['metrics']['target_cp_gain'] for r in cp)/len(cp) if cp else None)
    out = aggregate(rows)
    out.update(games=len(grouped),positions=len(rows),per_game={g:aggregate(rs) for g,rs in grouped.items()})
    rng = random.Random(42)
    keys = list(grouped)
    draws = defaultdict(list)
    for _ in range(2000):
        sample = [r for g in rng.choices(keys,k=len(keys)) for r in grouped[g]]
        values = aggregate(sample)
        for key in ('target_gain','weighted_target_gain','played_minus_actor_greedy'):
            draws[key].append(values[key])
    out['game_bootstrap_95ci'] = {k:[sorted(v)[50],sorted(v)[1950]] for k,v in draws.items()}
    out['harmful_positions'] = sorted([r for r in rows if r['metrics']['target_gain']< -1e-12],
                                       key=lambda r:r['metrics']['target_gain'])[:10]
    out['interpretation'] = 'Small-sample engine-based proxies, not Elo or measured win-rate gains. Played moves include training exploration.'
    return out


def acquire_games(args, cfg, seeds, vocab):
    selection_path = args.output/'selection.json'
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())
        return [json.loads((args.output/'games'/f'{gid}.json').read_text()) for gid in selected['game_ids']]
    games = []
    if args.replay:
        store = SelfPlayStore(args.replay,read_only=True)
        ids = sorted(store.game_ids(None))
        random.Random(42).shuffle(ids)
        for gid in ids:
            game = store.read_game(gid)
            if qualifies(game,cfg.learning):
                validate_trajectory(game,vocab)
                games.append(game)
                if len(games)==10:
                    break
        provenance = dict(kind='read_only_replay',path=str(args.replay),manifest_hash=file_hash(args.replay/'manifest.json'))
    else:
        provenance = dict(kind='frozen_actor54_fallback',checkpoint=file_hash(args.checkpoint),
                          remote_failure=args.remote_failure,noise='training Gumbel',attempt_limit=20)
        runtime, maximum = load_runtime(cfg,args.checkpoint,args.device)
        actor = player_id(file_hash(args.checkpoint),'gumbel-training',200)
        # Waves launch only as many games as still needed; preserve opening order
        # when selecting, independently of scheduler completion order.
        index = 0
        while index < 20 and len(games) < 10:
            wave = list(range(index, min(20, index + 10-len(games))))
            def factory():
                for attempt in wave:
                    path = args.output/'attempts'/f'{attempt}.json'
                    if path.exists():
                        continue
                    seed = seeds[25+attempt]
                    gid = stable_hash(f'42:target:{actor}:{seed.seed_id}')
                    yield str(attempt), play_game(seed=seed,game_id=gid,actor_id=actor,
                        runtime=runtime,search_config=cfg.search,max_positions=maximum,
                        max_game_plies=cfg.collection.max_game_plies,run_seed=42,config_id=cfg.identifier)
            def done(key, game):
                atomic_json(args.output/'attempts'/f'{key}.json',game)
                print(f'target attempt {int(key)+1}: {game["status"]}',flush=True)
            BatchScheduler(game_factory=iter(factory()),executors=runtime.executors,
                concurrent_games=min(cfg.collection.concurrent_games,len(wave)),
                on_game_done=done,
                on_game_error=lambda key,e:done(key,dict(status='unfinished',termination='error',error=repr(e))),
                completion_order=True).run()
            for attempt in wave:
                game = json.loads((args.output/'attempts'/f'{attempt}.json').read_text())
                if qualifies(game,cfg.learning):
                    validate_trajectory(game,vocab)
                    games.append(game)
            index += len(wave)
            print(f'target games: {len(games)}/10, attempts {index}/20',flush=True)
        runtime.clear_caches()
    for game in games:
        atomic_json(args.output/'games'/f'{game["game_id"]}.json',game)
    atomic_json(selection_path,dict(provenance=provenance,game_ids=[g['game_id'] for g in games],
                actors=[g['actor_id'] for g in games],shortfall=10-len(games),
                hashes={g['game_id']:file_hash(args.output/'games'/f'{g["game_id"]}.json') for g in games}))
    return games


def run_target(args,cfg,seeds):
    repo=load_repo_config(Path(cfg.base_config))
    vocab=MoveVocab.load(repo.vocab.path)
    identity=dict(revision=REVISION,config=cfg.identifier,checkpoint=file_hash(args.checkpoint),
                  replay=str(args.replay) if args.replay else None,remote_failure=args.remote_failure,
                  openings=[asdict(s) for s in seeds[25:45]],engine=file_hash(args.stockfish),
                  nodes=100000,threads=1,hash_mib=64,seed=42,positions_per_game=10)
    state=resume(args.output/'target.json',identity)
    start=time.monotonic()
    games=acquire_games(args,cfg,seeds,vocab)
    selection=json.loads((args.output/'selection.json').read_text())
    for gid,digest in selection['hashes'].items():
        if file_hash(args.output/'games'/f'{gid}.json')!=digest:
            raise ValueError('frozen replay game changed')
    positions=[]
    for game in games:
        boards=validate_trajectory(game,vocab)
        for i,weight in sample_game(game,cfg.learning):
            positions.append((game,i,weight,boards[i]))
    atomic_json(args.output/'sample.json',[dict(game_id=g['game_id'],index=i,weight=w,fen=b.fen()) for g,i,w,b in positions])
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({'Threads':1,'Hash':64,'UCI_LimitStrength':False,'UCI_ShowWDL':True})
        state['engine']=engine.id
        for game,i,weight,board in positions:
            key=f'{game["game_id"]}:{i}'
            if key in state['results']:
                continue
            target=game['targets'][i]
            moves=[vocab.decode(mid) for mid in target['legal_ids']]
            cache_path=args.output/'engine_cache'/f'{stable_hash(key)}.json'
            scores=json.loads(cache_path.read_text()) if cache_path.exists() else {}
            for move in moves:
                if move not in scores:
                    scores[move]=score_move(engine,board,chess.Move.from_uci(move),100000)
                    atomic_json(cache_path,scores)
            # Remove allowed stored FP32 target normalization roundoff.
            normalized=dict(target,policy=[p/math.fsum(target['policy']) for p in target['policy']])
            state['results'][key]=dict(game_id=game['game_id'],actor_id=game['actor_id'],index=i,
                fen=board.fen(),history=[m.uci() for m in board.move_stack],moves=moves,target=target,
                weight=weight,scores=scores,metrics=metrics(normalized,moves,scores))
            state['summary']=target_summary(list(state['results'].values()))
            atomic_json(args.output/'target.json',state)
            print(f'Stockfish positions {len(state["results"])}/{len(positions)}',flush=True)
            if args.stop_after_positions and len(state['results'])>=args.stop_after_positions:
                break
    state['seconds']+=time.monotonic()-start
    state['summary']=target_summary(list(state['results'].values()))
    state['summary']['shortfall_games']=10-len(games)
    state['summary']['planned_positions']=len(positions)
    atomic_json(args.output/'target.json',state)
    atomic_json(args.output/'target-summary.json',state['summary'])


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1] in ('policy-match','policy-positions','mixed-search','transfer'):
        from scripts.audit_policy_regression import main as regression_main
        return regression_main()
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('match','target'):
        p=sub.add_parser(command)
        p.add_argument('--config',type=Path,required=True)
        p.add_argument('--checkpoint',type=Path,required=True)
        p.add_argument('--seeds',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--device',default='cuda')
        if command=='match':
            p.add_argument('--pairs',type=int,default=25)
            p.add_argument('--budgets',type=int,nargs='+',default=[32,200],choices=[32,200])
        else:
            p.add_argument('--replay',type=Path)
            p.add_argument('--remote-failure',default='')
            p.add_argument('--stockfish',type=Path,default=Path('/usr/bin/stockfish'))
            p.add_argument('--stop-after-positions',type=int,default=0,help='Smoke test; resume with zero to finish')
    args=parser.parse_args()
    torch.set_num_threads(4)
    with run_lock(args.output):
        args.config=frozen(args.config,args.output/'inputs/config.toml')
        args.checkpoint=frozen(args.checkpoint,args.output/'inputs/checkpoint.pt')
        args.seeds=frozen(args.seeds,args.output/'inputs/monitor-seeds.json')
        cfg=load_config(args.config)
        cfg=replace(cfg,run=replace(cfg.run,seed=42),search=replace(cfg.search,simulations=200,value_scale=.1,top_m=16,max_depth=32))
        frozen(cfg.base_config,args.output/'inputs/base-config.toml')
        seeds=load_seeds(args.seeds,'monitor')
        if len({s.source_id for s in seeds})!=len(seeds):
            raise ValueError('monitor openings must have distinct source games')
        if args.command=='match':
            if args.pairs<1 or len(seeds)<args.pairs:
                raise ValueError('not enough openings')
            run_match(args,cfg,seeds)
        else:
            if not cfg.learning.policy_surprise_enabled:
                raise ValueError('target audit requires the generating run surprise configuration')
            if not args.replay and (not args.remote_failure or len(seeds)<45):
                raise ValueError('fallback requires remote failure provenance and 20 openings after match openings')
            run_target(args,cfg,seeds)


if __name__=='__main__':
    main()
