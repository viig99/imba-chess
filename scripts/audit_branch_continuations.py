"""Frozen diagnostic rollouts from audited move pairs; never writes training replay."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import chess
import chess.engine
import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.self_play.collector import play_game, terminal_outcome
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import Seed, file_hash, source_split
from scripts.audit_search_scales import score_move

ARMS = ('bad', 'good', 'bad_forced_reply')


def board_for(prefix):
    board = chess.Board()
    for uci in prefix:
        board.push_uci(uci)
    return board


def select_positions(screen, audit, count, seed):
    by_source = defaultdict(list)
    for rec in sorted(audit['records'], key=lambda r: r['position_id']):
        if rec['cause'] == 'value_misorder':
            pos = screen['stockfish'][rec['position_id']]
            by_source[pos['source']].append((pos, rec))
    rng = random.Random(seed)
    sources = sorted(by_source)
    if count > len(sources):
        raise ValueError('not enough distinct source games')
    selected = []
    for source in rng.sample(sources, count):
        pos, rec = rng.choice(by_source[source])
        board = board_for(pos['prefix'])
        if board.fen() != pos['fen']:
            raise ValueError('history does not match cached position')
        selected.append(dict(position_id=pos['position_id'], source=source,
                             prefix=pos['prefix'], fen=pos['fen'], root_white=board.turn,
                             bad=rec['chosen'], good=rec['best_good_move'],
                             stockfish_bad=pos['scores'][rec['chosen']],
                             stockfish_good=pos['scores'][rec['best_good_move']], audit=rec))
    return selected


def root_score(game, root_white):
    if game['status'] != 'completed':
        return None
    result = game['outcome_white']
    if result not in (-1, 0, 1):
        raise ValueError('completed game lacks a valid outcome')
    return (1 + (result if root_white else -result)) / 2


def summarize(positions, games, repeats, seed):
    indexed = defaultdict(list)
    for game in games:
        indexed[(game['position_id'], game['arm'])].append(game)
    rows = []
    for pos in positions:
        row = dict(position_id=pos['position_id'], source=pos['source'])
        for arm in ARMS:
            gs = indexed[(pos['position_id'], arm)]
            scores = [root_score(g, pos['root_white']) for g in gs]
            scores = [s for s in scores if s is not None]
            row[arm] = dict(attempted=len(gs), completed=len(scores),
                            wins=scores.count(1), draws=scores.count(.5), losses=scores.count(0),
                            score=statistics.mean(scores) if scores else None)
        natural = indexed[(pos['position_id'], 'bad')]
        first = [g for g in natural if g.get('first_reply') is not None]
        row['first_reply_good'] = sum(g['first_reply_good'] for g in first)
        row['first_reply_observed'] = len(first)
        rows.append(row)
    contrasts = {}
    for lhs, rhs in [('good', 'bad'), ('bad_forced_reply', 'bad')]:
        # One position per source game: bootstrap source games, never individual branches.
        diffs = [r[lhs]['score'] - r[rhs]['score'] for r in rows
                 if r[lhs]['completed'] == repeats and r[rhs]['completed'] == repeats]
        if diffs:
            rng = random.Random(seed)
            draws = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
            contrasts[f'{lhs}_minus_{rhs}'] = dict(positions=len(diffs), mean=statistics.mean(diffs),
                source_game_bootstrap_95ci=[draws[249], draws[9749]],
                positive=sum(x > 0 for x in diffs), tied=diffs.count(0), negative=sum(x < 0 for x in diffs))
    arms = {}
    for arm in ARMS:
        rs = [r[arm] for r in rows]
        counts = {k: sum(r[k] for r in rs) for k in ('attempted', 'completed', 'wins', 'draws', 'losses')}
        counts['score'] = (counts['wins'] + .5 * counts['draws']) / counts['completed'] if counts['completed'] else None
        arms[arm] = counts
    return dict(arms=arms, contrasts=contrasts, positions=rows,
                first_reply_good=sum(r['first_reply_good'] for r in rows),
                first_reply_observed=sum(r['first_reply_observed'] for r in rows),
                terminations=dict(Counter(g['termination'] for g in games)),
                incomplete=sum(g['status'] != 'completed' for g in games))


def source_identity():
    paths = sorted(Path('src/imba_chess').rglob('*.py')) + [Path(__file__), Path('scripts/audit_search_scales.py')]
    return {str(p): file_hash(p) for p in paths}


def prepare(args, cfg):
    screen_path = args.screen / 'results.json'
    screen = json.loads(screen_path.read_text())
    audit = json.loads(args.blunders.read_text())
    if audit['checkpoint'] != screen['identity']['checkpoint']:
        raise ValueError('audit and screen checkpoint mismatch')
    positions = select_positions(screen, audit, args.positions, args.seed)
    identity = dict(checkpoint=file_hash(args.checkpoint), config=file_hash(args.config),
                    base_config=file_hash(cfg.base_config), screen=file_hash(screen_path),
                    blunders=file_hash(args.blunders), search=asdict(cfg.search),
                    seed=args.seed, repeats=args.repeats, positions=positions,
                    engine=file_hash(args.stockfish), nodes_per_reply=args.nodes_per_reply,
                    concurrent=args.concurrent, max_game_plies=512, dtype='float32', tf32=False,
                    gumbel_noise=True, sources=source_identity(), device=args.device)
    path = args.output / 'manifest.json'
    if path.exists():
        data = json.loads(path.read_text())
        if data['identity'] != identity:
            raise ValueError('incompatible diagnostic resume')
    else:
        data = dict(identity=identity, reply_labels={})
        atomic_json(path, data)
    pending = [p for p in positions if p['position_id'] not in data['reply_labels']]
    if pending:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        try:
            engine.configure({'Threads': 1, 'Hash': 64, 'UCI_LimitStrength': False, 'UCI_ShowWDL': True})
            data['engine_identity'] = engine.id
            for pos in pending:
                board = board_for(pos['prefix'] + [pos['bad']])
                if terminal_outcome(board) is not None:
                    raise ValueError('selected bad move is already terminal')
                labels = {m.uci(): score_move(engine, board, m, args.nodes_per_reply)
                          for m in sorted(board.legal_moves, key=lambda m: m.uci())}
                best = max(s['expectation'] for s in labels.values())
                # Equal node budgets per reply; deterministic UCI tie-break, then CP/mate.
                def rank(uci):
                    s = labels[uci]
                    mate = s['mate']
                    cp = s['cp'] if mate is None else (100000 - mate if mate > 0 else -100000 - mate)
                    return (s['expectation'], cp)
                chosen = max(sorted(labels), key=rank)
                data['reply_labels'][pos['position_id']] = dict(scores=labels, best_reply=chosen,
                    good_replies=sorted(m for m, s in labels.items() if s['expectation'] >= best - .02))
                atomic_json(path, data)
                print(json.dumps(dict(phase='reply_labels', completed=len(data['reply_labels']), total=len(positions))), flush=True)
        finally:
            engine.quit()
    return data


def rollout_task(pos, arm, repeat, manifest, runtime, cfg, max_positions, args):
    pid = pos['position_id']
    forced = [pos['good'] if arm == 'good' else pos['bad']]
    if arm == 'bad_forced_reply':
        forced.append(manifest['reply_labels'][pid]['best_reply'])
    prefix = pos['prefix'] + forced
    board = board_for(prefix)
    gid = f'{pid}-{arm}-{repeat}'
    terminal = terminal_outcome(board)
    if terminal is not None:
        game = dict(game_id=gid, prefix_moves=prefix, moves=[], targets=[], status='completed',
                    outcome_white=terminal[0], termination=terminal[1])
    else:
        seed = Seed(gid, pos['source'], prefix, len(prefix), source_split(pos['source']), 'branch-diagnostic')
        game = yield from play_game(seed=seed, game_id=gid, actor_id=manifest['identity']['checkpoint'],
            runtime=runtime, search_config=cfg.search, max_positions=max_positions,
            max_game_plies=512, run_seed=args.seed, config_id=manifest['identity']['config'], gumbel_noise=True)
    game.update(position_id=pid, source=pos['source'], arm=arm, repeat=repeat,
                root_white=pos['root_white'], forced_moves=forced,
                diagnostic_only=True, first_reply=None, first_reply_good=None)
    if arm == 'bad' and game['moves']:
        game['first_reply'] = game['moves'][0]
        game['first_reply_good'] = game['first_reply'] in manifest['reply_labels'][pid]['good_replies']
    return game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--blunders', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--concurrent', type=int, default=12)
    parser.add_argument('--nodes-per-reply', type=int, default=100000)
    parser.add_argument('--stockfish', default='/usr/bin/stockfish')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-games', type=int, default=None, help='Smoke-test cap; resume runs remaining games')
    args = parser.parse_args()
    if min(args.positions, args.repeats, args.concurrent, args.nodes_per_reply) < 1:
        parser.error('counts must be positive')
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    with run_lock(args.output):
        manifest = prepare(args, cfg)
        if args.prepare_only:
            return
        positions = manifest['identity']['positions']
        directory = args.output / 'games'
        directory.mkdir(exist_ok=True)
        expected = {f"{p['position_id']}-{arm}-{r}" for p in positions for r in range(args.repeats) for arm in ARMS}
        games = {}
        for path in directory.glob('*.json'):
            game = json.loads(path.read_text())
            if path.stem not in expected or game['game_id'] != path.stem:
                raise ValueError('unexpected saved game')
            games[path.stem] = game
        pending = [(p, a, r) for p in positions for r in range(args.repeats) for a in ARMS
                   if f"{p['position_id']}-{a}-{r}" not in games]
        if args.max_games is not None:
            pending = pending[:args.max_games]
        started = time.monotonic()
        def save():
            result = summarize(positions, list(games.values()), args.repeats, args.seed)
            result.update(expected_games=len(expected), saved_games=len(games),
                          phase='complete' if len(games) == len(expected) else 'partial',
                          current_process_seconds=time.monotonic() - started)
            atomic_json(args.output / 'summary.json', result)
            return result
        if not pending:
            print(json.dumps(save(), indent=2), flush=True)
            return
        runtime, max_positions = load_runtime(cfg, args.checkpoint, args.device)
        atomic_json(args.output / 'runtime.json', dict(options=runtime.options, max_positions=max_positions,
                    torch_version=torch.__version__, device=torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'))
        def factory():
            for p, a, r in pending:
                gid = f"{p['position_id']}-{a}-{r}"
                yield gid, rollout_task(p, a, r, manifest, runtime, cfg, max_positions, args)
        def done(gid, game):
            if game is None:
                raise RuntimeError(f'no result for {gid}')
            for target in game['targets']:
                if target['simulations'] != cfg.search.simulations:
                    raise ValueError('simulation budget mismatch')
            atomic_json(directory / f'{gid}.json', game)
            games[gid] = game
            result = save()
            print(json.dumps(dict(phase='rollouts', saved=len(games), total=len(expected), game=gid,
                 outcome_white=game['outcome_white'], termination=game['termination'],
                 plies=len(game['moves']), seconds=round(result['current_process_seconds'], 1))), flush=True)
        def error(gid, exc):
            raise RuntimeError(gid) from exc
        save()
        BatchScheduler(game_factory=iter(factory()), executors=runtime.executors,
                       concurrent_games=args.concurrent, on_game_done=done, on_game_error=error,
                       completion_order=True).run()
        result = save()
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
