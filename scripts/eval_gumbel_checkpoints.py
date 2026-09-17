"""Matched Gumbel checkpoint games with fixed openings, colors and concurrency."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import torch

from eval_gumbel_fast import StockfishPool, no_noise_gumbel, probe, runtime_for, status
from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.self_play import collector
from imba_chess.self_play.config import SelfPlayConfig
from imba_chess.self_play.evaluation import paired_interval
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import Seed


def routed_executors(runtimes):
    def executor(kind):
        def execute(payloads):
            groups = defaultdict(list)
            for i, payload in enumerate(payloads):
                groups[payload[0][0]].append((i, payload))
            output = [None] * len(payloads)
            for actor, entries in groups.items():
                values = runtimes[actor].executors[kind]([p for _, p in entries])
                if len(values) != len(entries):
                    raise RuntimeError('Executor result count mismatch')
                for (i, _), value in zip(entries, values):
                    output[i] = value
            return output
        return execute
    return {kind: executor(kind) for r in runtimes.values() for kind in r.executors}


def play_match(*, out, seeds, candidate_id, opponent_id, runtimes, search,
               max_positions, concurrency, release=lambda key: None):
    candidate = runtimes[candidate_id]
    records = []
    begun = time.perf_counter()
    def jobs():
        for pair, seed in enumerate(seeds):
            for white in (True, False):
                game_id = f'{pair}:{int(white)}'
                def actors(turn, white=white):
                    actor = candidate_id if turn == white else opponent_id
                    return actor, runtimes[actor]
                yield game_id, collector.play_game(
                    seed=seed, game_id=game_id, actor_id=candidate_id, runtime=candidate,
                    search_config=search, max_positions=max_positions, max_game_plies=512,
                    run_seed=7042, actor_for_turn=actors)
    def done(key, game):
        release(key)
        pair, white = map(int, key.split(':'))
        game.pop('targets', None)
        game.update(candidate=candidate_id, opponent=opponent_id, candidate_white=bool(white))
        atomic_json(out/f'game-{pair:03d}-{white}.json', game)
        if game['status'] != 'completed':
            raise RuntimeError(f'Incomplete game {key}: {game.get("termination")}: {game.get("error")}')
        score = (game['outcome_white'] * (1 if white else -1) + 1) / 2
        records.append(dict(pair=pair, candidate_white=bool(white), score=score,
                            status=game['status'], outcome_white=game['outcome_white'],
                            termination=game['termination'], plies=len(game['moves'])))
        atomic_json(out/'game_records.json', records)
        status(out, 'games', games_completed=len(records), games_requested=2*len(seeds),
               wins=sum(r['score'] == 1 for r in records),
               draws=sum(r['score'] == .5 for r in records),
               losses=sum(r['score'] == 0 for r in records),
               score=sum(r['score'] for r in records)/len(records),
               elapsed_seconds=time.perf_counter()-begun, concurrency=concurrency,
               candidate=candidate_id, opponent=opponent_id, simulations=search.simulations)
    def fail(key, error):
        raise RuntimeError(f'Game {key} failed') from error
    status(out, 'games', games_completed=0, games_requested=2*len(seeds),
           concurrency=concurrency, candidate=candidate_id, opponent=opponent_id,
           gameplay_started_timestamp=time.time())
    with torch.inference_mode():
        BatchScheduler(game_factory=iter(jobs()), executors=routed_executors(runtimes),
                       concurrent_games=concurrency, completion_order=True,
                       on_game_done=done, on_game_error=fail).run()
    assert len(records) == 2*len(seeds)
    summary = dict(games=len(records), candidate=candidate_id, opponent=opponent_id,
                   wins=sum(r['score'] == 1 for r in records),
                   draws=sum(r['score'] == .5 for r in records),
                   losses=sum(r['score'] == 0 for r in records),
                   score=sum(r['score'] for r in records)/len(records),
                   paired_interval=paired_interval(records, pairs=len(seeds), seed=7042),
                   elapsed_seconds=time.perf_counter()-begun, concurrency=concurrency,
                   simulations=search.simulations, decoder_mode='compiled', inference_mode=True)
    atomic_json(out/'summary.json', summary)
    status(out, 'complete', **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--leg', choices=['ckpt34_sf2400', 'actor111_sf2400', 'head_to_head'], required=True)
    args = parser.parse_args()
    root = args.experiment
    meta = json.loads((root/'protocol.json').read_text())
    out = root/args.leg/'results'
    out.mkdir(parents=True, exist_ok=False)
    seeds = [Seed(**s) for s in json.loads((root/'openings.json').read_text())]
    assert len(seeds) == 50 and len({s.source_id for s in seeds}) == 50
    for seed in seeds:
        assert seed.split == 'monitor'
        seed.board()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    collector.gumbel_stepwise = no_noise_gumbel
    cfg = SelfPlayConfig(base_config=meta['config'], search=GumbelConfig(simulations=512, top_m=16, max_depth=32))
    candidate = 'ckpt34' if args.leg.startswith('ckpt34') else 'actor111'
    opponent = 'ckpt34' if args.leg == 'head_to_head' else 'stockfish'
    positions = json.loads((root/'probe_positions.json').read_text())
    runtimes, limits = {}, []
    for label in dict.fromkeys([candidate, opponent]):
        if label == 'stockfish':
            continue
        status(out, 'loading', checkpoint=label)
        eager, limit = load_runtime(cfg, meta['models'][label]['snapshot'], 'cuda', decoder_mode='current')
        limits.append(limit)
        assert not eager.model.training
        status(out, 'checking_compiled_decoder', checkpoint=label)
        _, reference = probe(eager, positions, 8, 512)
        compiled = runtime_for(eager, 'compiled')
        measurement, actual = probe(compiled, positions, 8, 512)
        errors = []
        for key, a in reference.items():
            b = actual[key]
            assert a.legal_ids == b.legal_ids and a.move_uci == b.move_uci
            errors.append(max(abs(x-y) for x, y in zip(a.policy, b.policy)))
        assert max(errors) <= 2e-4, 'Compiled/eager search parity failed'
        atomic_json(out/f'compile_check_{label}.json', dict(passed=True, max_policy_error=max(errors), measurement=measurement))
        eager.clear_caches()
        runtimes[label] = compiled
        del eager, compiled
        gc.collect()
        torch.cuda.empty_cache()
    pool = None
    try:
        if opponent == 'stockfish':
            pool = StockfishPool(runtimes[candidate], meta['concurrency'], 2400)
            pool.executors = {'stockfish': pool.execute}
            runtimes['stockfish'] = pool
        play_match(out=out, seeds=seeds, candidate_id=candidate, opponent_id=opponent,
                   runtimes=runtimes, search=cfg.search, max_positions=min(limits),
                   concurrency=meta['concurrency'], release=pool.release if pool else lambda key: None)
    finally:
        for runtime in runtimes.values():
            if hasattr(runtime, 'clear_caches'):
                runtime.clear_caches()
        if pool:
            pool.close()


if __name__ == '__main__':
    main()
