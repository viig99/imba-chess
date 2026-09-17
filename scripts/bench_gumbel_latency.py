"""Calibrate Gumbel simulation budgets against halving using the same actor backend.

Worker processes stay torch-free. CUDA, model loading, and coordinator imports
are confined to main(). This is an offline experiment, not a training change.
"""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import random
import statistics
import time


def worker(conn, worker_id, vocab_path, board_config):
    import chess
    from imba_chess.data.move_vocab import MoveVocab
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.models import BoardTokenConfig
    from imba_chess.eval.actor_worker import _PlainSequenceHistory, _select_model_move
    from imba_chess.eval.gumbel_search import GumbelConfig
    from imba_chess.eval.search import HalvingConfig

    vocab = MoveVocab.load(vocab_path)
    encoder = BoardStateEncoder(BoardTokenConfig(**board_config))
    while True:
        job = conn.recv()
        if job is None:
            return
        board = chess.Board()
        history = _PlainSequenceHistory(worker_id=worker_id, move_vocab=vocab, board_state_encoder=encoder)
        moves = job['prefix_moves']
        assert len(moves) >= 2
        common = dict(conn=conn, worker_id=worker_id, board=board, move_vocab=vocab,
                      board_state_encoder=encoder, history=history, value_rerank_top_k=16,
                      value_rerank_lambda=.05,
                      halving_config=HalvingConfig(budget=2048, top_m=16, refutation_top_r=4,
                                                  expand_top=3, max_depth=8, lam=.05))
        # Prefill the previous model turn outside timing. Timed requests then
        # append two actual plies, as in the existing Stockfish game workers.
        for move in moves[:-2]:
            history.append_observed_position(board)
            history.record_played_move(move)
            board.push_uci(move)
        _select_model_move(**common, turn_id=0, policy='greedy')
        for move in moves[-2:]:
            history.append_observed_position(board)
            history.record_played_move(move)
            board.push_uci(move)
        assert not board.is_game_over(claim_draw=True)
        conn.send(dict(kind='ready'))
        assert conn.recv() == 'go'
        start = time.perf_counter()
        chosen, debug = _select_model_move(
            **common, turn_id=1,
            policy='value_search_halving' if job['simulations'] == 0 else 'gumbel',
            gumbel_config=GumbelConfig(simulations=job['simulations'], top_m=16, max_depth=32)
            if job['simulations'] else None,
        )
        assert chosen in board.legal_moves
        conn.send(dict(kind='done', seconds=time.perf_counter()-start,
                       move=chosen.uci(), search_stats=debug.get('search_stats', {})))


def receive_until(conns, processes, server, kind):
    from imba_chess.eval.actor_protocol import RootEvalRequest, WaveRequest
    pending = set(range(len(conns)))
    output = {}
    deadline = time.monotonic() + 600
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f'Worker phase {kind} exceeded ten minutes')
        ready = wait([conns[i] for i in pending], timeout=5)
        if not ready:
            if any(not processes[i].is_alive() for i in pending):
                raise RuntimeError('Benchmark worker exited unexpectedly')
            continue
        requests, owners = [], []
        for i in sorted(pending):
            if conns[i] not in ready:
                continue
            message = conns[i].recv()
            if isinstance(message, (RootEvalRequest, WaveRequest)):
                owners.append(i)
                requests.append(message)
            else:
                assert message['kind'] == kind
                output[i] = message
                pending.remove(i)
        if requests:
            responses = server.service(requests)
            for i, response in zip(owners, responses):
                conns[i].send(response)
    return output


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def summarize(rows):
    import numpy as np
    groups = {}
    for row in rows:
        key = (row['checkpoint'], row['concurrency'], row['simulations'])
        groups.setdefault(key, []).append(row)
    output = []
    for (checkpoint, concurrency, simulations), sample in sorted(groups.items()):
        times = [r['seconds'] for r in sample]
        output.append(dict(checkpoint=checkpoint, concurrency=concurrency, simulations=simulations,
                           searches=len(times), mean_seconds=statistics.mean(times),
                           median_seconds=statistics.median(times), p90_seconds=float(np.quantile(times,.9)),
                           group_wall_seconds_per_move=statistics.mean(r['group_wall_seconds']/concurrency for r in sample),
                           peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in sample)))
    return output


def main():
    import torch
    from imba_chess.config import load_repo_config
    from imba_chess.data.move_vocab import MoveVocab
    from imba_chess.eval.actor_server import ActorInferenceServer
    from imba_chess.eval.position_evaluator import load_hstu_checkpoint
    # The queued snapshot contains the established evaluator next to this file.
    import eval_vs_stockfish as evaluation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Optional second checkpoint for a matched comparison')
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulation-budgets', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    args = parser.parse_args()
    if min(args.simulation_budgets) < 1 or len(set(args.simulation_budgets)) != len(args.simulation_budgets):
        parser.error('simulation budgets must be distinct positive integers')
    checkpoints = [('actor111', args.actor)]
    if args.baseline is not None:
        checkpoints.append(('baseline', args.baseline))
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output/'timings.json').exists():
        raise RuntimeError('Use a new output directory; previous timings already exist')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(4042)
    device = torch.device('cuda')
    repo = load_repo_config(args.config)
    vocab = MoveVocab.load(repo.vocab.path)
    positions = json.loads(args.positions.read_text())
    assert len(positions) == 12
    rows = []
    candidates = args.simulation_budgets
    ctx = mp.get_context('spawn')
    # Keep only one checkpoint on the GPU at a time.
    for label, path in checkpoints:
        model, _ = load_hstu_checkpoint(checkpoint_path=path, repo_config=repo,
                                        move_vocab=vocab, device=device,
                                        compile_model=False, require_value_head=True)
        model.eval()
        for concurrency in (1, 4):
            server = ActorInferenceServer(model=model, device=device, dtype=torch.float32)
            conns, processes = [], []
            try:
                for i in range(concurrency):
                    parent, child = ctx.Pipe()
                    process = ctx.Process(target=worker, args=(child, i, str(repo.vocab.path), asdict(repo.board_state)))
                    process.start()
                    child.close()
                    conns.append(parent)
                    processes.append(process)
                def measure(group, simulations):
                    for i, position in enumerate(group):
                        conns[i].send(dict(prefix_moves=position['prefix_moves'], simulations=simulations))
                    with torch.inference_mode():
                        receive_until(conns, processes, server, 'ready')
                        for i in range(concurrency):
                            server.release_turn(i, 0)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        for conn in conns:
                            conn.send('go')
                        result = receive_until(conns, processes, server, 'done')
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-start
                        peak = torch.cuda.max_memory_allocated()
                    for i in range(concurrency):
                        server.release_turn(i, 1)
                        server.release_game(i)
                    return result, elapsed, peak
                # Every variant warms before recording. Repeated trials are
                # interleaved to reduce temperature/order bias.
                for simulations in [0]+candidates:
                    measure(positions[:concurrency], simulations)
                    print(json.dumps(dict(phase='warmup', checkpoint=label, concurrency=concurrency,
                                          simulations=simulations)), flush=True)
                for repeat in range(3):
                    order = [0]+candidates
                    random.Random(4042+repeat).shuffle(order)
                    for simulations in order:
                        for offset in range(0, len(positions), concurrency):
                            group = positions[offset:offset+concurrency]
                            result, elapsed, peak = measure(group, simulations)
                            for i, position in enumerate(group):
                                rows.append(dict(checkpoint=label, concurrency=concurrency, simulations=simulations,
                                                 repeat=repeat, position_id=position['seed_id'],
                                                 group_wall_seconds=elapsed, peak_allocated_bytes=peak, **result[i]))
                            save(args.output/'timings.json', rows)
                        print(json.dumps(dict(phase='timing', checkpoint=label, concurrency=concurrency,
                                              repeat=repeat, simulations=simulations)), flush=True)
            finally:
                for process in processes:
                    if process.is_alive(): process.terminate()
                for process in processes:
                    process.join(timeout=5)
                    if process.is_alive(): process.kill(); process.join()
                for conn in conns: conn.close()
                del server
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    summaries = summarize(rows)
    save(args.output/'timing_summary.json', summaries)
    # Select using time only, separately for interactive play and four-game
    # operation. Average requested checkpoints equally; do not tune on strength.
    choices = {}
    for concurrency in (1, 4):
        ratios = {}
        for simulations in candidates:
            values = []
            for label, _ in checkpoints:
                subset = [r for r in summaries if r['checkpoint']==label and r['concurrency']==concurrency]
                baseline = next(r['mean_seconds'] for r in subset if r['simulations']==0)
                gumbel = next(r['mean_seconds'] for r in subset if r['simulations']==simulations)
                values.append(gumbel/baseline)
            ratios[simulations] = statistics.mean(values)
        eligible = [n for n, ratio in ratios.items() if .85 <= ratio <= 1.15]
        selected = min(eligible, key=lambda n: abs(ratios[n]-1)) if eligible else None
        choices[concurrency] = dict(selected_simulations=selected, ratios=ratios,
                                   tolerance=.15, interpretation='Within 15% of average move latency on this calibration sample; per-position tails may differ.')
    save(args.output/'budget_selection.json', choices)
    selected = choices[4]['selected_simulations']
    if selected is None:
        save(args.output/'status.json', dict(phase='calibration_needs_refinement', reason='No tested budget within 15% at four active games; game tests not launched.'))
        return

    # Fresh strength screens on the SAME backend and selected common budget.
    # Existing full halving runs provide context, but are not paired games.
    for label, path in checkpoints:
        directory = args.output / f'gumbel_sf2400_{label}'
        directory.mkdir(exist_ok=False)
        save(args.output/'status.json', dict(phase='strength_screen', checkpoint=label, simulations=selected, games=100))
        model, _ = load_hstu_checkpoint(checkpoint_path=path, repo_config=repo, move_vocab=vocab,
                                        device=device, compile_model=False, require_value_head=True)
        server = ActorInferenceServer(model=model, device=device, dtype=torch.float32)
        conns, processes = [], []
        try:
            for i in range(4):
                parent, child = ctx.Pipe()
                config = dict(worker_id=i, game_indices=list(range(i,100,4)), seed=5042, max_plies=512,
                              opening_random_plies=0, model_move_policy='gumbel',
                              gumbel_config=dict(simulations=selected, top_m=16, max_depth=32),
                              vocab_path=str(repo.vocab.path), board_state_config=asdict(repo.board_state),
                              engine=dict(stockfish_path='/usr/bin/stockfish',
                                          stockfish_options={'Threads':1,'Hash':64,'UCI_LimitStrength':True,'UCI_Elo':2400},
                                          stockfish_limit=dict(nodes=40000,time=5.0)))
                from imba_chess.eval.actor_worker import run_eval_worker
                process=ctx.Process(target=run_eval_worker, args=(child,config))
                process.start(); child.close()
                conns.append(parent); processes.append(process)
            with torch.inference_mode():
                summary=evaluation._serve_actor_workers(server=server,parent_conns=conns,processes=processes,games=100,segment_name=f'gumbel_{label}')
            evaluation._join_and_verify_workers(processes)
            save(directory/'results.json', asdict(summary))
            if summary.completed_games != 100 or summary.incomplete_games:
                raise RuntimeError('Incomplete Gumbel strength screen')
        finally:
            evaluation._terminate_worker_processes(processes)
            for conn in conns: conn.close()
            del server, model
            torch.cuda.empty_cache()
    strength = {}
    for label, _ in checkpoints:
        summary = json.loads((args.output/f'gumbel_sf2400_{label}/results.json').read_text())
        strength[label] = dict(
            games=summary['completed_games'], wins=summary['wins'], draws=summary['draws'], losses=summary['losses'],
            score=(summary['wins']+.5*summary['draws'])/summary['completed_games'],
            mean_model_move_seconds=summary['model_selection_seconds']/summary['model_turns'],
            simulations=selected,
        )
    save(args.output/'strength_summary.json', dict(
        checkpoints=strength,
        actor_minus_baseline=(strength['actor111']['score']-strength['baseline']['score']) if 'baseline' in strength else None,
        note='100-game screens per checkpoint; descriptive score difference, not a significance claim. Game trajectories differ from latency calibration and previous halving runs.',
    ))
    save(args.output/'status.json', dict(phase='complete', simulations=selected))


if __name__ == '__main__':
    main()
