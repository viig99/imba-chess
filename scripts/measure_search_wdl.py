"""Read-only study of leaf-WDL averaging; production search stays untouched.

A line observer records the existing search's root-perspective leaf WDL before
its accumulation. GPU inference is outside the tracing scope. A/B/C/D are all
computed from that one realized search, never from rerunning alternative labels.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

from imba_chess.eval import gumbel_search, inference_runtime
from imba_chess.data.self_play_store import atomic_json


class LeafObserver:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.counts = None
        self.edge_sums = None
        self.total = [0.0, 0.0, 0.0]
        self.samples = 0
        self.terminal_samples = 0
        self.odd_depth_samples = 0
        self.metadata = None
        lines, start = inspect.getsourcelines(gumbel_search.gumbel_stepwise)
        matches = [start + i for i, line in enumerate(lines)
                   if 'root_pov = leaf_wdl[::-1] if depth % 2 else leaf_wdl' in line]
        if len(matches) != 1:
            raise RuntimeError('search observation point changed; re-audit instrumentation')
        self.line = matches[0]
        self.code = gumbel_search.gumbel_stepwise.__code__

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        if event == 'line' and frame.f_lineno == self.line:
            self.observe(frame.f_locals)
        return self.trace

    def observe(self, local):
        root, config = local['root_eval'], self.kwargs['config']
        if self.counts is None:
            n = len(root.legal_ids)
            self.counts = [0] * n
            self.edge_sums = [[0.0, 0.0, 0.0] for _ in range(n)]
            noise, priors = list(local['noise']), list(local['priors'])
            order = sorted(range(n), key=lambda i: (-(noise[i] + priors[i]), i))
            top = set(order[:min(config.top_m, n)])
            forcing = {i for i, flag in enumerate(root.legal_forcing) if flag}
            candidates = top | forcing if config.root_forcing else top
            self.metadata = dict(legal_ids=list(root.legal_ids), legal_ucis=list(root.legal_ucis),
                                 legal_forcing=list(root.legal_forcing),
                                 noise=noise, root_log_priors=priors,
                                 base_top_m=sorted(top), forcing_only=sorted(forcing - top),
                                 eligible_root_candidates=len(candidates),
                                 root_candidates=int(local['considered']))
        edge = local['path'][0][1]
        depth, leaf = local['depth'], local['leaf']
        wdl = tuple(local['leaf_wdl'])
        if depth % 2:
            wdl = wdl[::-1]
            self.odd_depth_samples += 1
        self.terminal_samples += leaf.evaluation is None
        self.samples += 1
        self.counts[edge] += 1
        for i, probability in enumerate(wdl):
            self.total[i] += probability
            self.edge_sums[edge][i] += probability

    def finish(self, result):
        if self.samples != result.simulations or self.counts != result.visits:
            raise AssertionError('observed simulation counts do not match the real search')
        current = tuple(x / self.samples for x in self.total)
        if result.search_wdl is None or current != tuple(result.search_wdl):
            raise AssertionError('A does not exactly match result.search_wdl')
        chosen = result.legal_ids.index(result.move_id)
        if not self.counts[chosen]:
            raise AssertionError('chosen move has no observed simulations')
        means = [tuple(x / n for x in sums) if n else None
                 for n, sums in zip(self.counts, self.edge_sums)]
        mass = math.fsum(p for p, n in zip(result.policy, self.counts) if n)
        if mass <= 0:
            raise AssertionError('improved policy has no mass on visited edges')
        weighted = tuple(math.fsum(p * mean[i] / mass
                                  for p, mean in zip(result.policy, means) if mean is not None)
                         for i in range(3))
        labels = dict(A=current, B=means[chosen], C=weighted, D=result.root_wdl)
        for label in labels.values():
            if label is None or any(not math.isfinite(x) or x < 0 for x in label) or abs(sum(label) - 1) > 1e-5:
                raise AssertionError('invalid loss/draw/win label')
        added = self.metadata['forcing_only'] if self.kwargs['config'].root_forcing else []
        return dict(labels=labels, chosen_move=result.move_uci, simulations=self.samples,
                    visits=self.counts, edge_mean_wdl=means, improved_policy=list(result.policy),
                    improved_policy_visited_mass=mass,
                    outside_chosen_share=1 - self.counts[chosen] / self.samples,
                    forcing_added_share=sum(self.counts[i] for i in added) / self.samples,
                    forcing_added_candidates=len(added), visited_root_edges=sum(n > 0 for n in self.counts),
                    terminal_simulations=self.terminal_samples, odd_depth_simulations=self.odd_depth_samples,
                    A_matches_search_wdl=True, **self.metadata)


@contextmanager
def capture_leaf_wdl():
    """Instrument the runtime alias temporarily; no production files are edited."""
    original = inference_runtime.gumbel_stepwise
    observed = {}

    def measured(**kwargs):
        observer = LeafObserver(kwargs)
        gen = original(**kwargs)
        first, answer = True, None
        try:
            while True:
                prior_trace = sys.gettrace()
                if prior_trace is not None:
                    raise RuntimeError('measurement cannot replace an existing Python trace')
                sys.settrace(observer.trace)
                try:
                    request = next(gen) if first else gen.send(answer)
                except StopIteration as stop:
                    observed[id(stop.value)] = observer.finish(stop.value)
                    return stop.value
                finally:
                    sys.settrace(prior_trace)
                first = False
                answer = yield request
        finally:
            gen.close()

    inference_runtime.gumbel_stepwise = measured
    try:
        yield observed
    finally:
        inference_runtime.gumbel_stepwise = original


def position_sample(seed_manifest, *, per_phase=64, run_seed=42):
    from imba_chess.self_play.seeds import load_seeds
    import chess
    seeds = load_seeds(seed_manifest, split='monitor')
    rows = []
    for ply in (12, 32, 64, 96):
        eligible = [s for s in seeds if len(s.prefix_moves) >= ply]
        eligible.sort(key=lambda s: hashlib.sha256(f'{run_seed}:{ply}:{s.source_id}'.encode()).hexdigest())
        for seed in eligible[:per_phase]:
            prefix = list(seed.prefix_moves[:ply])
            board = chess.Board()
            for uci in prefix:
                board.push_uci(uci)
            if board.is_game_over(claim_draw=True):
                continue
            position_id = hashlib.sha256(f'{seed.source_id}:{prefix}'.encode()).hexdigest()
            rows.append(dict(position_id=position_id, source_id=seed.source_id,
                             original_seed_id=seed.seed_id, prefix_moves=prefix,
                             ply=ply, phase='early' if ply < 24 else 'middle' if ply < 80 else 'late',
                             fen=board.fen()))
    if not rows:
        raise ValueError('no eligible monitor positions')
    return rows


def score(wdl):
    return wdl[2] - wdl[0]


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def quantile(p):
        index = p * (len(values) - 1)
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)
    return dict(count=len(values), mean=statistics.fmean(values), p05=quantile(.05),
                median=quantile(.5), p95=quantile(.95), min=values[0], max=values[-1])


def correlation(xs, ys):
    xmean, ymean = statistics.fmean(xs), statistics.fmean(ys)
    xy = sum((x-xmean)*(y-ymean) for x,y in zip(xs,ys))
    denominator = math.sqrt(sum((x-xmean)**2 for x in xs)*sum((y-ymean)**2 for y in ys))
    return xy/denominator if denominator else None


def clustered_interval(rows, metric, *, seed=42, samples=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_id']].append(metric(row))
    groups = list(grouped.values())
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        chosen = rng.choices(groups, k=len(groups))
        means.append(sum(sum(g) for g in chosen) / sum(len(g) for g in chosen))
    means.sort()
    return dict(lower=means[int(.025 * samples)], upper=means[min(samples-1, int(.975 * samples))],
                source_games=len(groups), resamples=samples)


def summarize(rows):
    output = dict(label_order=['loss', 'draw', 'win'], expected_value='P(win) - P(loss)',
                  interpretation='A-B and A-C measure averaging differences, not ground-truth best-play bias',
                  variants={}, paired_controls={})
    for variant in sorted({r['variant'] for r in rows}):
        entries = [r for r in rows if r['variant'] == variant]
        result = dict(positions=len(entries), source_games=len({r['source_id'] for r in entries}),
                      all_A_matches=all(r['A_matches_search_wdl'] for r in entries),
                      labels={label: dict(mean_wdl=[statistics.fmean(r['labels'][label][i] for r in entries) for i in range(3)],
                                         expected_value=distribution([score(r['labels'][label]) for r in entries]))
                              for label in ('A', 'B', 'C', 'D')},
                      outside_chosen_share=distribution([r['outside_chosen_share'] for r in entries]),
                      forcing_added_share=distribution([r['forcing_added_share'] for r in entries]),
                      root_candidates=distribution([r['root_candidates'] for r in entries]), differences={}, by_phase={})
        for other in ('B', 'C', 'D'):
            metric = lambda r, other=other: score(r['labels']['A']) - score(r['labels'][other])
            result['differences']['A_minus_' + other] = dict(
                **distribution([metric(r) for r in entries]),
                ci95=clustered_interval(entries, metric),
                mean_wdl_delta=[statistics.fmean(r['labels']['A'][i]-r['labels'][other][i] for r in entries) for i in range(3)],
                main_value_mix_delta=0.25*statistics.fmean(metric(r) for r in entries))
        for phase in ('early', 'middle', 'late'):
            phase_rows = [r for r in entries if r['phase'] == phase]
            result['by_phase'][phase] = dict(positions=len(phase_rows),
                A_minus_B=distribution([score(r['labels']['A'])-score(r['labels']['B']) for r in phase_rows]),
                outside_chosen_share=distribution([r['outside_chosen_share'] for r in phase_rows]),
                forcing_added_share=distribution([r['forcing_added_share'] for r in phase_rows]))
        gap = [score(r['labels']['A'])-score(r['labels']['B']) for r in entries]
        result['gap_correlations'] = dict(
            outside_chosen_share=correlation([r['outside_chosen_share'] for r in entries], gap),
            forcing_added_share=correlation([r['forcing_added_share'] for r in entries], gap))
        output['variants'][variant] = result
    by_variant = {v: {r['position_id']: r for r in rows if r['variant']==v} for v in output['variants']}
    for name, target, baseline in [('root_forcing_effect', 'root_only', 'no_forcing'),
                                   ('opponent_floor_effect', 'full_tactical', 'root_only')]:
        if target not in by_variant or baseline not in by_variant:
            continue
        ids = sorted(by_variant[target].keys() & by_variant[baseline].keys())
        paired = []
        for position_id in ids:
            t, b = by_variant[target][position_id], by_variant[baseline][position_id]
            paired.append(dict(source_id=t['source_id'],
                delta_A=score(t['labels']['A'])-score(b['labels']['A']),
                delta_gap=(score(t['labels']['A'])-score(t['labels']['B']))-(score(b['labels']['A'])-score(b['labels']['B']))))
        if paired:
            output['paired_controls'][name] = dict(positions=len(paired),
                delta_A=dict(**distribution([r['delta_A'] for r in paired]), ci95=clustered_interval(paired, lambda r:r['delta_A'])),
                delta_A_minus_B=dict(**distribution([r['delta_gap'] for r in paired]), ci95=clustered_interval(paired, lambda r:r['delta_gap'])))
    return output


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open('rb+') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(start)  # Only an interrupted final diagnostic row.
                break
            rows.append(json.loads(line))
    return rows


def main():
    import chess
    import torch
    from imba_chess.config import load_repo_config
    from imba_chess.self_play.config import load_config
    from imba_chess.eval.batch_scheduler import BatchScheduler
    from imba_chess.eval.position_evaluator import _SequenceHistory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, default=Path('artifacts/corpus/v4_self_play_seeds_4096.json'))
    parser.add_argument('--self-play-config', type=Path, default=Path('config/self_play_5090.toml'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-phase', type=int, default=64)
    parser.add_argument('--concurrent-searches', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.per_phase < 1 or args.concurrent_searches < 1:
        parser.error('phase sample size and concurrent searches must be positive')
    if inference_runtime.RUNTIME_REVISION != 'shared-search-v2':
        raise RuntimeError('study requires shared-search-v2')
    cfg = load_config(args.self_play_config)
    positions = position_sample(args.seeds, per_phase=args.per_phase, run_seed=args.seed)
    variants = dict(no_forcing=replace(cfg.search, root_forcing=False, forcing_floor=False),
                    root_only=replace(cfg.search, root_forcing=True, forcing_floor=False),
                    full_tactical=replace(cfg.search, root_forcing=True, forcing_floor=True))
    if args.check:
        print(json.dumps(dict(runtime_revision=inference_runtime.RUNTIME_REVISION, positions=len(positions),
                              by_phase={phase:sum(r['phase']==phase for r in positions) for phase in ('early','middle','late')},
                              searches=len(positions)*len(variants), variants={k:asdict(v) for k,v in variants.items()}),indent=2))
        return
    runtime, max_positions = inference_runtime.load_runtime(repo_config=load_repo_config(cfg.base_config),
                                                           checkpoint=args.checkpoint, device='cuda')
    if runtime.options.get('runtime_revision') != 'shared-search-v2':
        raise RuntimeError('loaded runtime is not shared-search-v2')
    if any(r['ply'] + 2 + cfg.search.max_depth > max_positions for r in positions):
        raise ValueError('sample exceeds model context; do not alter depth to fit')
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dict(runtime=runtime.options, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_hash(args.checkpoint),
                    seeds_sha256=file_hash(args.seeds), model_config_sha256=file_hash(cfg.base_config),
                    self_play_config_sha256=file_hash(args.self_play_config), measurement_sha256=file_hash(__file__),
                    search_source_sha256=file_hash(inspect.getsourcefile(gumbel_search.gumbel_stepwise)),
                    variants={k:asdict(v) for k,v in variants.items()}, run_seed=args.seed,
                    noise='training_gumbel_noise; same RNG seed at a position across configurations',
                    positions=positions, source='monitor human-prefix manifest; outcomes not read',
                    purpose='measurement only; no training, production-search, or replay writes')
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('study inputs changed; use a fresh output directory')
    atomic_json(manifest_path, manifest)
    path = args.out / 'positions.jsonl'
    rows = read_rows(path)
    finished = {(r['position_id'],r['variant']) for r in rows}
    if len(finished) != len(rows):
        raise AssertionError('duplicate completed study rows')
    def publish_progress(state):
        atomic_json(args.out / 'progress.json', dict(state=state, completed_searches=len(rows),
                    planned_searches=len(positions)*len(variants), runtime_revision=runtime.options['runtime_revision']))
    publish_progress('running')
    with runtime, capture_leaf_wdl() as captured, path.open('a') as stream:
        def play(position, variant, search_config):
            board = chess.Board()
            history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
            for uci in position['prefix_moves']:
                history.append_observed_position(board)
                history.record_played_move(uci)
                board.push_uci(uci)
            key = f"{position['position_id']}:{variant}"
            began = time.perf_counter()
            result = yield from runtime.search(board=board, history=history, actor_id='wdl-study', game_id=key,
                config=search_config, rng=random.Random(f"{args.seed}:{position['position_id']}"))
            observed = captured.pop(id(result))
            expected_forcing = [board.is_capture(m) or bool(m.promotion) or board.gives_check(m)
                                for m in map(chess.Move.from_uci, observed['legal_ucis'])]
            observed['root_forcing_flags_verified'] = True
            # The candidate audit records the exact priors/noise consumed by v2.
            if observed['legal_forcing'] != expected_forcing:
                raise AssertionError('runtime root forcing flags differ from actual captures/checks/promotions')
            return dict(**position, variant=variant, seconds=time.perf_counter()-began,
                        config=asdict(search_config), **observed)
        def factory():
            for position in positions:
                for variant, search_config in variants.items():
                    if (position['position_id'],variant) not in finished:
                        yield f"{position['position_id']}:{variant}", play(position,variant,search_config)
        def done(key,row):
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            publish_progress('running')
        def fail(key,exc):
            raise exc
        try:
            BatchScheduler(game_factory=iter(factory()), executors=runtime.executors,
                concurrent_games=args.concurrent_searches, on_game_done=done, on_game_error=fail,
                completion_order=True).run()
        except BaseException:
            publish_progress('interrupted_or_failed')
            raise
    if len(rows) != len(positions)*len(variants):
        raise AssertionError('study did not finish every planned position/configuration')
    # Validate matched controls really used identical root noise and priors.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['position_id']].append(row)
    prior_drift = 0.0
    for entries in grouped.values():
        for row in entries[1:]:
            if row['legal_ids'] != entries[0]['legal_ids'] or row['noise'] != entries[0]['noise']:
                raise AssertionError('paired controls did not use identical legal projections and root noise')
            prior_drift = max(prior_drift, max(abs(a-b) for a,b in zip(row['root_log_priors'],entries[0]['root_log_priors'])))
    if prior_drift > 1e-4:
        raise AssertionError(f'paired root priors differ beyond floating-point batch effects: {prior_drift}')
    summary = summarize(rows)
    summary['max_paired_root_log_prior_difference'] = prior_drift
    summary['sample_limits'] = 'Monitor prefixes only; late-ply sample is limited by available long histories. No external best-play ground truth.'
    atomic_json(args.out / 'summary.json', summary)
    with (args.out / 'positions.csv').open('w') as stream:
        fields = ['position_id','source_id','variant','ply','phase','root_candidates','outside_chosen_share','forcing_added_share']
        fields += [f'{label}_{component}' for label in 'ABCD' for component in ('loss','draw','win','expected_value')]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {k:row[k] for k in fields[:8]}
            for label in 'ABCD':
                flattened.update({f'{label}_{component}':row['labels'][label][i] for i,component in enumerate(('loss','draw','win'))})
                flattened[f'{label}_expected_value'] = score(row['labels'][label])
            writer.writerow(flattened)
    publish_progress('completed')
    print(json.dumps(dict(event='study_completed', positions=len(positions), searches=len(rows),
                         summary=str(args.out/'summary.json'))),flush=True)


if __name__ == '__main__':
    main()
