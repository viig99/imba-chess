"""Small WDL probes on cached frozen features; no production model is trained."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.seeds import file_hash
from scripts.audit_tactical_recognition import ensure_manifest, make_batch


PROTOCOL = dict(version=1, seed=42, outer_folds=5, inner_validation_fraction=.2,
    penalties=[.001, .01, .1, 1.0], optimizer='LBFGS', max_iterations=200,
    tolerance_grad=1e-7, tolerance_change=1e-10, training_dtype='float64',
    extraction_dtype='float32', tf32=False, standard_deviation_floor=.01,
    train_weighting='equal source game, equal rows within game',
    selection_metric='validation game-weighted soft WDL cross entropy',
    bootstrap_replicates=10000,
    probes={'value512': 512, 'shared1024': 1024, 'output_only': 3})


def target_ldw(engine_wdl, turn_white, root_white):
    # Engine records are [win, draw, loss] for the ORIGINAL player.
    # Network targets are [loss, draw, win] for the CURRENT player.
    target = engine_wdl[::-1] if turn_white == root_white else list(engine_wdl)
    if len(target) != 3 or sum(target) != 1000 or min(target) < 0:
        raise ValueError('invalid engine WDL')
    return [v / 1000 for v in target]


def load_rows(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    endpoints = {r['position_id']: r['endpoints']['physical'] for r in summary['positions_detail']}
    rows, files, history_sources = [], [], defaultdict(set)
    for pos in manifest['positions']:
        for arm in ('bad', 'good'):
            path = directory / 'lines' / f'{pos["position_id"]}-{arm}.json'
            baseline_path = directory / 'actor300' / path.name
            files += [path, baseline_path]
            line = json.loads(path.read_text())
            baseline = json.loads(baseline_path.read_text())['states']
            if not line['complete']:
                raise ValueError('incomplete engine line')
            prefix = list(line['prefix'])
            for state in line['states']:
                if state['move']:
                    prefix.append(state['move'])
                if state['terminal']:
                    continue
                hid = hashlib.sha256(' '.join(prefix).encode()).hexdigest()
                history_sources[hid].add(pos['source'])
                rows.append(dict(position_id=pos['position_id'], source=pos['source'], arm=arm,
                    ply=state['ply'], prefix=list(prefix), history_hash=hid, fen=state['fen'],
                    root_white=pos['root_white'], turn_white=state['turn_white'],
                    target=target_ldw(state['verified']['wdl_wdl'], state['turn_white'], pos['root_white']),
                    reference=state['verified']['expectation'],
                    stable=abs(state['engine']['expectation'] - state['verified']['expectation']) <= .1,
                    physical_endpoint=arm == 'bad' and state['ply'] == endpoints[pos['position_id']],
                    baseline_wdl=baseline[state['ply']]['main_wdl_ldw']))
    if any(len(sources) > 1 for sources in history_sources.values()):
        raise ValueError('identical full histories cross source games; merge these groups before splitting')
    return manifest, rows, files


def split_games(rows):
    sources = sorted({r['source'] for r in rows})
    random.Random(PROTOCOL['seed']).shuffle(sources)
    folds = {s: i % PROTOCOL['outer_folds'] for i, s in enumerate(sources)}
    plans = []
    for fold in range(PROTOCOL['outer_folds']):
        test = sorted(s for s in sources if folds[s] == fold)
        development = sorted(set(sources) - set(test))
        random.Random(PROTOCOL['seed'] + 100 + fold).shuffle(development)
        nval = max(1, round(len(development) * PROTOCOL['inner_validation_fraction']))
        validation, train = sorted(development[:nval]), sorted(development[nval:])
        if set(test) & (set(train) | set(validation)) or set(train) & set(validation):
            raise AssertionError('game split leakage')
        plans.append(dict(fold=fold, train=train, validation=validation, test=test))
    return plans


def prepare(args):
    from imba_chess.self_play.config import load_config
    parent, rows, files = load_rows(args.audit)
    checkpoint = Path(parent['checkpoints']['actor300'])
    if file_hash(checkpoint) != parent['checkpoint_hashes']['actor300']:
        raise ValueError('actor checkpoint changed')
    base_hash = file_hash(Path(load_config(Path(parent['config'])).base_config))
    if base_hash != parent['base_config_hash']:
        raise ValueError('base model configuration changed')
    sources = sorted(Path('src/imba_chess').rglob('*.py')) + [Path(__file__), Path('scripts/audit_tactical_recognition.py')]
    identity = dict(protocol=PROTOCOL, checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint), config=parent['config'],
        parent_manifest_sha256=file_hash(args.audit / 'manifest.json'),
        config_sha256=file_hash(Path(parent['config'])),
        base_config_sha256=base_hash,
        input_hashes={str(p): file_hash(p) for p in files + [args.audit / 'summary.json']},
        source_hashes={str(p): file_hash(p) for p in sources},
        rows=len(rows), source_games=len({r['source'] for r in rows}),
        unique_histories=len({r['history_hash'] for r in rows}), folds=split_games(rows))
    ensure_manifest(args.output / 'manifest.json', identity)
    return identity, rows


def extract(args, manifest, rows):
    from imba_chess.eval.position_evaluator import _forward_model
    from imba_chess.self_play.config import load_config
    from imba_chess.self_play.runtime import load_runtime
    path = args.output / 'features.pt'
    if path.exists():
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['manifest_sha256'] != file_hash(args.output / 'manifest.json'):
            raise ValueError('incompatible feature cache')
        print('Verified existing frozen feature cache', flush=True)
        return
    runtime, _ = load_runtime(load_config(Path(manifest['config'])), Path(manifest['checkpoint']), 'cuda')
    captured = {}
    hooks = [runtime.model.value_head[0].register_forward_pre_hook(
                 lambda m, i: captured.__setitem__('shared1024', i[0])),
             runtime.model.value_head[-1].register_forward_pre_hook(
                 lambda m, i: captured.__setitem__('value512', i[0]))]
    tensors = {name: [] for name in ('shared1024', 'value512', 'output_only')}
    maximum_error, reused = 0., {}
    started = time.monotonic()
    try:
        for i, row in enumerate(rows):
            hid = row['history_hash']
            if hid not in reused:
                board, batch = make_batch(runtime, row['prefix'])
                if board.fen() != row['fen']:
                    raise ValueError('feature history mismatch')
                output = _forward_model(model=runtime.model, batch=batch, device=runtime.device, dtype=torch.float32)
                reused[hid] = {k: captured[k][-1].detach().cpu().clone() for k in ('shared1024', 'value512')}
                reused[hid]['output_only'] = output['value_logits'][-1].float().detach().cpu().clone()
            for key in tensors:
                tensors[key].append(reused[hid][key])
            p = reused[hid]['output_only'].softmax(-1)
            error = float((p - torch.tensor(row['baseline_wdl'])).abs().max())
            maximum_error = max(maximum_error, error)
            if error > 2e-4:
                raise AssertionError('cached-feature baseline differs from previous audit')
            if (i + 1) % 200 == 0:
                print(json.dumps(dict(phase='extract', complete=i + 1, total=len(rows))), flush=True)
        data = {k: torch.stack(v) for k, v in tensors.items()}
        data['targets'] = torch.tensor([r['target'] for r in rows])
        data['manifest_sha256'] = file_hash(args.output / 'manifest.json')
        # Verify the frozen final projection exactly reconstructs cached outputs.
        with torch.inference_mode():
            output = runtime.model.value_head[-1](data['value512'].to(runtime.device)).float().cpu()
        reconstruction_error = float((output.softmax(-1) - data['output_only'].softmax(-1)).abs().max())
        if reconstruction_error > 2e-4:
            raise AssertionError('wrong value features captured')
        temporary = path.with_suffix('.tmp')
        torch.save(data, temporary)
        temporary.replace(path)
        atomic_json(args.output / 'rows.json', rows)
        atomic_json(args.output / 'feature-verification.json', dict(
            baseline_max_wdl_error=maximum_error, reconstruction_max_wdl_error=reconstruction_error,
            seconds=time.monotonic() - started, unique_forwards=len(reused),
            shapes={k: list(data[k].shape) for k in tensors}, features_sha256=file_hash(path)))
    finally:
        for hook in hooks:
            hook.remove()
        runtime.clear_caches()


def game_weights(rows, indices):
    counts = defaultdict(int)
    for i in indices:
        counts[rows[i]['source']] += 1
    return torch.tensor([1 / (len(counts) * counts[rows[i]['source']]) for i in indices], dtype=torch.float64)


def soft_ce(logits, targets):
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1)


def fit_linear(features, targets, weights, penalty):
    features, targets = features.double(), targets.double()
    mean = (weights[:, None] * features).sum(0)
    scale = ((weights[:, None] * (features - mean).square()).sum(0)).sqrt().clamp_min(.01)
    x = (features - mean) / scale
    model = torch.nn.Linear(x.shape[1], 3, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=PROTOCOL['max_iterations'],
        tolerance_grad=PROTOCOL['tolerance_grad'], tolerance_change=PROTOCOL['tolerance_change'],
        history_size=20, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        objective = (weights * soft_ce(model(x), targets)).sum() + penalty / 2 * model.weight.square().sum()
        objective.backward()
        return objective
    optimizer.step(closure)
    objective = float(closure().detach())
    gradient_max = max(float(p.grad.abs().max()) for p in model.parameters())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        ce = float((weights * soft_ce(model(x), targets)).sum())
    return dict(mean=mean, scale=scale, state=state,
                diagnostics=dict(objective=objective, training_ce=ce, gradient_max=gradient_max,
                                 iterations=optimizer.state[model.weight]['n_iter']))


def predict(fitted, features):
    x = (features.double() - fitted['mean']) / fitted['scale']
    return F.linear(x, fitted['state']['weight'], fitted['state']['bias'])


def fit(args, manifest, rows):
    path = args.output / 'features.pt'
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['manifest_sha256'] != file_hash(args.output / 'manifest.json'):
        raise ValueError('incompatible feature cache')
    target = data['targets'].double()
    all_indices = list(range(len(rows)))
    for plan in manifest['folds']:
        indices = {split: [i for i in all_indices if rows[i]['source'] in plan[split]]
                   for split in ('train', 'validation', 'test')}
        dev = indices['train'] + indices['validation']
        for name in PROTOCOL['probes']:
            output = args.output / 'folds' / f'{name}-{plan["fold"]}.json'
            if output.exists():
                continue
            x = data[name]
            candidates = []
            for penalty in PROTOCOL['penalties']:
                fitted = fit_linear(x[indices['train']], target[indices['train']],
                                    game_weights(rows, indices['train']), penalty)
                ce = float((game_weights(rows, indices['validation']) * soft_ce(
                    predict(fitted, x[indices['validation']]), target[indices['validation']])).sum())
                candidates.append(dict(penalty=penalty, validation_ce=ce, **fitted['diagnostics']))
            best = min(candidates, key=lambda c: (c['validation_ce'], -c['penalty']))
            fitted = fit_linear(x[dev], target[dev], game_weights(rows, dev), best['penalty'])
            probabilities = predict(fitted, x[indices['test']]).softmax(-1)
            prior = (game_weights(rows, dev)[:, None] * target[dev]).sum(0)
            models = args.output / 'probe_weights'
            models.mkdir(exist_ok=True)
            torch.save(fitted, models / f'{name}-{plan["fold"]}.pt')
            atomic_json(output, dict(probe=name, fold=plan['fold'], test_indices=indices['test'],
                probabilities=probabilities.tolist(), training_prior=prior.tolist(),
                selected_penalty=best['penalty'], candidates=candidates,
                refit=fitted['diagnostics'], train_rows=len(dev), test_rows=len(indices['test'])))
            print(json.dumps(dict(phase='fit', probe=name, fold=plan['fold'], penalty=best['penalty'],
                                  validation_ce=best['validation_ce'], **fitted['diagnostics'])), flush=True)


def grouped_summary(values, rows, mask=None):
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if mask[i]:
            groups[row['source']].append(float(values[i]))
    if not groups:
        return dict(n=0, games=0, mean=None, ci95=None)
    means = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(PROTOCOL['seed'])
    samples = rng.choice(means, size=(PROTOCOL['bootstrap_replicates'], len(means)), replace=True).mean(1)
    return dict(n=int(sum(mask)), games=len(groups), mean=float(means.mean()),
                position_mean=float(np.mean(np.asarray(values)[mask])),
                ci95=np.quantile(samples, [.025, .975]).tolist())


def summarize(args, manifest, rows):
    data = torch.load(args.output / 'features.pt', map_location='cpu', weights_only=True)
    targets = data['targets'].double().numpy()
    probabilities = {'baseline': data['output_only'].double().softmax(-1).numpy(),
                     'training_prior': np.zeros_like(targets)}
    diagnostics = {}
    for name in PROTOCOL['probes']:
        probabilities[name] = np.zeros_like(targets)
        seen, diagnostics[name] = set(), []
        for fold in range(PROTOCOL['outer_folds']):
            d = json.loads((args.output / 'folds' / f'{name}-{fold}.json').read_text())
            if seen.intersection(d['test_indices']):
                raise AssertionError('duplicate out-of-fold predictions')
            seen.update(d['test_indices'])
            probabilities[name][d['test_indices']] = d['probabilities']
            probabilities['training_prior'][d['test_indices']] = d['training_prior']
            diagnostics[name].append({k:v for k,v in d.items() if k not in ('probabilities','test_indices')})
        if seen != set(range(len(rows))):
            raise AssertionError('incomplete out-of-fold predictions')
    masks = dict(all=np.ones(len(rows), dtype=bool), stable=np.array([r['stable'] for r in rows]),
                 physical=np.array([r['physical_endpoint'] for r in rows]))
    target_score = targets[:, 2] + .5 * targets[:, 1]
    base = probabilities['baseline']
    baseline_ce = -(targets * np.log(base.clip(1e-12))).sum(1)
    baseline_error = np.abs(base[:, 2] + .5 * base[:, 1] - target_score)
    result = dict(rows=len(rows), source_games=manifest['source_games'], metrics={}, physical_cases=[],
                  diagnostics=diagnostics, game_scores={})
    pair_indices = defaultdict(dict)
    for i, r in enumerate(rows):
        if r['ply'] == 1:
            pair_indices[r['position_id']][r['arm']] = i
    flip = np.array([r['turn_white'] != r['root_white'] for r in rows])
    for name, probs in probabilities.items():
        ce = -(targets * np.log(probs.clip(1e-12))).sum(1)
        entropy = -(targets * np.log(targets.clip(1e-12))).sum(1)
        score = probs[:, 2] + .5 * probs[:, 1]
        error = np.abs(score - target_score)
        result['metrics'][name] = {subset: {
            'ce': grouped_summary(ce, rows, mask), 'kl': grouped_summary(ce - entropy, rows, mask),
            'mae': grouped_summary(error, rows, mask),
            'ce_minus_baseline': grouped_summary(ce - baseline_ce, rows, mask),
            'mae_minus_baseline': grouped_summary(error - baseline_error, rows, mask)}
            for subset, mask in masks.items()}
        root_scores = np.where(flip, 1 - score, score)
        pair_rows, correct = [], []
        for pid, pair in sorted(pair_indices.items()):
            if set(pair) != {'bad', 'good'}:
                continue
            assert rows[pair['bad']]['source'] == rows[pair['good']]['source']
            pair_rows.append(rows[pair['bad']])
            correct.append(float(root_scores[pair['good']] > root_scores[pair['bad']] + 1e-12))
        result['metrics'][name]['pair_ordering'] = grouped_summary(np.array(correct), pair_rows)
        result['metrics'][name]['pair_ordering']['correct_pairs'] = int(sum(correct))
        result['metrics'][name]['pair_ordering']['total_pairs'] = len(correct)
        result['game_scores'][name] = {source: dict(
            ce=float(ce[[i for i,r in enumerate(rows) if r['source']==source]].mean()),
            mae=float(error[[i for i,r in enumerate(rows) if r['source']==source]].mean()))
            for source in sorted({r['source'] for r in rows})}
        for i in np.flatnonzero(masks['physical']):
            result['physical_cases'].append(dict(probe=name, position_id=rows[i]['position_id'],
                source=rows[i]['source'], ply=rows[i]['ply'], reference=rows[i]['reference'],
                prediction=float(root_scores[i]), absolute_error=float(error[i])))
    atomic_json(args.output / 'summary.json', result)
    atomic_json(args.output / 'out_of_fold.json', dict(
        probabilities={k:v.tolist() for k,v in probabilities.items()},
        order='same as rows.json; probabilities are loss/draw/win from side to move'))
    print(json.dumps({name: dict(ce=m['all']['ce']['mean'], mae=m['all']['mae']['mean'],
                                correct_pairs=m['pair_ordering']['correct_pairs'])
                      for name,m in result['metrics'].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['extract', 'fit', 'summarize'])
    parser.add_argument('--audit', type=Path, default=Path('artifacts/eval/tactical-recognition-2026-09-24'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/eval/frozen-value-probes-2026-09-24'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    from imba_chess.self_play.runtime import run_lock
    with run_lock(args.output):
        manifest, rows = prepare(args)
        globals()[args.phase](args, manifest, rows)


if __name__ == '__main__':
    main()
