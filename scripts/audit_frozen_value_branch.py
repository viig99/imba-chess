"""Exploratory private-branch refit after linear probes; shared features stay frozen."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.seeds import file_hash
from scripts.audit_tactical_recognition import ensure_manifest
import scripts.audit_frozen_value_probes as probes


PROTOCOL = dict(version=1, learning_rates=[1e-4, 1e-3], max_steps=200,
                validation_every=10, weight_decay=.01, seed=42,
                decay='matrix weights only; no bias or normalization decay',
                initialization='copy of original actor300 private value branch',
                inputs='cached shared1024; no trunk updates', dtype='float32', tf32=False,
                optimizer='AdamW', gradient_clip=1.0,
                selection='minimum inner-validation game-weighted WDL cross entropy; step 0 allowed',
                reason='linear probes improved log loss but did not clearly improve expected-score error or ordering')


def train_copy(template, x, y, weights, lr, steps, *, validation=None):
    head = copy.deepcopy(template).cuda().train().requires_grad_(True)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in head.parameters() if p.ndim >= 2], weight_decay=PROTOCOL['weight_decay']),
        dict(params=[p for p in head.parameters() if p.ndim < 2], weight_decay=0.)], lr=lr)
    history = []
    for step in range(steps + 1):
        if validation is not None and step % PROTOCOL['validation_every'] == 0:
            vx, vy, vw = validation
            with torch.no_grad():
                history.append(dict(step=step, lr=lr,
                                    validation_ce=float((vw * probes.soft_ce(head(vx), vy)).sum())))
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = (weights * probes.soft_ce(head(x), y)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), PROTOCOL['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        training_ce = float((weights * probes.soft_ce(head(x), y)).sum())
    return head, history, training_ce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('artifacts/eval/frozen-value-probes-2026-09-24'))
    args = parser.parse_args()
    parent = args.parent
    output = parent / 'branch_refit'
    from imba_chess.self_play.runtime import load_runtime, run_lock
    from imba_chess.self_play.config import load_config
    torch.set_num_threads(4)
    torch.manual_seed(PROTOCOL['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(output):
        manifest = json.loads((parent / 'manifest.json').read_text())
        rows = json.loads((parent / 'rows.json').read_text())
        if file_hash(Path(manifest['checkpoint'])) != manifest['checkpoint_sha256']:
            raise ValueError('actor weights changed')
        identity = dict(protocol=PROTOCOL, parent_manifest_sha256=file_hash(parent / 'manifest.json'),
                        features_sha256=file_hash(parent / 'features.pt'), source_sha256=file_hash(Path(__file__)),
                        helper_sha256=file_hash(Path(probes.__file__)), folds=manifest['folds'])
        ensure_manifest(output / 'manifest.json', identity)
        data = torch.load(parent / 'features.pt', map_location='cpu', weights_only=True)
        if data['manifest_sha256'] != identity['parent_manifest_sha256']:
            raise ValueError('wrong feature cache')
        runtime, _ = load_runtime(load_config(Path(manifest['config'])), Path(manifest['checkpoint']), 'cuda')
        template = copy.deepcopy(runtime.model.value_head).cpu()
        runtime.clear_caches()
        del runtime
        x, y = data['shared1024'].cuda(), data['targets'].cuda()
        with torch.no_grad():
            reconstructed = copy.deepcopy(template).cuda()(x).softmax(-1).cpu()
        reconstruction_error = float((reconstructed - data['output_only'].softmax(-1)).abs().max())
        if reconstruction_error > 2e-4:
            raise AssertionError('private branch does not reproduce original output')
        started = time.monotonic()
        for plan in manifest['folds']:
            path = output / 'folds' / f'value_branch-{plan["fold"]}.json'
            if path.exists():
                continue
            ids = {key: [i for i,r in enumerate(rows) if r['source'] in plan[key]]
                   for key in ('train', 'validation', 'test')}
            tr, va, te = ids['train'], ids['validation'], ids['test']
            wt, wv = (probes.game_weights(rows, indices).float().cuda() for indices in (tr, va))
            candidates = []
            for lr in PROTOCOL['learning_rates']:
                head, history, training_ce = train_copy(template, x[tr], y[tr], wt, lr,
                    PROTOCOL['max_steps'], validation=(x[va], y[va], wv))
                candidates += history
                del head
            best = min(candidates, key=lambda r: (r['validation_ce'], r['step'], r['lr']))
            dev = tr + va
            wd = probes.game_weights(rows, dev).float().cuda()
            head, _, training_ce = train_copy(template, x[dev], y[dev], wd, best['lr'], best['step'])
            with torch.no_grad():
                probabilities = head(x[te]).softmax(-1).cpu().tolist()
                prior = (wd[:, None] * y[dev]).sum(0).cpu().tolist()
            (output / 'probe_weights').mkdir(exist_ok=True)
            torch.save({k:v.detach().cpu() for k,v in head.state_dict().items()},
                       output / 'probe_weights' / f'value_branch-{plan["fold"]}.pt')
            atomic_json(path, dict(probe='value_branch', fold=plan['fold'], test_indices=te,
                probabilities=probabilities, training_prior=prior, candidates=candidates,
                selected=best, training_ce=training_ce, train_rows=len(dev), test_rows=len(te)))
            print(json.dumps(dict(phase='branch_refit', fold=plan['fold'], selected=best,
                                  training_ce=training_ce, elapsed=time.monotonic() - started)), flush=True)
            del head
        # Reuse the exact same scoring functions and cached data, without changing the primary summary.
        feature_link = output / 'features.pt'
        if not feature_link.exists():
            feature_link.symlink_to('../features.pt')
        probes.PROTOCOL = dict(probes.PROTOCOL, probes={'value_branch': 1024})
        probes.summarize(argparse.Namespace(output=output), manifest, rows)
        atomic_json(output / 'verification.json', dict(original_branch_max_wdl_error=reconstruction_error,
            seconds=time.monotonic() - started, parameters=sum(p.numel() for p in template.parameters()),
            checkpoint_sha256_after=file_hash(Path(manifest['checkpoint'])),
            shared_features_sha256_after=file_hash(parent / 'features.pt')))


if __name__ == '__main__':
    main()
