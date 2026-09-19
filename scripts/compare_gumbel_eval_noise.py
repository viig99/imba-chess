"""Re-evaluate frozen remote actors with zero noise against matched noisy results.

Preserves checkpoint, opponent, config, openings and pair count. Copies inputs
before launching, verifies identities, and reports paired differences over
opening pairs. Existing noisy results are retained as the control arm.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import run_lock
from imba_chess.self_play.seeds import file_hash, load_seeds


def pair_scores(data):
    count = data['identity']['pairs']
    scores = []
    for i in range(count):
        rows = [data['results'][f'{i}:{color}'] for color in (0, 1)]
        if any(r['status'] != 'completed' for r in rows):
            raise ValueError('cannot summarize unfinished pairs')
        scores.append(sum((r['outcome_white'] * (1 if r['candidate_white'] else -1) + 1) / 2 for r in rows) / 2)
    return scores


def comparison(noisy, zero):
    expected = copy.deepcopy(noisy['identity'])
    expected['inference']['exploration'] = 'zero_gumbel_noise'
    if zero['identity'] != expected:
        raise ValueError('comparison changed more than evaluation noise')
    a, b = pair_scores(noisy), pair_scores(zero)
    differences = [y - x for x, y in zip(a, b)]
    rng = random.Random(42)
    bootstrap = sorted(sum(rng.choices(differences, k=len(a))) / len(a) for _ in range(10000))
    return dict(noisy_score=sum(a)/len(a), zero_noise_score=sum(b)/len(b),
                difference=sum(differences)/len(a), difference_lower=bootstrap[250],
                difference_upper=bootstrap[9750], pairs=len(a),
                noisy_interval=noisy['interval'], zero_noise_interval=zero['interval'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--best', type=Path, required=True)
    parser.add_argument('--actors', nargs='+', default=['actor-000073', 'actor-000053', 'actor-000033'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=7200)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output):
        cfg = load_config(args.source / 'config.toml')
        seeds = load_seeds(args.source / 'monitor-seeds.json', 'monitor')
        best_hash = file_hash(args.best)
        for filename in ('config.toml', 'monitor-seeds.json'):
            source, target = args.source / filename, args.output / filename
            if not target.exists():
                shutil.copy2(source, target)
            if file_hash(source) != file_hash(target):
                raise ValueError('frozen config/openings changed')
        frozen_best = args.output / 'ckpt34.pt'
        if not frozen_best.exists():
            shutil.copy2(args.best, frozen_best)
        if file_hash(frozen_best) != best_hash:
            raise ValueError('baseline checkpoint changed')
        summary = {}
        # Freeze all candidates now, before any periodic cleanup can remove them.
        for actor in args.actors:
            directory = args.output / actor
            directory.mkdir(exist_ok=True)
            checkpoint = directory / 'candidate.pt'
            noisy_path = directory / 'noisy.json'
            if not checkpoint.exists():
                shutil.copy2(args.source / actor / 'candidate.pt', checkpoint)
            if not noisy_path.exists():
                shutil.copy2(args.source / actor / 'vs_ckpt34/results.json', noisy_path)
            noisy = json.loads(noisy_path.read_text())
            identity = noisy['identity']
            if (identity['candidate'] != file_hash(checkpoint) or identity['best'] != best_hash
                or identity['config'] != cfg.identifier or identity['seeds'] != [s.seed_id for s in seeds[:identity['pairs']]]
                or identity['inference']['exploration'] != 'gumbel_noise' or 'interval' not in noisy):
                raise ValueError(f'{actor}: historical control does not match frozen inputs')
        for actor in args.actors:
            directory = args.output / actor
            noisy = json.loads((directory / 'noisy.json').read_text())
            output = directory / 'zero/results.json'
            command = [sys.executable, '-u', 'scripts/eval_self_play.py',
                       '--config', str(args.output / 'config.toml'),
                       '--checkpoint', str(directory / 'candidate.pt'), '--best', str(frozen_best),
                       '--seeds', str(args.output / 'monitor-seeds.json'), '--pairs', str(noisy['identity']['pairs']),
                       '--seconds', str(args.seconds), '--device', args.device,
                       '--output', str(output)]
            atomic_json(directory / 'command.json', command)
            atomic_json(args.output / 'status.json', dict(actor=actor, status='running', updated=time.time(), completed=summary))
            if not output.exists() or 'interval' not in json.loads(output.read_text()):
                print(f'START {actor}', flush=True)
                with (directory / 'eval.log').open('a') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                                            timeout=args.seconds + 120)
                if result.returncode:
                    raise RuntimeError(f'{actor}: evaluator failed ({result.returncode}); see eval.log')
            zero = json.loads(output.read_text())
            if 'interval' not in zero:
                raise RuntimeError(f'{actor}: incomplete evaluation; rerun campaign to resume')
            summary[actor] = comparison(noisy, zero)
            atomic_json(directory / 'comparison.json', summary[actor])
            atomic_json(args.output / 'summary.json', summary)
            print(json.dumps(dict(actor=actor, **summary[actor])), flush=True)
        atomic_json(args.output / 'status.json', dict(status='complete', updated=time.time(), completed=summary))


if __name__ == '__main__':
    main()
