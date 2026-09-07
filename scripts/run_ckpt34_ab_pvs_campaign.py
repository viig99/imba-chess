#!/usr/bin/env python3
"""Prepare or run the fixed equal-evaluation-budget screening campaign.

Default prepares commands/manifests only. --run executes smoke then screen
for each variant, stopping on failures or incomplete games. Never adopts a
policy or launches confirmation matches. Use a new output directory per run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

VARIANTS = [
    ('halving', 'value_search_halving', 'off', False, 8),
    ('alphabeta', 'value_search_alphabeta', 'off', False, 9),
    ('pvs', 'value_search_pvs', 'off', False, 9),
    ('pvs_cache', 'value_search_pvs', 'context', False, 9),
    ('pvs_lmr', 'value_search_pvs', 'off', True, 9),
    ('pvs_cache_lmr', 'value_search_pvs', 'context', True, 9),
]


def sha(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command(args, variant, games, output):
    _, policy, cache, lmr, depth = variant
    return [sys.executable, 'scripts/eval_vs_stockfish.py', '--config', str(args.config.resolve()),
            '--checkpoint', str(args.checkpoint.resolve()), '--stockfish-path', str(args.stockfish.resolve()),
            '--ladder-elos', '2400', '--ladder-games-per-segment', str(games),
            '--no-include-full-strength-segment', '--stockfish-limit-strength', '--stockfish-elo', '2400',
            '--stockfish-nodes', '40000', '--stockfish-time-sec', '5', '--stockfish-threads', '1',
            '--stockfish-hash-mb', '128', '--device', 'cuda', '--dtype', 'float32', '--compile',
            '--concurrent-games', '4', '--seed', '42', '--opening-random-plies', '0', '--max-plies', '512',
            '--model-move-policy', policy, '--search-budget', '2048', '--search-max-depth', str(depth),
            '--search-top-m', '16', '--search-expand-top', '3', '--search-refutation-top-r', '4',
            '--halving-rounds', '0', '--value-rerank-lambda', '0.05', '--no-search-tactical-coverage',
            '--search-quiescence-plies', '0', '--search-iterative-deepening', '--search-score-cache', cache,
            '--search-lmr' if lmr else '--no-search-lmr', '--no-save-games', '--debug-trace-games', '0',
            '--output-json', str(output.resolve())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt'))
    parser.add_argument('--config', type=Path, default=Path('config/imba_chess_v4.toml'))
    parser.add_argument('--stockfish', type=Path, default=Path('/usr/bin/stockfish'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    python=sys.version, platform=platform.platform(),
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                    stockfish=str(args.stockfish.resolve()), stockfish_sha256=sha(args.stockfish),
                    config_sha256=sha(args.config), status='prepared', runs=[])
    (args.output/'working-tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD']))
    (args.output/'git-status.txt').write_bytes(subprocess.check_output(['git','status','--short']))
    shutil.copyfile(args.config, args.output/'config.toml')
    with (args.output/'source.tar').open('wb') as stream:
        subprocess.run(['git','archive','HEAD'], stdout=stream, check=True)
    import torch
    manifest['torch_version'] = torch.__version__
    manifest['cuda_available'] = torch.cuda.is_available()
    manifest['gpu'] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    manifest['blocking_prerequisites'] = []
    if not args.checkpoint.is_file():
        manifest['blocking_prerequisites'].append('ckpt34 file absent')
    if not torch.cuda.is_available():
        manifest['blocking_prerequisites'].append('CUDA unavailable')
    commands = []
    for variant in VARIANTS:
        for phase, games in [('smoke',2), ('screen',100)]:
            name = f'{variant[0]}-{phase}'
            cmd = command(args, variant, games, args.output/f'{name}.json')
            manifest['runs'].append(dict(name=name, games=games, command=cmd, status='not_run'))
            commands.append(shlex.join(cmd))
    (args.output/'commands.sh').write_text('#!/bin/bash\nset -euo pipefail\n'+ '\n'.join(commands)+'\n')
    path = args.output/'manifest.json'
    def save():
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    if not args.run:
        print(f'Prepared {path}; blockers: {manifest["blocking_prerequisites"]}')
        return
    if manifest['blocking_prerequisites']:
        raise SystemExit('Cannot run: '+', '.join(manifest['blocking_prerequisites']))
    try:
        for row in manifest['runs']:
            row['status'] = 'running'
            save()
            with (args.output/f'{row["name"]}.log').open('w') as stream:
                result = subprocess.run(row['command'], stdout=stream, stderr=subprocess.STDOUT)
            row['exit_code'] = result.returncode
            row['status'] = 'failed' if result.returncode else 'finished'
            save()
            if result.returncode:
                raise RuntimeError(f'{row["name"]} failed; see log')
            payload = json.loads((args.output/f'{row["name"]}.json').read_text())['aggregate']
            if not payload['run_config']['compile']:
                row['status'] = 'invalid_protocol'
                raise RuntimeError('Compilation was disabled at runtime; comparison conditions not met')
            row['results'] = {k: payload[k] for k in ['games','completed_games','incomplete_games','wins','draws','losses','score_rate','search_stats','inference_stats']}
            if payload['completed_games'] != row['games']:
                row['status'] = 'incomplete'
                raise RuntimeError(f'{row["name"]} contains incomplete games; screen stopped')
            save()
        manifest['status'] = 'finished'
    except BaseException as exc:
        manifest['status'] = 'stopped'
        manifest['error'] = str(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
