"""Freeze replay and prepare matched ckpt34 training and paired evaluation commands.

Preparation is read-only with respect to the source replay and checkpoint.
Run the emitted run.sh separately on an idle training device.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

from imba_chess.data.self_play_store import SelfPlayStore
from imba_chess.self_play.config import load_config
from imba_chess.self_play.seeds import load_seeds
from audit_policy_surprise import audit


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exposures', type=int)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seeds', type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or (args.exposures is not None and args.exposures < 1):
        parser.error('require positive exposures and a positive even game count')
    cfg = load_config(args.config)
    if len(load_seeds(args.seeds, "monitor")) < args.games // 2:
        parser.error('not enough monitoring seeds for requested pairs')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cfg.base_config, out / 'base.toml')
    source = SelfPlayStore(args.replay, read_only=True)
    replay = out / 'replay'
    replay.mkdir()
    # Capture published manifest once. Copy immutable shards; never link live data.
    (replay / 'manifest.json').write_text(json.dumps(source.manifest))
    for name in source.active_shards():
        shutil.copy2(source.directory / name, replay / name)
    shutil.copy2(args.seeds, out / 'monitor-seeds.json')
    checkpoint = out / 'initial.pt'
    shutil.copy2(args.checkpoint, checkpoint)
    frozen = SelfPlayStore(replay, read_only=True)
    report = audit(frozen)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    exposures = args.exposures or 2 * report['positions']
    # Explicit settings make the sole experimental difference reviewable.
    import dataclasses
    settings = dataclasses.asdict(cfg)
    settings.pop('streaming', None)
    commands = []
    for enabled, name in ((False, 'control'), (True, 'weighted')):
        settings['learning']['policy_surprise_enabled'] = enabled
        settings['learning']['policy_surprise_fraction'] = .5
        settings['learning']['policy_surprise_cap'] = 3.
        text = 'base_config = ' + json.dumps(str(out / "base.toml")) + '\n'
        for section, values in settings.items():
            if section == 'base_config':
                continue
            text += '\n[' + section + ']\n'
            text += ''.join(f'{key} = {json.dumps(value)}\n' for key, value in values.items())
        config = out / f'{name}.toml'
        config.write_text(text)
        load_config(config)
        commands.append(shlex.join(['.venv/bin/python', 'scripts/train_self_play.py', '--config', str(config), '--initialize', str(checkpoint), '--replay', str(replay), '--output', str(out / name / 'final.pt'), '--exposures', str(exposures), '--seconds', '604800']) + f' > {shlex.quote(str(out / (name + ".log")))} 2>&1')
    # Refuse evaluation if interruptions or unequal training exposure occurred.
    check = "import torch; a,b=[torch.load(p,weights_only=False,map_location='cpu') for p in __import__('sys').argv[1:]]; assert a['phase_exposures'] >= " + str(exposures) + "; assert all(a[k]==b[k] for k in ('steps','exposures','queue','reuse_counts','sampler_rng')), 'unmatched training exposures'"
    commands.append(shlex.join(['.venv/bin/python', '-c', check, str(out / 'control/final.pt'), str(out / 'weighted/final.pt')]))
    eval_config = out / 'evaluation.toml'
    evaluation = (out / 'control.toml').read_text()
    import re
    evaluation = re.sub(r'^simulations = .*$', 'simulations = 512', evaluation, flags=re.MULTILINE)
    eval_config.write_text(evaluation)
    for name, candidate, opponent in (
        ('control-vs-ckpt34', out / 'control/final.pt', checkpoint),
        ('weighted-vs-ckpt34', out / 'weighted/final.pt', checkpoint),
        ('weighted-vs-control', out / 'weighted/final.pt', out / 'control/final.pt'),
    ):
        commands.append(shlex.join(['.venv/bin/python', 'scripts/eval_self_play.py',
            '--config', str(eval_config), '--checkpoint', str(candidate), '--best', str(opponent),
            '--seeds', str(out / 'monitor-seeds.json'), '--pairs', str(args.games // 2),
            '--seconds', '604800', '--output', str(out / name / 'results.json')]))
    hashes = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
    (out / 'inputs.json').write_text(json.dumps(dict(hashes=hashes, source_checkpoint=str(args.checkpoint.resolve()), exposures=exposures, evaluation_games=args.games), indent=2) + '\n')
    verification = "import hashlib,json,pathlib; p=pathlib.Path(" + repr(str(out)) + "); m=json.loads((p/'inputs.json').read_text()); assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in m['hashes'].items()), 'frozen input changed'"
    (out / 'run.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(Path.cwd())) + '\n' + shlex.join(['.venv/bin/python', '-c', verification]) + '\n' + '\n'.join(commands) + '\n')
    print(json.dumps(dict(output=str(out), audit=report, exposures=exposures), indent=2))


if __name__ == '__main__':
    main()
