"""Pull and freeze the latest remote actor; run fixed paired local evaluations."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/eval/remote5090-q01-detached-value-eval01-2026-09-20'
RUN = ROOT / 'artifacts/self_play/remote5090-ckpt34-s200-q01-detached-value-surprise-2026-09-20'
BASE = ROOT / 'artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt'
ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def summarize(path):
    if not path.exists():
        return {'status': 'not_started'}
    data = json.loads(path.read_text())
    counts = dict(wins=0, draws=0, losses=0, incomplete=0)
    for row in data.get('results', {}).values():
        if row['status'] != 'completed':
            counts['incomplete'] += 1
            continue
        value = row['outcome_white'] * (1 if row['candidate_white'] else -1)
        counts[{1: 'wins', 0: 'draws', -1: 'losses'}[value]] += 1
    return dict(status='complete' if 'interval' in data else 'incomplete', **counts, interval=data.get('interval'))


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        result = subprocess.run(['bash', 'sync_remote.sh', 'pull'], timeout=1800)
        if result.returncode == 0:
            break
    else:
        raise RuntimeError('Remote artifact sync failed')
    state = json.loads((RUN / 'state.json').read_text())
    actor = ROOT / state['actor']
    batch = OUT / actor.stem
    batch.mkdir(exist_ok=True)
    frozen = batch / 'candidate.pt'
    if not frozen.exists():
        os.link(actor, frozen)
    digest = hashlib.file_digest(frozen.open('rb'), 'sha256').hexdigest()
    save(batch / 'manifest.json', dict(checkpoint_sha256=digest, source=str(actor), remote_state=state, simulations=512, scale=0.1, exploration="zero_gumbel_noise", pairs=50, stockfish_elo=2400, stockfish_nodes=40000, created=time.time()))
    legs = [
        ('vs_ckpt34', frozen, ['--best', str(BASE)], batch / 'vs_ckpt34'),
        ('vs_stockfish2400', frozen, ['--stockfish', '/usr/bin/stockfish', '--stockfish-limit-strength', '--stockfish-elo', '2400'], batch / 'vs_stockfish2400'),
        ('ckpt34_vs_stockfish2400', BASE, ['--stockfish', '/usr/bin/stockfish', '--stockfish-limit-strength', '--stockfish-elo', '2400'], OUT / 'baseline_stockfish2400'),
    ]
    statuses = {}
    for label, candidate, opponent, directory in legs:
        directory.mkdir(exist_ok=True)
        output = directory / 'results.json'
        cmd = [sys.executable, '-u', 'scripts/eval_self_play.py', '--config', str(ROOT / 'config/self_play_eval.toml'), '--checkpoint', str(candidate), *opponent, '--seeds', str(OUT / 'monitor-seeds.json'), '--output', str(output), '--pairs', '50', '--seconds', '5400', '--device', 'cuda']
        save(directory / 'command.json', cmd)
        if summarize(output)['status'] != 'complete':
            print(f'START {actor.stem} {label}', flush=True)
            with (directory / 'eval.log').open('a') as log:
                result = subprocess.run(cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT, timeout=5500)
            print(f'END {label}: {result.returncode}', flush=True)
        statuses[label] = summarize(output)
        save(batch / 'summary.json', statuses)
        save(OUT / 'latest.json', dict(actor=actor.stem, checkpoint_sha256=digest, results=statuses, updated=time.time()))
    # Preserve the running comparison baseline and three most recent evaluated actors.
    candidates = sorted(OUT.glob('actor-*/candidate.pt'), key=lambda p: p.parent.name, reverse=True)
    for old in candidates[3:]:
        old.unlink()


if __name__ == '__main__':
    main()
