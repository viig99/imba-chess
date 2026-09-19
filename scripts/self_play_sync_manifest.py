"""List current self-play evaluation/monitoring artifacts (not a resume backup)."""
import json
from pathlib import Path
import sys

root = Path('artifacts/self_play')
runs = [p.parent for p in root.glob('*/state.json')]
if not runs:
    raise SystemExit('No self-play run with published state found')
run = max(runs, key=lambda p: (p / 'state.json').stat().st_mtime_ns)
state = json.loads((run / 'state.json').read_text())
files = {run / 'state.json'}
for key in ('actor', 'best', 'checkpoint'):
    p = Path(state[key])
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if not p.is_relative_to(run.resolve()):
        raise ValueError('Checkpoint is outside selected run')
    files.add(Path(p.relative_to(Path.cwd())))
for pattern in ('*.json', '*.jsonl', '*.toml', '*.log', 'operation/**/*', 'tensorboard/**/*'):
    files.update(p for p in run.glob(pattern) if p.is_file() and p.suffix != '.pt')
for p in sorted(files):
    if not p.is_file():
        raise FileNotFoundError(p)
    sys.stdout.buffer.write(str(p.relative_to('artifacts')).encode() + b'\0')
