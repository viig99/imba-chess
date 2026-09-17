"""Read normal self-play metrics once a minute; no torch, CUDA or replay reads."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    path = args.run / 'metrics.jsonl'
    offset = 0
    collections = {}
    steps = 0

    def read():
        nonlocal offset, steps
        if path.exists():
            with path.open() as stream:
                stream.seek(offset)
                while line := stream.readline():
                    if not line.endswith('\n'):
                        break
                    row = json.loads(line)
                    offset = stream.tell()
                    if row.get('phase') == 'collect' and 'completed_games' in row:
                        old = collections.get(row['iteration'], (0, 0))
                        collections[row['iteration']] = (
                            max(old[0], row['completed_games']),
                            max(old[1], row['searched_positions']),
                        )
                    if row.get('phase') == 'train' and 'steps' in row:
                        steps = max(steps, row['steps'])
        return (sum(x[0] for x in collections.values()),
                sum(x[1] for x in collections.values()), steps)

    baseline = read()
    started = time.monotonic()
    while (elapsed := time.monotonic() - started) < args.seconds:
        time.sleep(min(60, args.seconds - elapsed))
        elapsed = time.monotonic() - started
        counts = [n - b for n, b in zip(read(), baseline)]
        print(json.dumps(dict(
            elapsed_seconds=round(elapsed, 1),
            completed_games=counts[0], searched_positions=counts[1], optimizer_steps=counts[2],
            games_per_hour=round(counts[0] * 3600 / elapsed, 1),
            positions_per_hour=round(counts[1] * 3600 / elapsed, 1),
            optimizer_steps_per_hour=round(counts[2] * 3600 / elapsed, 1),
        )), flush=True)


if __name__ == '__main__':
    main()
