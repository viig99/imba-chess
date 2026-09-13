"""Read published self-play progress into TensorBoard without touching replay state."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter

from imba_chess.self_play.runtime import run_lock, StopBudget


def published_progress(directory):
    """A published snapshot excludes the collector's bounded pending buffer."""
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())
    manifest_path = directory / "replay" / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"shards": []}
    )
    entries = [g for s in manifest["shards"] for g in s["games"]]
    current = Counter()
    # Shard entries describe all published games, including retired replay data.
    current["published_games"] = len(entries)
    current["published_positions"] = sum(g["positions"] for g in entries)
    current["published_training_positions"] = sum(
        g["positions"] for g in entries if g["split"] == "train"
    )
    return dict(
        current,
        iteration=state["iteration"],
        halted=int(state.get("halted", False)),
        phase={"collect": 0, "train": 1, "evaluate": 2}[state["phase"]],
    )


def scalars(value, prefix=""):
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            yield from scalars(item, name)
        elif isinstance(item, (int, float)) and math.isfinite(item):
            yield name, item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--session", default="", help="Separate TensorBoard run label")
    parser.add_argument(
        "--evaluations", type=Path, help="Also watch a morning evaluation directory"
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    output = args.run / "tensorboard" / args.session
    with (
        run_lock(output),
        StopBudget(seconds=args.seconds) as stop,
        SummaryWriter(str(output)) as writer,
    ):
        writer.add_text(
            "notes",
            "Published replay counts exclude buffered games. Phase: 0 collection, 1 training, 2 evaluation. Counts span all iterations. Training metrics are not held-out strength measurements.",
        )
        offset = 0
        metric_step = 0
        previous = None
        while not stop.stop():
            try:
                progress = published_progress(args.run)
            except FileNotFoundError:
                progress = None
            if progress is not None:
                for key, value in progress.items():
                    writer.add_scalar(
                        f"progress/{key}", value, int(time.monotonic() - stop.started)
                    )
                if progress != previous:
                    print(json.dumps(progress), flush=True)
                    previous = progress
            metrics = args.run / "metrics.jsonl"
            if metrics.exists():
                with metrics.open() as stream:
                    stream.seek(offset)
                    while line := stream.readline():
                        if not line.endswith("\n"):
                            break  # Retry a partially written record on the next poll.
                        row = json.loads(line)
                        metric_step += 1
                        for key, value in scalars(row):
                            writer.add_scalar(
                                f"{row.get('phase', 'metrics')}/{key}",
                                value,
                                metric_step,
                            )
                        offset = stream.tell()
            evaluations = [
                (path, path.stem)
                for path in sorted(args.run.glob("*-*.json"))
                if path.name.startswith(("screen-", "confirmation-"))
            ]
            if args.evaluations:
                evaluations += [
                    (path, "morning-" + path.stem)
                    for path in sorted(args.evaluations.glob("*.json"))
                ]
            for path, label in evaluations:
                evaluation = json.loads(path.read_text())
                if "results" not in evaluation:
                    continue
                rows = list(evaluation.get("results", {}).values())
                values = dict(
                    completed_games=sum(r["status"] == "completed" for r in rows),
                    protocol_failed=int(bool(evaluation.get("protocol_failure"))),
                )
                values.update(evaluation.get("interval", {}))
                for key, value in scalars(values):
                    writer.add_scalar(
                        f"evaluation/{label}/{key}",
                        value,
                        int(time.monotonic() - stop.started),
                    )
            writer.flush()
            time.sleep(min(args.interval, max(0, stop.deadline - time.monotonic())))


if __name__ == "__main__":
    main()
