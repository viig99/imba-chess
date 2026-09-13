"""Resume until an absolute deadline, then run bounded, restartable morning evaluations."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.runtime import run_lock, StopBudget
from imba_chess.self_play.seeds import file_hash


def run_command(command, log_path, stop):
    """Keep child logs and propagate supervisor cancellation to the current child."""
    with Path(log_path).open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while child.poll() is None:
                if stop.stop():
                    child.terminate()
                    try:
                        return child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        return child.wait()
                time.sleep(1)
            return child.returncode
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def morning_commands(
    *, python, config, seeds, output, candidate, baseline, stockfish, seconds
):
    common = [
        python,
        "-u",
        "scripts/eval_self_play.py",
        "--config",
        str(config),
        "--seeds",
        str(seeds),
        "--pairs",
        "50",
        "--device",
        "cuda",
        "--seconds",
        str(seconds),
    ]
    return [
        (
            "candidate-vs-ckpt34",
            common
            + [
                "--checkpoint",
                str(candidate),
                "--best",
                str(baseline),
                "--output",
                str(output / "candidate-vs-ckpt34.json"),
            ],
        ),
        (
            "ckpt34-vs-stockfish",
            common
            + [
                "--checkpoint",
                str(baseline),
                "--stockfish",
                str(stockfish),
                "--output",
                str(output / "ckpt34-vs-stockfish.json"),
            ],
        ),
        (
            "candidate-vs-stockfish",
            common
            + [
                "--checkpoint",
                str(candidate),
                "--stockfish",
                str(stockfish),
                "--output",
                str(output / "candidate-vs-stockfish.json"),
            ],
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", type=datetime.fromisoformat, required=True)
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/bin/stockfish"))
    parser.add_argument("--eval-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.until.tzinfo is None or args.eval_seconds < 1:
        parser.error("deadline needs a timezone and evaluation budget must be positive")
    identity = dict(
        run=str(args.run.resolve()),
        config=file_hash(args.config),
        seeds=file_hash(args.seeds),
        until=args.until.isoformat(),
        eval_seconds=args.eval_seconds,
        stockfish=file_hash(args.stockfish),
    )
    progress_path = args.output / "progress.json"
    seconds = max(0, args.until.timestamp() - time.time()) + 3 * args.eval_seconds + 300
    with run_lock(args.output / "supervisor"), StopBudget(seconds=seconds) as stop:
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["identity"] != identity:
                raise ValueError("overnight output belongs to a different schedule")
        else:
            if args.until.timestamp() <= time.time():
                raise ValueError("new overnight job requires a future deadline")
            progress = dict(identity=identity, phase="train", evaluations={})
            atomic_json(progress_path, progress)
        if progress["phase"] == "train":
            command = [
                sys.executable,
                "-u",
                "scripts/run_self_play.py",
                "--config",
                str(args.config),
                "--seeds",
                str(args.seeds),
                "--output",
                str(args.run),
                "--resume",
                "--device",
                "cuda",
                "--until",
                args.until.isoformat(),
                "--screen-every",
                "3",
                "--defer-confirmation",
            ]
            # On recovery after the deadline, proceed directly to the published actor.
            if time.time() < args.until.timestamp():
                progress["training_command"] = command
                atomic_json(progress_path, progress)
                code = run_command(command, args.output / "training.log", stop)
                progress["training_exit_code"] = code
                if stop.stop():
                    atomic_json(progress_path, progress)
                    return
                if code not in (0, 124):
                    progress.update(
                        phase="failed", error=f"training exited with code {code}"
                    )
                    atomic_json(progress_path, progress)
                    raise RuntimeError(progress["error"])
            progress["phase"] = "waiting"
            atomic_json(progress_path, progress)
        while progress["phase"] == "waiting" and time.time() < args.until.timestamp():
            if stop.stop():
                return
            time.sleep(min(10, max(0, args.until.timestamp() - time.time())))
        if progress["phase"] == "waiting":
            # Latest actor files contain completed training phases, never partial training.
            # Keep the candidate even when a strength screen rolled the collection actor back.
            with run_lock(args.run):
                candidates = sorted(args.run.glob("actor-*.pt"))
                if not candidates:
                    raise FileNotFoundError("no published actor checkpoint")
                baseline = args.run / "actor-000000.pt"
                snapshot = args.output / "checkpoints"
                snapshot.mkdir(exist_ok=True)
                for name, source in (
                    ("candidate", candidates[-1]),
                    ("ckpt34", baseline),
                ):
                    target = snapshot / f"{name}.pt"
                    temp = target.with_suffix(".pt.tmp")
                    shutil.copyfile(source, temp)
                    temp.replace(target)
                    progress[name] = dict(
                        path=str(target.resolve()),
                        sha256=file_hash(target),
                        source=str(source),
                    )
            progress["phase"] = "evaluate"
            atomic_json(progress_path, progress)
        if progress["phase"] == "evaluate":
            commands = morning_commands(
                python=sys.executable,
                config=args.config,
                seeds=args.seeds,
                output=args.output,
                candidate=Path(progress["candidate"]["path"]),
                baseline=Path(progress["ckpt34"]["path"]),
                stockfish=args.stockfish,
                seconds=args.eval_seconds,
            )
            for name, command in commands:
                if stop.stop():
                    return
                if progress["evaluations"].get(name, {}).get("status") in (
                    "completed",
                    "protocol_failed",
                ):
                    continue
                progress["evaluations"][name] = dict(status="running", command=command)
                atomic_json(progress_path, progress)
                code = run_command(command, args.output / f"{name}.log", stop)
                result_path = args.output / f"{name}.json"
                result = (
                    json.loads(result_path.read_text()) if result_path.exists() else {}
                )
                status = (
                    "completed"
                    if "interval" in result
                    else "protocol_failed"
                    if "protocol_failure" in result
                    else "unfinished"
                )
                progress["evaluations"][name].update(
                    status=status, exit_code=code, interval=result.get("interval")
                )
                atomic_json(progress_path, progress)
            progress["phase"] = (
                "complete"
                if all(
                    r["status"] == "completed" for r in progress["evaluations"].values()
                )
                else "evaluate"
            )
            atomic_json(progress_path, progress)
        print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
