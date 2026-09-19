"""Start or resume a streaming self-play experiment until the next 08:00 Toronto.

Intended as the command for a user systemd service. run_self_play owns the run
lock, recovery checkpoints, graceful drain, and absolute hard deadline.
"""

import argparse
from datetime import datetime, time, timedelta
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

from imba_chess.self_play.config import load_config
from imba_chess.data.self_play_store import atomic_json


def command(args, now):
    cfg = load_config(args.config)
    if cfg.streaming is None:
        raise ValueError(
            "nightly streaming launcher requires [streaming] configuration"
        )
    until = args.until
    if until is None:
        morning = now.date() + timedelta(days=now.hour >= 8)
        until = datetime.combine(morning, time(8), tzinfo=ZoneInfo("America/Toronto"))
    if until.tzinfo is None:
        raise ValueError("--until needs an explicit timezone")
    if (until - now).total_seconds() <= cfg.run.reserve_minutes * 60:
        raise ValueError("not enough time before the morning reserve")
    source = (
        ["--resume"]
        if (args.run / "state.json").exists()
        else ["--initialize", str(args.initialize)]
    )
    return [
        sys.executable,
        "-u",
        "scripts/run_self_play.py",
        "--config",
        str(args.config),
        "--seeds",
        str(args.seeds),
        "--output",
        str(args.run),
        *source,
        "--device",
        "cuda",
        "--until",
        until.isoformat(),
        "--screen-seconds",
        "10800",
        "--checkpoint-seconds",
        "3600",
        "--keep-recovery-checkpoints",
        "2",
        "--observe-only-screen",
        "--defer-confirmation",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--until", type=datetime.fromisoformat)
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("America/Toronto"))
    argv = command(args, now)
    print("Starting nightly streaming self-play:", argv, flush=True)
    until = datetime.fromisoformat(argv[argv.index("--until") + 1])
    session = args.run / "sessions" / now.strftime("%Y-%m-%d-%H%M%S")
    session.mkdir(parents=True)
    with (session / "rates.jsonl").open("ab", buffering=0) as rates:
        monitor = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "scripts/monitor_self_play_rates.py",
                "--run",
                str(args.run),
                "--seconds",
                str((until - now).total_seconds()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=rates,
            stderr=subprocess.STDOUT,
        )
        try:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)

            def forward(signum, frame):
                if child.poll() is None:
                    child.send_signal(signum)

            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, forward)
            atomic_json(
                session / "launch.json",
                dict(
                    command=argv,
                    pid=child.pid,
                    started=now.isoformat(),
                    until=until.isoformat(),
                ),
            )
            code = child.wait()
            atomic_json(
                session / "finished.json",
                dict(
                    exit_code=code,
                    finished=datetime.now(ZoneInfo("America/Toronto")).isoformat(),
                ),
            )
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    sys.exit(code)


if __name__ == "__main__":
    main()
