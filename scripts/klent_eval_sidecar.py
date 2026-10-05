"""Evaluate each KLENT actor snapshot against Stockfish 2600 as it appears.

Runs beside training and never touches it: waits for actor-NNNN.pt files
(written atomically every --save-every iterations), plays the greedy policy
(no search) against limited-strength Stockfish with the same settings as the
standing SF2600 ruler, and records the score in <run>/sf2600/ and in
TensorBoard at the snapshot's position count. Restartable: snapshots that
already have a result are skipped.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from torch.utils.tensorboard import SummaryWriter

STOCKFISH_ARGS = [
    "--stockfish-limit-strength", "--stockfish-elo", "2600",
    "--stockfish-path", "/usr/bin/stockfish", "--stockfish-time-sec", "5",
    "--stockfish-nodes", "40000", "--stockfish-threads", "1", "--stockfish-hash-mb", "64",
]


def positions_at(run, iteration):
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["iteration"] == iteration:
            return row["positions"]
    raise ValueError(f"no metrics row for iteration {iteration}")


def evaluate(run, snapshot, args):
    out = run / "sf2600" / f"{snapshot.stem}-policy.json"
    if out.exists():
        return None
    command = [
        sys.executable, "scripts/eval_vs_stockfish.py",
        "--config", str(args.config), "--checkpoint", str(snapshot),
        "--model-move-policy", "policy", "--inference-dtype", "bfloat16",
        "--games", str(args.games), "--concurrent-games", str(args.concurrent_games),
        *STOCKFISH_ARGS, "--output-json", str(out),
    ]
    start = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(out.read_text())
    iteration = int(snapshot.stem.split("-")[1])
    row = dict(
        snapshot=snapshot.name,
        iteration=iteration,
        positions=positions_at(run, iteration),
        games=args.games,
        score_rate=result["score_rate"],
        win_rate=result["win_rate"],
        draw_rate=result["draw_rate"],
        loss_rate=result["loss_rate"],
        seconds=round(time.perf_counter() - start, 1),
    )
    with (run / "sf2600" / "summary.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="KLENT output directory")
    parser.add_argument("--config", type=Path, default=Path("config/imba_chess_v4.toml"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrent-games", type=int, default=8)
    parser.add_argument("--every", type=int, default=40,
                        help="Evaluate only snapshots whose iteration is a multiple of this.")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    (args.run / "sf2600").mkdir(exist_ok=True)
    writer = SummaryWriter(args.run / "tb_sf2600")
    while True:
        for snapshot in sorted(args.run.glob("actor-*.pt")):
            if int(snapshot.stem.split("-")[1]) % args.every:
                continue
            row = evaluate(args.run, snapshot, args)
            if row is not None:
                print(json.dumps(row), flush=True)
                for key in ("score_rate", "win_rate", "draw_rate", "loss_rate"):
                    writer.add_scalar(f"sf2600_policy/{key}", row[key], row["positions"])
                writer.flush()
        if not args.follow:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
