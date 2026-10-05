"""Evaluate a KLENT run's snapshots against Stockfish 2600, beside training.

Never touches the training process. For every actor snapshot whose iteration
is a multiple of --every:
  1. copy it to <run>/keep/ (the run's rolling --keep-snapshots pruning only
     looks at the run directory, so kept copies survive);
  2. play greedy "policy" (argmax pi, the KLENT paper's eval) and greedy
     "policy_q" (argmax of KLENT's improved policy pi', uses the Q head).
  3. play it head-to-head against the previous kept snapshot, both sides
     greedy policy_q, with paired openings and colour reversal ("h2h_prev":
     a working self-play loop should keep beating its own earlier versions).
Greedy evals are light enough to share the GPU with training. Once the final
snapshot (--final-iteration) is kept, training is over and the GPU is free,
so it runs Gumbel 512 (the best 512-budget search: raw Q, root forcing,
forcing floor, minimax 0.5) on the --baseline checkpoint and every kept
snapshot.

Scores go to <run>/sf2600/summary.jsonl and TensorBoard (<run>/tb_sf2600),
at the snapshot's position count (0 for the baseline). Restartable: any
evaluation whose result JSON exists is skipped.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from torch.utils.tensorboard import SummaryWriter

# The repo configs set ladder mode ([eval_vs_stockfish] ladder_elos), which
# overrides --stockfish-elo and --games, so the Elo and game count go through
# the ladder flags. Ladder segments set UCI_LimitStrength and UCI_Elo
# themselves (--stockfish-limit-strength would demand --stockfish-elo).
STOCKFISH_ARGS = [
    "--ladder-elos", "2600",
    "--stockfish-path", "/usr/bin/stockfish", "--stockfish-time-sec", "5",
    "--stockfish-nodes", "40000", "--stockfish-threads", "1", "--stockfish-hash-mb", "64",
]
MODES = {
    "policy": ["--model-move-policy", "policy"],
    "policy_q": ["--model-move-policy", "policy_q"],
    "gumbel512": [
        "--model-move-policy", "gumbel", "--gumbel-simulations", "512",
        "--gumbel-root-forcing", "--gumbel-forcing-floor", "--gumbel-minimax-weight", "0.5",
    ],
}


def positions_at(run, iteration):
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["iteration"] == iteration:
            return row["positions"]
    raise ValueError(f"no metrics row for iteration {iteration}")


def evaluate(run, checkpoint, name, mode, args):
    """Play one evaluation; returns a summary row, or None if already done."""
    out = run / "sf2600" / f"{name}-{mode}.json"
    if out.exists():
        return None
    command = [
        sys.executable, "scripts/eval_vs_stockfish.py",
        "--config", str(args.config), "--checkpoint", str(checkpoint),
        *MODES[mode], "--inference-dtype", "bfloat16",
        "--ladder-games-per-segment", str(args.games),
        "--concurrent-games", str(args.concurrent_games),
        *STOCKFISH_ARGS, "--output-json", str(out),
    ]
    start = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    # Segmented (ladder) output nests the totals under "aggregate".
    result = json.loads(out.read_text())
    result = result.get("aggregate", result)
    if result["rate_denominator_games"] != args.games:
        raise ValueError(f"expected {args.games} games, eval played {result['rate_denominator_games']}")
    iteration = 0 if name == "baseline" else int(name.split("-")[1])
    row = dict(
        name=name,
        mode=mode,
        iteration=iteration,
        positions=0 if name == "baseline" else positions_at(run, iteration),
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


def head_to_head(run, snapshot, previous, args):
    """Snapshot vs the previous kept snapshot; returns a summary row or None."""
    out = run / "sf2600" / f"{snapshot.stem}-vs-{previous.stem}-h2h.json"
    if out.exists():
        return None
    command = [
        sys.executable, "scripts/match_two_checkpoints.py",
        "--config", str(args.config),
        "--checkpoint-a", str(snapshot), "--label-a", snapshot.stem,
        "--checkpoint-b", str(previous), "--label-b", previous.stem,
        "--inference-dtype-a", "bfloat16", "--inference-dtype-b", "bfloat16",
        "--model-move-policy", "policy_q",
        "--games", str(args.h2h_games), "--concurrent-games", str(args.concurrent_games),
        "--output-json", str(out),
    ]
    start = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(out.read_text())
    games = result["a_wins"] + result["a_draws"] + result["a_losses"]
    if games != args.h2h_games:
        raise ValueError(f"expected {args.h2h_games} games, match played {games}")
    iteration = int(snapshot.stem.split("-")[1])
    row = dict(
        name=f"{snapshot.stem}-vs-{previous.stem}",
        mode="h2h_prev",
        iteration=iteration,
        positions=positions_at(run, iteration),
        games=games,
        score_rate=result["a_score_rate"],
        win_rate=result["a_wins"] / games,
        draw_rate=result["a_draws"] / games,
        loss_rate=result["a_losses"] / games,
        score_se=result["a_score_se"],
        seconds=round(time.perf_counter() - start, 1),
    )
    with (run / "sf2600" / "summary.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def keep_snapshots(run, every):
    """Copy every multiple-of-`every` snapshot into <run>/keep/ (atomically)."""
    keep = run / "keep"
    keep.mkdir(exist_ok=True)
    for snapshot in sorted(run.glob("actor-*.pt")):
        if int(snapshot.stem.split("-")[1]) % every or (keep / snapshot.name).exists():
            continue
        tmp = keep / (snapshot.name + ".tmp")
        shutil.copy2(snapshot, tmp)
        os.replace(tmp, keep / snapshot.name)
    return sorted(keep.glob("actor-*.pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="KLENT output directory")
    parser.add_argument("--config", type=Path, default=Path("config/imba_chess_v4.toml"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--h2h-games", type=int, default=400,
                        help="Games per snapshot-vs-previous match (paired openings).")
    parser.add_argument("--concurrent-games", type=int, default=8)
    parser.add_argument("--every", type=int, default=40,
                        help="Evaluate only snapshots whose iteration is a multiple of this.")
    parser.add_argument("--final-iteration", type=int, default=200,
                        help="Gumbel evals start once this snapshot exists (training is done).")
    parser.add_argument("--baseline", type=Path,
                        default=Path("artifacts/checkpoints_keep/flatten53250_supervised_last_checkpoint.pt"))
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    (args.run / "sf2600").mkdir(exist_ok=True)
    writer = SummaryWriter(args.run / "tb_sf2600")

    def record(row):
        if row is None:
            return
        print(json.dumps(row), flush=True)
        for key in ("score_rate", "win_rate", "draw_rate", "loss_rate"):
            writer.add_scalar(f"sf2600_{row['mode']}/{key}", row[key], row["positions"])
        writer.flush()

    # The baseline has no Q head, so it gets the greedy-policy eval only.
    record(evaluate(args.run, args.baseline, "baseline", "policy", args))
    while True:
        kept = keep_snapshots(args.run, args.every)
        for index, snapshot in enumerate(kept):
            for mode in ("policy", "policy_q"):
                record(evaluate(args.run, snapshot, snapshot.stem, mode, args))
            if index > 0:
                record(head_to_head(args.run, snapshot, kept[index - 1], args))
        if any(int(s.stem.split("-")[1]) == args.final_iteration for s in kept):
            record(evaluate(args.run, args.baseline, "baseline", "gumbel512", args))
            for snapshot in kept:
                record(evaluate(args.run, snapshot, snapshot.stem, "gumbel512", args))
            break
        if not args.follow:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
