"""Deterministic single-process self-play record for exact parity checks.

Plays a fixed set of games (fixed checkpoint, seeds, game IDs and concurrency)
through the production game coroutine, scheduler and inference runtime, and
records every search result. Two records from behavior-preserving code must be
identical; `--compare` reports the first difference.

Concurrency fixes GEMM batch shapes, so compare records made with the same
--concurrent-games.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch

from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.self_play.collector import launch_game, raise_game_error
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import load_seeds


def record(args):
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    runtime, max_positions = load_runtime(
        cfg, args.checkpoint, "cuda", dtype=cfg.collection.inference_dtype
    )
    seeds = load_seeds(args.seeds, split="train")[args.seed_offset :]
    results, games = {}, {}

    def factory():
        for index in range(args.games):
            gid = f"parity-{index:04d}"
            results[gid] = []

            def on_position(result, elapsed, gid=gid):
                results[gid].append(asdict(result))

            yield gid, launch_game(
                runtime=runtime, config=cfg, config_id=cfg.identifier,
                actor_id="parity", seed=seeds[index % len(seeds)], game_id=gid,
                max_positions=max_positions,
                # Interrupt after a fixed number of searched positions per game.
                should_stop=lambda gid=gid: len(results[gid]) >= args.plies_per_game,
                on_position=on_position,
            )

    def done(gid, game):
        games[gid] = dict(
            moves=game["moves"], status=game["status"], termination=game["termination"],
            outcome_white=game["outcome_white"],
        )

    start = time.perf_counter()
    BatchScheduler(
        game_factory=iter(factory()), executors=runtime.executors,
        concurrent_games=args.concurrent_games, on_game_done=done,
        on_game_error=raise_game_error, completion_order=True,
    ).run()
    elapsed = time.perf_counter() - start
    runtime.clear_caches()
    payload = dict(
        games={gid: dict(games[gid], searches=results[gid]) for gid in sorted(games)},
        seconds=elapsed,
        positions=sum(len(r) for r in results.values()),
    )
    args.output.write_text(json.dumps(payload))
    print(f"{payload['positions']} positions in {elapsed:.1f}s -> {args.output}")


def first_difference(a, b, path="record"):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for key in a:
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            return f"{path}: {a!r} != {b!r}"
    elif a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def compare(paths):
    reference, candidate = (json.loads(Path(p).read_text()) for p in paths)
    games = reference["games"]
    moves = sum(len(g["moves"]) for g in games.values())
    agree = sum(
        x == y
        for gid in games
        for x, y in zip(games[gid]["moves"], candidate["games"][gid]["moves"])
    )
    print(f"move agreement {agree}/{moves}")
    found = first_difference(reference["games"], candidate["games"])
    print("IDENTICAL" if found is None else f"FIRST DIFFERENCE {found}")
    return found is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--concurrent-games", type=int, default=24)
    parser.add_argument("--plies-per-game", type=int, default=24)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        raise SystemExit(0 if compare(args.compare) else 1)
    record(args)


if __name__ == "__main__":
    main()
