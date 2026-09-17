"""Benchmark either retained search algorithm on a fixed position manifest.

Run baseline and candidate sequentially on an idle GPU. Compilation/startup and
warm measurements are stored separately; use complete games for adoption too.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import chess
import torch
from torch._dynamo.utils import counters

from imba_chess.config import load_repo_config
from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.eval.inference_runtime import load_runtime
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.eval.search import HalvingConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "positions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--model-move-policy",
        choices=["gumbel", "value_search_halving"],
        default="value_search_halving",
    )
    parser.add_argument("--gumbel-simulations", type=int)
    parser.add_argument("--search-budget", type=int)
    parser.add_argument("--concurrent-games", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4042)
    args = parser.parse_args()
    gumbel = args.model_move_policy == "gumbel"
    if (gumbel and args.search_budget is not None) or (
        not gumbel and args.gumbel_simulations is not None
    ):
        parser.error("budget flag does not match selected algorithm")
    if args.repeats < 5 or args.threads < 1:
        parser.error("at least five repeats and positive thread count required")
    repo = load_repo_config(args.config)
    cfg = repo.eval_vs_stockfish
    concurrency = (
        args.concurrent_games
        if args.concurrent_games is not None
        else (24 if gumbel else cfg.concurrent_games)
    )
    positions = json.loads(args.positions.read_text())
    if not 1 <= concurrency <= len(positions):
        parser.error("positive concurrency must fit position manifest")
    config = (
        GumbelConfig(
            simulations=args.gumbel_simulations
            if args.gumbel_simulations is not None
            else 128
        )
        if gumbel
        else HalvingConfig(
            budget=args.search_budget
            if args.search_budget is not None
            else cfg.search_budget,
            top_m=cfg.search_top_m,
            rounds=cfg.halving_rounds,
            refutation_top_r=cfg.search_refutation_top_r,
            expand_top=cfg.search_expand_top,
            max_depth=cfg.search_max_depth,
            lam=cfg.search_lambda,
            tactical_coverage=cfg.search_tactical_coverage,
            quiescence_plies=cfg.search_quiescence_plies,
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    runtime, _ = load_runtime(
        repo_config=repo, checkpoint=args.checkpoint, algorithm=args.model_move_policy
    )
    torch.cuda.synchronize()
    atomic_json(
        args.output / "metadata.json",
        dict(
            algorithm=args.model_move_policy,
            search=asdict(config),
            concurrency=concurrency,
            exploration="zero_noise" if gumbel else "deterministic",
            precision="float32",
            tf32=False,
            runtime=runtime.options,
            seed=args.seed,
            threads=args.threads,
            interop_threads=1,
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            checkpoint_sha256=digest(args.checkpoint),
            config_sha256=digest(args.config),
            positions_sha256=digest(args.positions),
            harness_sha256=digest(__file__),
            load_seconds=time.perf_counter() - started,
            profiling=False,
        ),
    )
    rows = []

    def fail(key, error):
        raise RuntimeError(f"search {key} failed") from error

    with runtime, torch.inference_mode():
        for repeat in range(-1, args.repeats):
            runtime.clear_caches()
            results = {}
            requests = []
            for index, position in enumerate(positions[:concurrency]):
                board = chess.Board()
                history = _SequenceHistory(
                    move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder
                )
                for move in position["prefix_moves"]:
                    history.append_observed_position(board)
                    history.record_played_move(move)
                    board.push_uci(move)
                requests.append(
                    (
                        str(index),
                        runtime.search(
                            board=board,
                            history=history,
                            actor_id="benchmark",
                            game_id=str(index),
                            config=config,
                            rng=random.Random(args.seed + index),
                            noise=0.0 if gumbel else None,
                        ),
                    )
                )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            BatchScheduler(
                game_factory=iter(requests),
                executors=runtime.executors,
                concurrent_games=concurrency,
                on_game_done=lambda key, result: results.update({key: asdict(result)}),
                on_game_error=fail,
            ).run()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = dict(
                repeat=repeat,
                warmed=repeat >= 0,
                seconds=seconds,
                searches_per_second=concurrency / seconds,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                host_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                compiler={key: dict(value) for key, value in counters.items()},
                results=results,
            )
            rows.append(row)
            atomic_json(args.output / "measurements.json", rows)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("results", "compiler")
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
