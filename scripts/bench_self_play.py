"""Reproducible actual-collector throughput sweeps; profiling is a separate mode."""

import argparse
import cProfile
from dataclasses import asdict, replace
import json
import platform
from pathlib import Path
import resource
import statistics
import subprocess
import time
import torch
from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.self_play.collector import collect
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, StopBudget, run_lock
from imba_chess.self_play.seeds import load_seeds, file_hash


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--component",
        choices=[
            "collection",
            "controller",
            "root",
            "leaf",
            "search",
            "replay",
            "training",
        ],
        default="collection",
    )
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--exposures", type=int, default=8192)
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--simulations", default="32,64,128,256")
    parser.add_argument("--candidates", default="8,16")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--seconds",
        type=float,
        default=3600,
        help="Bound each trial; incomplete games remain unlabeled",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-search-results", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.games, args.threads, args.seconds) <= 0 or args.repeats < 0:
        parser.error(
            "bounds must be positive; repeats may be zero for a first-trial-only run"
        )

    def save_compiler_counters():
        from torch._dynamo.utils import counters

        atomic_json(
            args.output / "compiler_counters.json",
            {k: dict(v) for k, v in counters.items()},
        )

    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    seeds = load_seeds(args.seeds)
    actor_id = file_hash(args.checkpoint)
    with run_lock(args.output):
        start = time.perf_counter()
        runtime, max_positions = load_runtime(cfg, args.checkpoint, args.device)
        synchronize(runtime.device)
        load_seconds = time.perf_counter() - start
        metadata = dict(
            config=asdict(cfg),
            config_id=cfg.identifier,
            checkpoint=str(args.checkpoint),
            checkpoint_sha256=actor_id,
            seed_manifest_sha256=file_hash(args.seeds),
            seed_ids=[s.seed_id for s in seeds],
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            dirty_tree=subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ),
            cpu=platform.processor(),
            platform=platform.platform(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(runtime.device)
            if runtime.device.type == "cuda"
            else None,
            dtype="float32",
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            decoder_source_sha256=file_hash(
                Path("src/imba_chess/model/tensor_decoder.py")
            ),
            executor_source_sha256=file_hash(
                Path("src/imba_chess/eval/merged_executors.py")
            ),
            compile="whole neural decoder",
            threads=args.threads,
            load_seconds=load_seconds,
            profiled=args.profile,
            timing="synchronized wall-clock boundaries; executor component times are host timings",
            args={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        )
        atomic_json(args.output / "metadata.json", metadata)
        if args.component != "collection":
            from imba_chess.self_play.benchmarks import benchmark_component

            benchmark_component(args, cfg, runtime, seeds, max_positions, actor_id)
            save_compiler_counters()
            return
        summaries = []
        for simulations in map(int, args.simulations.split(",")):
            for candidates in map(int, args.candidates.split(",")):
                previous_rate = None
                for concurrency in map(int, args.concurrency.split(",")):
                    trial_cfg = replace(
                        cfg,
                        search=replace(
                            cfg.search, simulations=simulations, top_m=candidates
                        ),
                        collection=replace(
                            cfg.collection, concurrent_games=concurrency
                        ),
                    )
                    rates = []
                    stop_sweep = False
                    for repeat in range(args.repeats + 1):
                        label = f"s{simulations}-m{candidates}-g{concurrency}-r{repeat}"
                        directory = args.output / label
                        if directory.exists():
                            raise FileExistsError(
                                f"benchmark trial already exists: {directory}"
                            )
                        store = SelfPlayStore(
                            directory / "replay", **asdict(cfg.replay)
                        )
                        if runtime.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(runtime.device)
                        for waves in runtime.waves.values():
                            waves.clear()
                        runtime.seconds.clear()
                        runtime.inference_rows.clear()
                        profiler = cProfile.Profile() if args.profile else None
                        try:
                            synchronize(runtime.device)
                            wall = time.perf_counter()
                            with StopBudget(seconds=args.seconds) as stop:
                                if profiler:
                                    profiler.enable()
                                metrics = collect(
                                    seeds=seeds,
                                    runtime=runtime,
                                    config=trial_cfg,
                                    actor_id=actor_id,
                                    store=store,
                                    max_positions=max_positions,
                                    game_count=args.games,
                                    should_launch=stop.launch,
                                    should_stop=stop.stop,
                                    on_game=lambda game: print(
                                        label,
                                        "game",
                                        game["game_id"],
                                        game["status"],
                                        game["termination"],
                                        len(game.get("moves", [])),
                                        flush=True,
                                    ),
                                )
                                if profiler:
                                    profiler.disable()
                            synchronize(runtime.device)
                            elapsed = time.perf_counter() - wall
                            report = metrics.report()
                            report.update(
                                seconds=elapsed,
                                usable_positions_per_hour=3600
                                * metrics.counts["usable_positions"]
                                / elapsed,
                                searched_positions_per_hour=3600
                                * metrics.counts["searched_positions"]
                                / elapsed,
                                repeat=repeat,
                                warmed=repeat > 0,
                                waves={k: dict(v) for k, v in runtime.waves.items()},
                                executor_host_seconds=dict(runtime.seconds),
                                actual_neural_evaluations=sum(
                                    runtime.inference_rows.values()
                                ),
                                actual_neural_evaluations_per_second=sum(
                                    runtime.inference_rows.values()
                                )
                                / elapsed,
                                peak_host_rss_bytes=resource.getrusage(
                                    resource.RUSAGE_SELF
                                ).ru_maxrss
                                * 1024,
                                peak_vram_bytes=torch.cuda.max_memory_allocated(
                                    runtime.device
                                )
                                if runtime.device.type == "cuda"
                                else None,
                            )
                            atomic_json(directory / "metrics.json", report)
                            if profiler:
                                profiler.dump_stats(str(directory / "profile.pstats"))
                            if repeat:
                                rates.append(report["usable_positions_per_hour"])
                            print(label, json.dumps(report), flush=True)
                            if (
                                runtime.device.type == "cuda"
                                and report["peak_vram_bytes"]
                                > 0.85
                                * torch.cuda.get_device_properties(
                                    runtime.device
                                ).total_memory
                            ):
                                stop_sweep = True
                        except torch.cuda.OutOfMemoryError:
                            atomic_json(
                                directory / "failure.json", dict(reason="CUDA OOM")
                            )
                            torch.cuda.empty_cache()
                            stop_sweep = True
                            break
                    if rates:
                        median = statistics.median(rates)
                        summaries.append(
                            dict(
                                simulations=simulations,
                                candidates=candidates,
                                concurrency=concurrency,
                                median=median,
                                min=min(rates),
                                max=max(rates),
                            )
                        )
                        atomic_json(args.output / "summary.json", summaries)
                        if previous_rate is not None and median < previous_rate * 1.05:
                            stop_sweep = True
                        previous_rate = median
                    if stop_sweep:
                        break
        save_compiler_counters()


if __name__ == "__main__":
    main()
