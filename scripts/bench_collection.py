"""Matched self-play collection benchmark: one production-shaped phase.

A fixed checkpoint, seed manifest and config play one collection phase in the
production multi-worker collector (launch until fresh_positions completed
positions, then drain). A short warm-up phase first pays compilation in the
persistent workers. Each worker reports wall-clock stage timers; the
coordinator reports throughput, drain and GPU samples. Optionally py-spy
samples every worker during the measured phase.

Stage timers per worker (host wall time, no added synchronization):
  root_eval       merged root forwards (prepare, launch, readback)
  decode_prep     DecodeWorkspace.prepare (staging, H2D, ancestor gather launch)
  decode_launch   compiled decoder call (host launch time only)
  decode_consume  KV scatter, legal-logit gather, blocking readback (includes
                  waiting for the GPU) and CPU normalization into PositionEvals
  advance         everything else: scheduler, generators, tree search, native
                  selection/backup, game bookkeeping and IPC to the coordinator
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

from imba_chess.self_play.collector import CollectionMetrics
from imba_chess.self_play.config import load_config
from imba_chess.self_play.seeds import load_seeds
from imba_chess.self_play.workers import CollectionWorkers

STATS_DIR_ENV = "IMBA_BENCH_STATS_DIR"


class StageStats:
    root_eval = search_gpu = decode_prep = decode_project = 0.0
    search_eval_calls = search_eval_items = 0


def instrumented_runtime_loader(config, checkpoint, device):
    from imba_chess.self_play.runtime import load_runtime

    stats = StageStats()
    runtime, max_positions = load_runtime(
        config, checkpoint, device, stats=stats, dtype=config.collection.inference_dtype
    )
    directory = Path(os.environ[STATS_DIR_ENV])
    clear = runtime.clear_caches
    marks = []

    def clear_caches():
        # Workers clear before and after every phase; the after-phase call
        # closes the phase window.
        clear()
        now = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        marks.append(now)
        workspace = runtime.executors["decode_wave"].workspace
        snapshot = dict(
            pid=os.getpid(),
            marks=list(marks),
            root_eval=stats.root_eval,
            decode_prep=stats.decode_prep,
            decode_launch=stats.search_gpu,
            decode_consume=stats.decode_project,
            decode_calls=stats.search_eval_calls,
            decode_rows=stats.search_eval_items,
            executor_seconds=dict(runtime.seconds),
            waves={k: {str(n): c for n, c in v.items()} for k, v in runtime.waves.items()},
            inference_rows=dict(runtime.inference_rows),
            workspace_counters=dict(workspace.counters) if workspace else {},
            user_cpu=usage.ru_utime,
            system_cpu=usage.ru_stime,
            max_rss_kb=usage.ru_maxrss,
            cuda_max_reserved=torch.cuda.max_memory_reserved(),
        )
        path = directory / f"worker-{os.getpid()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(snapshot) + "\n")

    runtime.clear_caches = clear_caches
    return runtime, max_positions


class TimedMetrics(CollectionMetrics):
    def __init__(self):
        super().__init__()
        self.position_events = []
        self.game_events = []
        self.launch_stopped = None

    def position(self, result, elapsed, bucket=None):
        super().position(result, elapsed, bucket)
        self.position_events.append(
            (time.perf_counter(), result.neural_evaluations, result.simulations)
        )

    def done(self, game):
        super().done(game)
        self.game_events.append(
            (time.perf_counter(), game["status"], len(game.get("moves", ())))
        )


class NullStore:
    def __init__(self):
        self.seen = set()

    def add(self, game):
        pass

    def flush(self):
        pass


class CyclingStarts:
    """Deterministic starts: manifest seeds in order under fixed game IDs."""

    def __init__(self, seeds, prefix):
        self.seeds, self.prefix, self.launches = seeds, prefix, 0

    def begin_phase(self, iteration, actor_id, completed):
        pass

    def next_launch(self, iteration, actor_id):
        seed = self.seeds[self.launches % len(self.seeds)]
        gid = f"{self.prefix}-{self.launches:06d}"
        self.launches += 1
        return seed, gid

    def launch_metadata(self, gid):
        return {}

    def retire(self, gid, reason):
        pass

    def reconcile(self, completed):
        pass


class WeightsOnly:
    """The coordinator only publishes weights; no GPU runtime is needed here."""

    def __init__(self, state_dict):
        self.model = self
        self._state = state_dict

    def state_dict(self):
        return self._state


def load_model_state(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    raise KeyError(f"no model state in {checkpoint}: {list(payload)[:10]}")


def summarize_phase(metrics, start, end, games_concurrent):
    events = metrics.position_events
    games = metrics.game_events
    report = metrics.report()
    stop = metrics.launch_stopped or end
    in_window = [e for e in events if e[0] <= stop]
    summary = dict(
        wall=end - start,
        launch_window=stop - start,
        drain=end - stop,
        searched_positions=len(events),
        neural_evaluations=sum(e[1] for e in events),
        simulations=sum(e[2] for e in events),
        completed_games=sum(1 for g in games if g[1] == "completed"),
        finished_games=len(games),
        completed_positions=report.get("completed_positions", 0),
        usable_positions=report.get("usable_positions", 0),
        mean_completed_plies=(
            sum(g[2] for g in games if g[1] == "completed")
            / max(1, sum(1 for g in games if g[1] == "completed"))
        ),
        terminations=report["terminations"],
        decided_positions=report.get("decided_search_positions", 0),
        full_positions=report.get("full_search_positions", 0),
        move_latency_p50=report["move_latency_p50"],
        move_latency_p95=report["move_latency_p95"],
    )
    summary["usable_positions_per_hour"] = 3600 * summary["usable_positions"] / summary["wall"]
    summary["searched_positions_per_hour"] = 3600 * len(events) / summary["wall"]
    summary["evals_per_second"] = summary["neural_evaluations"] / summary["wall"]
    summary["window_evals_per_second"] = sum(e[1] for e in in_window) / max(
        1e-9, stop - start
    )
    summary["drain_evals_per_second"] = sum(
        e[1] for e in events if e[0] > stop
    ) / max(1e-9, end - stop)
    # Throughput over time, 15 s bins, to show ramp-up and drain.
    bins = Counter()
    for t, evals, _ in events:
        bins[int((t - start) // 15)] += evals
    summary["evals_per_second_15s_bins"] = [
        round(bins[i] / 15) for i in range(int((end - start) // 15) + 1)
    ]
    return summary


def worker_stage_report(directory):
    workers = []
    for path in sorted(Path(directory).glob("worker-*.jsonl")):
        snapshots = [json.loads(line) for line in path.read_text().splitlines()]
        # Snapshots: [warm-up start, warm-up end, measured start, measured end].
        before, after = snapshots[-2], snapshots[-1]
        wall = after["marks"][-1] - before["marks"][-1]
        stages = {
            key: after[key] - before[key]
            for key in ("root_eval", "decode_prep", "decode_launch", "decode_consume")
        }
        stages["advance"] = wall - sum(stages.values())
        rows = {
            k: after["inference_rows"].get(k, 0) - before["inference_rows"].get(k, 0)
            for k in after["inference_rows"]
        }
        waves = Counter()
        for size, count in after["waves"].get("decode_wave", {}).items():
            waves[int(size)] += count
        for size, count in before["waves"].get("decode_wave", {}).items():
            waves[int(size)] -= count
        calls = after["decode_calls"] - before["decode_calls"]
        root_waves = Counter()
        for size, count in after["waves"].get("root_eval", {}).items():
            root_waves[int(size)] += count
        for size, count in before["waves"].get("root_eval", {}).items():
            root_waves[int(size)] -= count
        counters = {
            k: v - before["workspace_counters"].get(k, 0)
            for k, v in after["workspace_counters"].items()
        }
        workers.append(
            dict(
                pid=after["pid"],
                wall=wall,
                stages=stages,
                stage_share={k: v / wall for k, v in stages.items()},
                cpu=(after["user_cpu"] + after["system_cpu"])
                - (before["user_cpu"] + before["system_cpu"]),
                max_rss_gb=after["max_rss_kb"] / 1e6,
                cuda_max_reserved_gb=after["cuda_max_reserved"] / 1e9,
                decode_waves=calls,
                decode_rows=rows.get("decode_wave", 0),
                root_rows=rows.get("root_eval", 0),
                root_calls=sum(root_waves.values()),
                mean_decode_batch=rows.get("decode_wave", 0) / max(1, calls),
                decode_batch_histogram={k: v for k, v in sorted(waves.items()) if v},
                workspace_counters=counters,
            )
        )
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--concurrent-games", type=int, default=160)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=60.0)
    parser.add_argument(
        "--fresh-positions", type=int,
        help="Completed positions before launching stops (default: the config's)",
    )
    parser.add_argument("--max-seconds", type=float, default=1800.0,
                        help="Hard cap on the measured phase; stops and interrupts games")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--py-spy", type=Path, help="py-spy binary; samples every worker")
    parser.add_argument("--py-spy-seconds", type=float, default=120.0)
    parser.add_argument("--py-spy-delay", type=float, default=30.0,
                        help="Seconds into the measured phase before sampling starts")
    parser.add_argument("--py-spy-rate", type=int, default=50)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    cfg = load_config(args.config)
    if args.fresh_positions is not None:
        cfg = replace(
            cfg, collection=replace(cfg.collection, fresh_positions=args.fresh_positions)
        )
    if cfg.regret is not None:
        raise ValueError("benchmark supports configs without regret restarts")
    seeds = load_seeds(args.seeds, split="train")
    seeds = seeds[args.seed_offset :] + seeds[: args.seed_offset]
    args.output.mkdir(parents=True, exist_ok=False)
    stats_dir = args.output / "worker-stats"
    stats_dir.mkdir()
    os.environ[STATS_DIR_ENV] = str(stats_dir.resolve())
    runtime = WeightsOnly(load_model_state(args.checkpoint))

    gpu_log = (args.output / "gpu.csv").open("w")
    gpu = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=gpu_log,
    )
    loaded = time.perf_counter()
    try:
        with CollectionWorkers(
            count=args.workers,
            config=cfg,
            checkpoint=args.checkpoint,
            device="cuda",
            directory=args.output / "collect-workers",
            cpu_threads=args.cpu_threads,
            runtime_loader=instrumented_runtime_loader,
        ) as workers:
            startup = time.perf_counter() - loaded
            pids = [p.pid for p in workers.processes]
            # Warm-up: compile and grow buffers, then interrupt every game.
            warm_start = time.perf_counter()
            workers.collect(
                runtime=runtime, config=cfg, actor_id="bench-warmup",
                store=NullStore(), max_positions=workers.max_positions,
                start_sampler=CyclingStarts(seeds[::-1], "warmup"),
                concurrent_games=args.concurrent_games,
                should_launch=lambda: time.perf_counter() - warm_start < args.warmup_seconds,
                should_stop=lambda: time.perf_counter() - warm_start >= args.warmup_seconds,
            )
            warmup = time.perf_counter() - warm_start

            metrics = TimedMetrics()
            spies = []
            start = time.perf_counter()

            def should_launch():
                now = time.perf_counter()
                if args.py_spy is not None and not spies and now - start >= args.py_spy_delay:
                    for index, pid in enumerate(pids):
                        spies.append(subprocess.Popen(
                            [str(args.py_spy), "record", "--pid", str(pid), "--native",
                             "--rate", str(args.py_spy_rate), "--duration",
                             str(int(args.py_spy_seconds)), "--format", "raw",
                             "-o", str(args.output / f"py-spy-worker{index}.txt")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        ))
                return True

            def should_stop():
                return time.perf_counter() - start >= args.max_seconds

            original_launch = metrics.launched

            # The phase calls should_launch for every launch; it stops on its own
            # once fresh_positions completed positions exist.
            def launched(metadata):
                original_launch(metadata)

            metrics.launched = launched
            from imba_chess.self_play import collector as collector_module

            phase_next = collector_module.CollectionPhase.next_launch

            def next_launch(self):
                launch = phase_next(self)
                if launch is None and metrics.launch_stopped is None:
                    metrics.launch_stopped = time.perf_counter()
                return launch

            collector_module.CollectionPhase.next_launch = next_launch
            workers.collect(
                runtime=runtime, config=cfg, actor_id="bench",
                store=NullStore(), max_positions=workers.max_positions,
                start_sampler=CyclingStarts(seeds, "bench"),
                concurrent_games=args.concurrent_games,
                should_launch=should_launch,
                should_stop=should_stop,
                metrics=metrics,
            )
            end = time.perf_counter()
            collector_module.CollectionPhase.next_launch = phase_next
            for spy in spies:
                spy.wait()
    finally:
        gpu.terminate()
        gpu.wait()
        gpu_log.close()

    summary = summarize_phase(metrics, start, end, args.concurrent_games)
    worker_reports = worker_stage_report(stats_dir)
    gpu_rows = []
    for line in (args.output / "gpu.csv").read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpu_rows.append((float(parts[1]), float(parts[2])))
    # The GPU log spans warm-up too; report the samples from the measured phase.
    measured = gpu_rows[-max(1, int((end - start) * 2)) :]
    result = dict(
        args={k: str(v) for k, v in vars(args).items()},
        git=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip(),
        startup_seconds=startup,
        warmup_seconds=warmup,
        phase=summary,
        gpu_util_mean=sum(r[0] for r in measured) / max(1, len(measured)),
        gpu_memory_max_mb=max((r[1] for r in gpu_rows), default=0),
        workers=worker_reports,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    totals = Counter()
    for worker in worker_reports:
        for key, value in worker["stages"].items():
            totals[key] += value
    wall = sum(w["wall"] for w in worker_reports)
    print(json.dumps({k: v for k, v in summary.items() if k != "evals_per_second_15s_bins"},
                     indent=1))
    print("gpu util mean", round(result["gpu_util_mean"], 1))
    print("stage share of worker wall:",
          {k: round(v / wall, 3) for k, v in totals.items()})
    print("worker cpu/wall:", [round(w["cpu"] / w["wall"], 2) for w in worker_reports])
    print("mean decode batch:", [round(w["mean_decode_batch"], 1) for w in worker_reports])


if __name__ == "__main__":
    main()
