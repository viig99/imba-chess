"""History-cache ablations against the optimized native CUDA collector."""

import argparse
import cProfile
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import pstats
import random
import statistics
import subprocess
import math
import time
from types import SimpleNamespace

import torch

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval import merged_executors
from imba_chess.self_play.benchmarks import histories
from imba_chess.self_play.collector import CollectionMetrics, collect
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, StopBudget, run_lock
from imba_chess.self_play.seeds import load_seeds, file_hash


def thermal_snapshot():
    fields = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=temperature.gpu,clocks.current.sm,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    cpu = []
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        name = directory / "name"
        if name.exists() and name.read_text().strip() in (
            "coretemp",
            "k10temp",
            "zenpower",
        ):
            sensor = directory / "temp1_input"
            if sensor.exists():
                cpu.append(int(sensor.read_text()) / 1000)
    return dict(
        timestamp=time.time(),
        gpu_celsius=float(fields[0]),
        sm_mhz=float(fields[1]),
        power_watts=float(fields[2]),
        sw_thermal_slowdown=fields[3],
        hw_thermal_slowdown=fields[4],
        cpu_celsius=max(cpu) if cpu else None,
    )


def thermal_ready(snapshot):
    return (
        snapshot["gpu_celsius"] <= 60
        and snapshot["cpu_celsius"] is not None
        and snapshot["cpu_celsius"] <= 70
        and snapshot["sw_thermal_slowdown"] == "Not Active"
        and snapshot["hw_thermal_slowdown"] == "Not Active"
    )


def cool_before_measurement(directory, status):
    """Keep compiled graphs warm; cool hardware outside the timed collection.

    No power, clock, fan or thread settings are changed. Save every sample and
    require a minute of rest plus three ready samples before starting a pass.
    """
    started = time.monotonic()
    samples, consecutive = [], 0
    while True:
        sample = thermal_snapshot()
        samples.append(sample)
        consecutive = consecutive + 1 if thermal_ready(sample) else 0
        elapsed = time.monotonic() - started
        if len(samples) % 6 == 1:
            status("thermal_cooldown", pass_name=directory.name, thermal=sample)
        atomic_json(directory / "cooldown.json", samples)
        if elapsed >= 60 and consecutive >= 3:
            return sample
        if elapsed >= 300:
            raise RuntimeError("Hardware did not reach the common thermal baseline")
        time.sleep(5)


def target_digest(targets):
    return hashlib.sha256(
        json.dumps(sorted(targets, key=lambda r: r["id"]), sort_keys=True).encode()
    ).hexdigest()


def traced(label, fn):
    def call(*args, **kwargs):
        with torch.profiler.record_function(label):
            return fn(*args, **kwargs)

    return call


def preparation_attribution(events):
    """Exclusive CPU/launch work in preparation scopes, excluding explicit waits.

    Ancestor traversal attributes each event once. Device execution and wait APIs
    do not count toward the trigger. This is diagnostic, never promotion timing.
    """
    scopes = {"preparation_ancestor_gather", "preparation_masks_positions"}
    work = 0.0
    wall = 0.0
    for event in events:
        if event.name == "gumbel_searches":
            wall += event.cpu_time_total
        parent = event
        while parent is not None and parent.name not in scopes:
            parent = parent.cpu_parent
        if parent is not None and event.device_type == torch.autograd.DeviceType.CPU:
            name = event.name.lower()
            if not any(
                wait in name for wait in ("synchronize", "waitevent", "streamwait")
            ):
                work += event.self_cpu_time_total
    fraction = work / wall if wall else 0.0
    return dict(
        exclusive_cpu_launch_us=work,
        profiled_wall_us=wall,
        fraction=fraction,
        threshold_met=bool(wall and fraction >= 0.10),
        cache_promotion_required=True,
        note="Attempt fusion only after the cache stage passes all gates.",
    )


def compare_targets(reference, candidate, path="games"):
    """Exact search decisions/counters, tight tolerances for stored float targets."""
    if isinstance(reference, dict):
        assert reference.keys() == candidate.keys(), path
        for key in reference:
            compare_targets(reference[key], candidate[key], f"{path}.{key}")
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(candidate), path
        for i, (a, b) in enumerate(zip(reference, candidate)):
            compare_targets(a, b, f"{path}[{i}]")
    elif isinstance(reference, float):
        assert math.isclose(
            reference, candidate, rel_tol=1e-6, abs_tol=1e-6
        ), f"{path}: {reference!r} != {candidate!r}"
    else:
        assert reference == candidate, path


def compare_workload_targets(references, game_count, targets):
    # Changing the collection size changes the tail's active batches. Compare
    # variants only within the same workload, including the warmup workload.
    assert len(targets) == game_count
    assert len({row["id"] for row in targets}) == game_count
    expected = references.setdefault(game_count, {})
    for row in targets:
        if row["id"] in expected:
            compare_targets(expected[row["id"]], row)
        else:
            expected[row["id"]] = row


def promotion_report(
    reports, games, pairs, reference="reference", candidate="candidate"
):
    paired = [
        (reports[f"pair_{i}_{reference}"], reports[f"pair_{i}_{candidate}"])
        for i in range(pairs)
    ]
    gains = [
        b["usable_positions_per_hour"] / a["usable_positions_per_hour"] - 1
        for a, b in paired
    ]
    latency = [b["move_latency_p95"] / a["move_latency_p95"] - 1 for a, b in paired]
    checks = dict(
        workload=games >= 128 and pairs >= 3,
        median_throughput=statistics.median(gains) >= 0.05,
        every_pair=all(x > 0 for x in gains),
        p95_latency=statistics.median(latency) <= 0.05,
        memory=all(b["peak_allocated_bytes"] < 6 * 1024**3 for _, b in paired),
        warmed_graphs=all(
            r["compiler_before"].get("stats", {}).get("unique_graphs", 0)
            == r["compiler_after"].get("stats", {}).get("unique_graphs", 0)
            for pair in paired
            for r in pair
        ),
        allocated_retention=max(
            r["allocated_after_collection"] for pair in paired for r in pair
        )
        - min(r["allocated_after_collection"] for pair in paired for r in pair)
        <= 1024**2,
        cache_cleanup=all(
            r["cache_empty_after_collection"] for pair in paired for r in pair
        ),
    )
    return dict(
        checks=checks,
        passes=all(checks.values()),
        throughput_gains=gains,
        p95_latency_changes=latency,
        median_throughput_gain=statistics.median(gains),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "seeds", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--reference-targets",
        type=Path,
        help="Compare all passes with a saved targets.json correctness reference",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed measured passes after verifying source and workload identity; repeat warmups",
    )
    parser.add_argument(
        "--thermal-cooldown",
        action="store_true",
        help="Cool before each measured pass; archive temperature/clock state outside timing",
    )
    parser.add_argument("--study", choices=["history", "legacy"], default="history")
    parser.add_argument(
        "--cold-only",
        action="store_true",
        help="One cold complete-game pass; run each variant in a fresh process",
    )
    parser.add_argument(
        "--reference-games",
        type=Path,
        help="Compare complete trajectories with a separate cold process",
    )
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--warmup-games", type=int, default=32)
    parser.add_argument(
        "--include-native",
        action="store_true",
        help="Include native selectors as a third paired variant",
    )
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument(
        "--variant",
        choices=["current", "revision", "direct", "reference", "candidate", "native"],
        default="current",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=0,
        help="Alternating warmed reference/candidate pairs; use 3 with --games 128 for promotion",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Only run unprofiled collector measurements",
    )
    args = parser.parse_args()
    if min(args.games, args.warmup_games, args.concurrency) < 1 or args.pairs < 0:
        parser.error("games/concurrency must be positive and pairs nonnegative")
    if args.study == "history" and (
        args.include_native or args.variant not in ("current", "revision", "direct")
    ):
        parser.error(
            "history study uses current/revision/direct with native selection throughout"
        )
    if args.study == "legacy" and args.variant not in (
        "reference",
        "candidate",
        "native",
    ):
        parser.error("legacy study requires a legacy --variant")
    if args.cold_only and (args.pairs or args.resume or not args.skip_profile):
        parser.error("cold-only requires --skip-profile and no pairs/resume")
    if args.study == "history":
        from torch._inductor import config as inductor_config

        # Compile each process from its own initial workload while preserving
        # reference kernel tuning choices. A cached dynamic graph first compiled
        # at another batch size can have different reduction arithmetic. This
        # affects startup only; warmed calls still reuse their compiled graph.
        inductor_config.fx_graph_cache = False
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config)
    cfg = replace(
        cfg, collection=replace(cfg.collection, concurrent_games=args.concurrency)
    )
    assert (
        cfg.search.simulations == 128
        and cfg.search.top_m == 16
        and cfg.search.max_depth == 32
    )
    seeds = load_seeds(args.seeds)
    actor = file_hash(args.checkpoint)
    with run_lock(args.output):
        previous_metadata = None
        if args.resume:
            previous_metadata = json.loads((args.output / "metadata.json").read_text())
            assert previous_metadata["checkpoint_sha256"] == actor
            assert (
                previous_metadata.get("thermal_cooldown", False)
                == args.thermal_cooldown
            )
            assert previous_metadata["harness_sha256"] == file_hash(Path(__file__))
            assert previous_metadata["seed_manifest_sha256"] == file_hash(args.seeds)
            assert previous_metadata["config"] == asdict(cfg)
            assert previous_metadata["games"] == args.games
            assert previous_metadata["concurrency"] == args.concurrency
            assert previous_metadata.get("study", "legacy") == args.study
            assert previous_metadata.get("include_native", False) == args.include_native
            for field in ("source_hashes", "native_source_hashes"):
                assert all(
                    file_hash(Path(name)) == digest
                    for name, digest in previous_metadata[field].items()
                ), field
            atomic_json(
                args.output / f"metadata.before-resume-{int(time.time())}.json",
                previous_metadata,
            )

        def status(phase, **extra):
            row = dict(phase=phase, timestamp=time.time(), **extra)
            atomic_json(args.output / "status.json", row)
            print(json.dumps(row), flush=True)

        status("loading")
        runtime, limit = load_runtime(
            cfg, args.checkpoint, "cuda", decoder_mode="compiled"
        )
        stats = SimpleNamespace(
            root_eval=0.0,
            search_gpu=0.0,
            decode_prep=0.0,
            decode_project=0.0,
            search_eval_calls=0,
            search_eval_items=0,
        )
        # Existing executor instrumentation measures host scopes, not GPU kernel time.
        root = merged_executors._make_root_eval_executor(
            model=runtime.model,
            device=runtime.device,
            dtype=torch.float32,
            stats=stats,
            max_tokens=cfg.collection.root_batch_tokens,
        )
        modes = (
            ("current", "revision", "direct")
            if args.study == "history"
            else ("reference", "candidate", "native")
        )
        leaves = {
            variant: merged_executors._make_decode_wave_executor(
                model=runtime.model,
                device=runtime.device,
                dtype=torch.float32,
                stats=stats,
                one_query_per_game=True,
                cache_prefixes=True,
                decoder_mode="compiled",
                batch_projection=True,
                batch_inputs=True,
                batch_suffix=True,
                reuse_decode_buffers=variant != "reference",
                history_cache_mode=variant if args.study == "history" else "current",
            )
            for variant in modes
        }

        def select(variant):
            runtime.clear_caches()
            runtime.options["reuse_decode_buffers"] = variant != "reference"
            runtime.options["history_cache_mode"] = (
                variant if args.study == "history" else "current"
            )
            runtime.native_gumbel = args.study == "history" or variant == "native"
            runtime.options["native_gumbel"] = runtime.native_gumbel
            runtime.executors = {
                k: runtime._identified(k, v)
                for k, v in [("root_eval", root), ("decode_wave", leaves[variant])]
            }

        select(args.variant)
        atomic_json(
            args.output / "metadata.json",
            dict(
                config=asdict(cfg),
                checkpoint=str(args.checkpoint),
                checkpoint_sha256=actor,
                seed_manifest_sha256=file_hash(args.seeds),
                games=args.games,
                concurrency=args.concurrency,
                pairs=args.pairs,
                warmup_games=args.warmup_games,
                include_native=args.include_native,
                study=args.study,
                cold_only=args.cold_only,
                thermal_cooldown=args.thermal_cooldown,
                cold_graph_compile=args.study == "history",
                persistent_kernel_tuning=True,
                harness_sha256=file_hash(Path(__file__)),
                variant=args.variant,
                runtime=runtime.options,
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                tf32=False,
                native_source_hashes={
                    str(p): file_hash(p)
                    for p in Path("native/imba_chess_native/src").rglob("*.rs")
                },
                source_hashes={
                    str(p): file_hash(p) for p in Path("src/imba_chess").rglob("*.py")
                },
                note="Whole collector games, original self-play Gumbel noise, independent scratch replay. No production training or replay mutation.",
            ),
        )

        def reset():
            runtime.clear_caches()
            for waves in runtime.waves.values():
                waves.clear()
            runtime.seconds.clear()
            runtime.inference_rows.clear()
            for k in vars(stats):
                setattr(stats, k, 0)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def counters():
            from torch._dynamo.utils import counters as c

            return {k: dict(v) for k, v in c.items()}

        signatures, reports = {}, {}
        reference_targets = {}
        if args.reference_targets:
            saved_targets = json.loads(args.reference_targets.read_text())
            compare_workload_targets(
                reference_targets, len(saved_targets), saved_targets
            )
        bitwise_references = {}
        if args.reference_targets:
            bitwise_references[len(saved_targets)] = target_digest(saved_targets)
        saved_games = (
            json.loads(args.reference_games.read_text())
            if args.reference_games
            else None
        )
        game_counts = {}
        variants = (
            (
                ("reference", "candidate", "native")
                if args.include_native
                else ("reference", "candidate")
            )
            if args.pairs
            else (args.variant,)
        )
        if args.study == "history":
            # Revision-only is an attribution ablation; gate the combined stage.
            variants = ("current", "direct") if args.pairs else (args.variant,)
        passes = [] if args.cold_only else [(f"warmup_{v}", v) for v in variants]
        if args.pairs:
            for i in range(args.pairs):
                order = variants if i % 2 == 0 else variants[::-1]
                passes.extend((f"pair_{i}_{v}", v) for v in order)
        else:
            passes.append(("baseline", args.variant))
        if not args.skip_profile:
            passes.append(("cprofile", args.variant))
        for label, variant in passes:
            pass_games = (
                args.warmup_games if label.startswith("warmup_") else args.games
            )
            game_counts[label] = pass_games
            directory = args.output / label
            if (
                args.resume
                and label.startswith("pair_")
                and all(
                    (directory / name).exists()
                    for name in ("metrics.json", "targets.json", "games.json")
                )
            ):
                report = json.loads((directory / "metrics.json").read_text())
                assert (
                    report["completed_games"] == pass_games
                    and report["variant"] == variant
                )
                assert not report["profiled"]
                targets = json.loads((directory / "targets.json").read_text())
                compare_workload_targets(reference_targets, pass_games, targets)
                reports[label] = report
                signatures[label] = report["game_signature"]
                assert (
                    len(
                        {
                            v
                            for k, v in signatures.items()
                            if game_counts[k] == pass_games
                        }
                    )
                    == 1
                )
                status(label + "_reused", seconds=report["seconds"])
                continue
            select(variant)
            workspace = leaves[variant].workspace
            if workspace is not None:
                workspace.counters.clear()
            status(label)
            directory = args.output / label
            store = SelfPlayStore(directory / "replay", **asdict(cfg.replay))
            reset()
            before = counters()
            profile = cProfile.Profile() if label == "cprofile" else None
            games, targets = [], []

            def done(game):
                targets.append(dict(id=game["game_id"], targets=game["targets"]))
                games.append(
                    dict(
                        id=game["game_id"],
                        status=game["status"],
                        moves=game["moves"],
                        outcome=game.get("outcome_white"),
                    )
                )
                if len(games) % 8 == 0:
                    status(
                        label, games_completed=len(games), games_requested=pass_games
                    )

            thermal_before = None
            if args.thermal_cooldown and not label.startswith("warmup_"):
                thermal_before = cool_before_measurement(directory, status)
                status(label)
            metrics = CollectionMetrics()
            if args.study == "history":
                metrics.latencies = []  # Archive every move in the bounded workload.
            start = time.perf_counter()
            if profile:
                profile.enable()
            with StopBudget(seconds=max(900, pass_games * 60)) as stop:
                metrics = collect(
                    seeds=seeds,
                    runtime=runtime,
                    config=cfg,
                    actor_id=actor,
                    store=store,
                    max_positions=limit,
                    game_count=pass_games,
                    should_launch=stop.launch,
                    should_stop=stop.stop,
                    on_game=done,
                    metrics=metrics,
                )
            torch.cuda.synchronize()
            if profile:
                profile.disable()
            elapsed = time.perf_counter() - start
            thermal_after = thermal_snapshot() if args.thermal_cooldown else None
            assert len(games) == pass_games and all(
                g["status"] == "completed" for g in games
            )
            digest = hashlib.sha256(
                json.dumps(
                    sorted(games, key=lambda g: g["id"]), sort_keys=True
                ).encode()
            ).hexdigest()
            signatures[label] = digest
            targets.sort(key=lambda row: row["id"])
            atomic_json(directory / "targets.json", targets)
            if saved_games is not None and pass_games == len(saved_games):
                assert sorted(games, key=lambda g: g["id"]) == sorted(
                    saved_games, key=lambda g: g["id"]
                )
            bitwise = target_digest(targets)
            expected_bitwise = bitwise_references.setdefault(pass_games, bitwise)
            atomic_json(
                directory / "latencies.json",
                dict(
                    seconds=list(metrics.latencies),
                    scope="all moves"
                    if args.study == "history"
                    else "last up to 4096 moves",
                ),
            )
            report = metrics.report()
            report.update(
                games_per_hour=3600 * len(games) / elapsed,
                usable_positions_per_hour=3600
                * report.get("usable_positions", 0)
                / elapsed,
                variant=variant,
                target_sha256=bitwise,
                thermal_before=thermal_before,
                thermal_after=thermal_after,
                bitwise_targets_equal=bitwise == expected_bitwise,
                single_game_tail=runtime.waves["decode_wave"].get(1, 0) > 0,
                transfers=dict(workspace.counters) if workspace else {},
                cache_empty_after_collection=workspace is None
                or (
                    not workspace.slots
                    and workspace.arena is None
                    and workspace.host is None
                ),
                allocated_after_collection=torch.cuda.memory_allocated(),
            )
            report.update(
                seconds=elapsed,
                executor_host_seconds=dict(runtime.seconds),
                executor_components_host_seconds=vars(stats).copy(),
                waves={k: dict(v) for k, v in runtime.waves.items()},
                inference_rows=dict(runtime.inference_rows),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                compiler_before=before,
                compiler_after=counters(),
                game_signature=digest,
                profiled=profile is not None,
            )
            reports[label] = report
            atomic_json(directory / "metrics.json", report)
            atomic_json(directory / "games.json", games)
            if profile:
                profile.dump_stats(str(directory / "profile.pstats"))
                ps = pstats.Stats(profile)
                rows = [
                    dict(
                        file=f[0],
                        line=f[1],
                        function=f[2],
                        primitive_calls=v[0],
                        calls=v[1],
                        self_seconds=v[2],
                        cumulative_seconds=v[3],
                    )
                    for f, v in ps.stats.items()
                ]
                atomic_json(
                    directory / "functions.json",
                    sorted(rows, key=lambda r: -r["self_seconds"]),
                )
                preparation_seconds = sum(
                    row["cumulative_seconds"]
                    for row in rows
                    if row["file"].endswith("/decode_workspace.py")
                    and row["function"] in ("_gather_ancestors", "_prepare_attention")
                )
                atomic_json(
                    directory / "preparation_attribution.json",
                    dict(
                        host_submission_seconds=preparation_seconds,
                        profiled_wall_seconds=elapsed,
                        fraction=preparation_seconds / elapsed,
                        threshold_met=preparation_seconds / elapsed >= 0.10,
                        note="Host submission upper bound; use the CUDA trace to exclude runtime waits before deciding on fusion.",
                    ),
                )
                search_seconds = sum(
                    row["self_seconds"]
                    for row in rows
                    if row["file"].endswith("/gumbel_search.py")
                )
                atomic_json(
                    directory / "preparation_calls.json",
                    {
                        name: sum(r["calls"] for r in rows if pattern in r["function"])
                        for name, pattern in {
                            "torch_tensor": "<built-in method torch.tensor>",
                            "padding": "<built-in method torch._C._nn.pad>",
                            "cpu_readbacks": "method 'cpu' of",
                            "copy": "method 'copy_' of",
                            "index_select": "<built-in method torch.index_select>",
                            "index_copy": "method 'index_copy_' of",
                        }.items()
                    },
                )
                atomic_json(
                    directory / "search_trigger.json",
                    dict(
                        exclusive_python_search_seconds=search_seconds,
                        profiled_wall_seconds=elapsed,
                        fraction=search_seconds / elapsed,
                        rust_trigger=search_seconds / elapsed >= 0.10,
                    ),
                )
                for sort in ("tottime", "cumtime"):
                    with (directory / f"{sort}.txt").open("w") as stream:
                        pstats.Stats(profile, stream=stream).sort_stats(
                            sort
                        ).print_stats(65)
            compare_workload_targets(reference_targets, pass_games, targets)
            assert (
                len({v for k, v in signatures.items() if game_counts[k] == pass_games})
                == 1
            ), "Game trajectories changed"
            status(
                label + "_complete",
                seconds=elapsed,
                positions=report["searched_positions"],
            )
        # Per-game targets and same-size signatures were checked after every pass.
        if args.pairs:
            atomic_json(
                args.output / "promotion.json",
                dict(
                    cache=promotion_report(reports, args.games, args.pairs),
                    native=promotion_report(
                        reports, args.games, args.pairs, "candidate", "native"
                    ),
                )
                if args.study == "legacy" and args.include_native
                else (
                    dict(
                        direct=promotion_report(
                            reports, args.games, args.pairs, "current", "direct"
                        ),
                        promoted=False,
                        note="Throughput gates only; separate cold-process and updated-weight checks are required before promotion.",
                    )
                    if args.study == "history"
                    else promotion_report(reports, args.games, args.pairs)
                ),
            )
        if args.skip_profile:
            status("complete", signatures_match=True, targets_match=True)
            return
        select(args.variant)
        # Fixed representative prefix lengths for a short, fully warmed CUDA trace.
        ordered = sorted(seeds, key=lambda s: (len(s.prefix_moves), s.seed_id))
        selected = [
            ordered[round(i * (len(ordered) - 1) / max(1, args.concurrency - 1))]
            for i in range(args.concurrency)
        ]
        atomic_json(args.output / "trace_positions.json", [asdict(s) for s in selected])
        prepared = list(histories(selected, runtime))

        def execute(executors):
            results = []

            def jobs():
                for seed, board, history in prepared:
                    yield (
                        seed.seed_id,
                        runtime.search(
                            board=board,
                            history=history,
                            actor_id=actor,
                            game_id=seed.seed_id,
                            config=cfg.search,
                            rng=random.Random(seed.seed_id),
                            should_stop=lambda: False,
                        ),
                    )

            def fail(gid, error):
                raise RuntimeError(gid) from error

            with torch.inference_mode():
                BatchScheduler(
                    game_factory=iter(jobs()),
                    executors=executors,
                    concurrent_games=args.concurrency,
                    completion_order=True,
                    on_game_done=lambda gid, r: results.append(r),
                    on_game_error=fail,
                ).run()
            torch.cuda.synchronize()
            assert len(results) == args.concurrency
            return results

        status("cuda_trace_warmup")
        for _ in range(2):
            reset()
            execute(runtime.executors)
        reset()

        def labeled(kind, fn):
            def call(payloads):
                with torch.profiler.record_function(kind):
                    return fn(payloads)

            return call

        workspace = leaves[args.variant].workspace
        if workspace is not None:
            for method, label in (
                ("_history_stamp", "history_validation"),
                ("_slot", "history_slot_refresh"),
                ("_refresh_direct", "history_direct_refresh"),
                ("_refresh_compact_row", "history_compact_refresh"),
                ("_refresh_current_prefix", "history_decoder_refresh"),
                ("_gather_ancestors", "preparation_ancestor_gather"),
                ("_prepare_attention", "preparation_masks_positions"),
                ("consume", "result_processing"),
            ):
                setattr(workspace, method, traced(label, getattr(workspace, method)))
            workspace.runner.decode = traced(
                "decoder_execution", workspace.runner.decode
            )
        status("cuda_trace")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.profiler.record_function("gumbel_searches"):
                results = execute(
                    {k: labeled(k, v) for k, v in runtime.executors.items()}
                )
        prof.export_chrome_trace(str(args.output / "trace.json"))
        atomic_json(
            args.output / "fusion_trigger.json", preparation_attribution(prof.events())
        )
        events = prof.key_averages()
        atomic_json(
            args.output / "torch_events.json",
            [
                dict(
                    name=e.key,
                    calls=e.count,
                    self_cpu_us=e.self_cpu_time_total,
                    cpu_us=e.cpu_time_total,
                    self_device_us=e.self_device_time_total,
                    device_us=e.device_time_total,
                )
                for e in events
            ],
        )
        (args.output / "torch_cpu.txt").write_text(
            events.table(sort_by="self_cpu_time_total", row_limit=45)
        )
        (args.output / "torch_cuda.txt").write_text(
            events.table(sort_by="self_device_time_total", row_limit=45)
        )
        atomic_json(
            args.output / "trace_summary.json",
            dict(
                searches=len(results),
                simulations=sum(r.simulations for r in results),
                neural_evaluations=sum(r.neural_evaluations for r in results),
                terminal_hits=sum(r.terminal_hits for r in results),
                depth_cutoffs=sum(r.depth_cutoffs for r in results),
                compiler_after=counters(),
            ),
        )
        status("complete", signatures_match=True, targets_match=True)


if __name__ == "__main__":
    main()
