"""Manually invoked, bounded collect/train/evaluate loop with restartable phases.

The external nightly schedule is deliberately not installed by this command.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import time
import torch
from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.self_play.collector import collect, CollectionMetrics
from imba_chess.self_play.config import load_config
from imba_chess.self_play.evaluation import (
    evaluate_pair_checkpoints,
    decision,
    EvaluationProtocolError,
)
from imba_chess.self_play.runtime import load_runtime, run_lock, StopBudget
from imba_chess.self_play.seeds import file_hash, load_seeds
from imba_chess.self_play.streaming import StreamingStarts
from imba_chess.self_play.trainer import Stage2Trainer, atomic_checkpoint
from imba_chess.self_play.workers import CollectionWorkers, split_games

# About 5 iterations (~347 completed games each), i.e. 3 hours, on the 5090 tactical recipe.
DEFAULT_SCREEN_GAMES = 1500

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize", type=Path)
    source.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-optimizer", action="store_true",
        help="Carry the supervised checkpoint optimizer and OneCycleLR schedule into self-play",
    )
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="torch intra-op CPU threads; overrides OMP_NUM_THREADS. Search CPU tensors are "
             "tiny, and idle OpenMP workers spin on every core and slow the single search thread",
    )
    parser.add_argument("--continuous", action="store_true", help="Run until interrupted; no time or default iteration cutoff")
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument(
        "--enable-ema",
        action="store_true",
        help="With --resume, let a run started without EMA continue under the config's "
             "learning.ema_decay (the average starts from the current weights); a no-op "
             "once the run has been upgraded",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=int,
        default=60,
        help="Training recovery-save interval; phase boundaries always save",
    )
    parser.add_argument(
        "--keep-recovery-checkpoints",
        type=int,
        default=1,
        help="Recent recovery checkpoints to retain in addition to actor and best",
    )
    parser.add_argument(
        "--screen-seconds",
        type=int,
        help="Screen at phase boundaries after this interval instead of by completed games",
    )
    parser.add_argument(
        "--screen-games",
        type=int,
        help=f"Screen at the first phase boundary after this many completed games "
             f"(default {DEFAULT_SCREEN_GAMES}, about 3 hours of 5090 tactical Gumbel 512)",
    )
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=1,
        help="Collection processes sharing the GPU; --concurrent-games is split across them. "
             "One search thread per process is the collection bottleneck (execution-only; "
             "preserves resume config identity)",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        help="Collection slots; execution-only override preserves resume config identity",
    )
    parser.add_argument("--worker-startup-timeout", type=float, default=900.0,
                        help="Seconds allowed for collection workers to load (execution-only)")
    parser.add_argument("--worker-progress-timeout", type=float, default=300.0,
                        help="Maximum seconds without worker progress (execution-only)")
    parser.add_argument("--worker-shutdown-timeout", type=float, default=5.0,
                        help="Seconds to stop workers cooperatively before killing them (execution-only)")
    parser.add_argument(
        "--until",
        type=datetime.fromisoformat,
        help="Absolute deadline with timezone; overrides the configured duration",
    )
    parser.add_argument(
        "--defer-confirmation",
        action="store_true",
        help="Keep the evaluated best and defer 500-game promotion confirmation",
    )
    parser.add_argument(
        "--observe-only-screen",
        action="store_true",
        help="Research runs: record strength screens without rollback or best promotion",
    )
    args = parser.parse_args()
    if args.initialize_optimizer and not args.initialize:
        parser.error("--initialize-optimizer requires --initialize")
    if args.concurrent_games is not None and args.concurrent_games < 1:
        parser.error("--concurrent-games must be positive")
    if args.collect_workers < 1:
        parser.error("--collect-workers must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.worker_startup_timeout, args.worker_progress_timeout, args.worker_shutdown_timeout)):
        parser.error("worker timeouts must be finite and positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.checkpoint_seconds < 1:
        parser.error("--checkpoint-seconds must be positive")
    if args.keep_recovery_checkpoints < 1:
        parser.error("--keep-recovery-checkpoints must be positive")
    if args.screen_seconds is not None and args.screen_seconds < 1:
        parser.error("--screen-seconds must be positive")
    if args.screen_games is not None and args.screen_seconds is not None:
        parser.error("--screen-games cannot be combined with --screen-seconds")
    if args.screen_seconds is None and args.screen_games is None:
        args.screen_games = DEFAULT_SCREEN_GAMES
    if args.screen_games is not None and args.screen_games < 1:
        parser.error("--screen-games must be positive")
    if args.until is not None and args.until.tzinfo is None:
        parser.error("--until needs an explicit timezone")
    if args.continuous and args.until is not None:
        parser.error("--continuous cannot be combined with --until")
    torch.set_num_threads(args.cpu_threads)
    cfg = load_config(args.config)
    if args.collect_workers > 1:
        try:
            split_games(args.concurrent_games or cfg.collection.concurrent_games, args.collect_workers)
        except ValueError as exc:
            parser.error(str(exc))
    elif cfg.collection.inference_dtype != "float32":
        # The in-process runtime is also the FP32 training model.
        parser.error("collection.inference_dtype other than float32 needs --collect-workers > 1")
    monitor =load_seeds(args.seeds, split="monitor")
    if len(monitor) < max(cfg.run.screen_pairs, cfg.run.confirmation_pairs):
        parser.error(
            "seed manifest needs enough distinct monitoring prefixes for screen and confirmation"
        )
    seconds = float("inf") if args.continuous else (
        cfg.run.hours * 3600
        if args.until is None
        else args.until.timestamp() - time.time()
    )
    if seconds <= cfg.run.reserve_minutes * 60:
        parser.error("deadline must leave time beyond the configured reserve")
    state_path = args.output / "state.json"
    with (
        run_lock(args.output),
        ExitStack() as resources,
        StopBudget(
            seconds=seconds,
            reserve_seconds=cfg.run.reserve_minutes * 60,
            drain_seconds=cfg.run.drain_minutes * 60,
            hard_exit=not args.continuous,
        ) as budget,
    ):
        saved_config_id = cfg.identifier
        if args.resume:
            state = json.loads(state_path.read_text())
            if args.enable_ema and state["config_id"] != cfg.identifier:
                # The only accepted change is switching EMA on.
                saved_config_id = replace(
                    cfg, learning=replace(cfg.learning, ema_decay=0.0)
                ).identifier
            if state["config_id"] != saved_config_id or state[
                "seed_manifest"
            ] != file_hash(args.seeds):
                raise ValueError("run configuration or seed manifest changed")
            state["config_id"] = cfg.identifier
            if state.get("halted"):
                raise RuntimeError(
                    "automatic learning was halted; investigate the recorded failure before starting another run"
                )
            checkpoint = Path(state["checkpoint"])
        else:
            if state_path.exists():
                raise FileExistsError("run exists; use --resume")
            checkpoint = args.initialize
            state = dict(
                schema_version=1,
                config_id=cfg.identifier,
                seed_manifest=file_hash(args.seeds),
                iteration=0,
                phase="collect",
                halted=False,
            )
        if args.resume and not (args.output / "stream" / "consumer.json").exists():
            raise FileNotFoundError("stream consumer state missing; refusing source rewind")
        start_sampler = resources.enter_context(
            StreamingStarts(
                args.output / "stream", cfg, should_stop=lambda: not budget.launch()
            )
        )
        if args.resume and state.get("stream_identity") != start_sampler.identity:
            raise ValueError("run and stream identities disagree")
        start_sampler.warm()
        state["stream_identity"] = start_sampler.identity
        runtime, max_positions = load_runtime(cfg, checkpoint, args.device)
        workers = None
        if args.collect_workers > 1:
            workers = resources.enter_context(
                CollectionWorkers(
                    count=args.collect_workers,
                    config=cfg,
                    checkpoint=checkpoint,
                    device=args.device,
                    directory=args.output / "collect-workers",
                    cpu_threads=args.cpu_threads,
                    startup_timeout=args.worker_startup_timeout,
                    progress_timeout=args.worker_progress_timeout,
                    shutdown_timeout=args.worker_shutdown_timeout,
                    should_stop=budget.stop,
                )
            )
        if args.resume and state.get("frozen_evaluator_id"):
            raise ValueError("frozen-evaluator runs are no longer supported")
        store = SelfPlayStore(args.output / "replay", **asdict(cfg.replay))
        trainer = Stage2Trainer(
            model=runtime.model,
            config=cfg.learning,
            move_vocab=runtime.move_vocab,
            encoder=runtime.encoder,
            device=runtime.device,
            max_positions=max_positions,
            run_seed=cfg.run.seed,
        )
        if args.resume:
            restored = trainer.resume(
                checkpoint,
                store=store,
                config_id=saved_config_id,
                enable_ema=saved_config_id != cfg.identifier,
            )
            if (
                restored["iteration"] != state["iteration"]
                or restored["phase"] != state["phase"]
            ):
                raise ValueError("checkpoint and published phase disagree")
        else:
            if args.initialize_optimizer:
                trainer.initialize_optimization(checkpoint)
            actor = args.output / "actor-000000.pt"
            state.update(actor=str(actor), best=str(actor), checkpoint=str(actor))
            trainer.checkpoint(
                actor, progress=state, store=store, config_id=cfg.identifier
            )
            state["actor_id"] = file_hash(actor)
            state["best_id"] = state["actor_id"]
            atomic_json(state_path, state)

        def publish_checkpoint():
            path = (
                args.output
                / f"state-{state['iteration']:06d}-{state['phase']}-{trainer.steps:09d}.pt"
            )
            state["checkpoint"] = str(path)
            trainer.checkpoint(
                path, progress=dict(state), store=store, config_id=cfg.identifier
            )
            atomic_json(state_path, state)
            keep = {
                str(Path(state[k]).resolve())
                for k in ("actor", "best", "checkpoint", "ema_actor")
                if k in state
            }
            recovery = sorted(
                args.output.glob("state-*.pt"),
                key=lambda p: (p.stat().st_mtime_ns, p.name),
                reverse=True,
            )
            keep.update(
                (str(p.resolve()) for p in recovery[: args.keep_recovery_checkpoints])
            )
            for old in args.output.glob("*.pt"):
                if str(old.resolve()) not in keep:
                    old.unlink()

        def log(metrics):
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            iteration=state["iteration"],
                            phase=state["phase"],
                            **metrics,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
            print(metrics, flush=True)

        log(
            dict(
                event="run_start",
                concurrent_games=args.concurrent_games
                or cfg.collection.concurrent_games,
                inference_options=getattr(runtime, "options", {}),
                cpu_threads=torch.get_num_threads(),
                collect_workers=args.collect_workers,
                worker_startup_timeout=args.worker_startup_timeout,
                worker_progress_timeout=args.worker_progress_timeout,
                worker_shutdown_timeout=args.worker_shutdown_timeout,
                until=args.until.isoformat() if args.until else None,
                continuous=args.continuous,
                checkpoint_seconds=args.checkpoint_seconds,
                keep_recovery_checkpoints=args.keep_recovery_checkpoints,
                screen_seconds=args.screen_seconds,
                screen_games=args.screen_games,
                defer_confirmation=args.defer_confirmation,
                observe_only_screen=args.observe_only_screen,
            )
        )

        def finish_iteration():
            state.update(iteration=state["iteration"] + 1, phase="collect")
            publish_checkpoint()
            pinned = set(store.active_shards())
            for path in args.output.glob("*.pt"):
                saved = torch.load(path, map_location="cpu", weights_only=False)
                pinned.update(saved.get("replay_shards", []))
                del saved
            store.collect_garbage(pinned_shards=pinned)

        next_screen = time.monotonic() + (args.screen_seconds or 0)
        while (args.continuous or state["iteration"] < args.max_iterations) and (not budget.stop()):
            if state["phase"] == "collect":
                if not budget.launch():
                    break
                metrics = CollectionMetrics()
                for summary in store.phase_summaries(
                    iteration=state["iteration"], actor_id=state["actor_id"]
                ):
                    metrics.done(summary)

                def record_game(game):
                    if game["status"] != "completed":
                        log(
                            dict(
                                game_id=game["game_id"],
                                status=game["status"],
                                reason=game["termination"],
                                searched_positions=game.get("searched_positions", 0),
                                error=game.get("error"),
                            )
                        )

                metrics = (workers.collect if workers else collect)(
                    runtime=runtime,
                    config=cfg,
                    actor_id=state["actor_id"],
                    store=store,
                    max_positions=max_positions,
                    should_launch=budget.launch,
                    should_stop=budget.stop_collection,
                    iteration=state["iteration"],
                    skip_ids=store.seen,
                    metrics=metrics,
                    on_game=record_game,
                    start_sampler=start_sampler,
                    **{"concurrent_games": args.concurrent_games}
                    if args.concurrent_games is not None
                    else {},
                )
                log(metrics.report())
                log(start_sampler.report())
                if (
                    metrics.counts["training_positions"]
                    < cfg.collection.fresh_positions
                ):
                    publish_checkpoint()
                    break
                trainer.begin_phase(store)
                start_sampler.finish_phase()
                # Counted once, persisted with the collect -> train transition.
                state["games_since_screen"] = (
                    state.get("games_since_screen", 0) + metrics.counts["completed_games"]
                )
                exposure_budget = int(
                    metrics.counts["training_positions"] * cfg.learning.reuse
                )
                if cfg.learning.gradient_accumulation > 1:
                    # An accumulated step can overshoot a phase's budget by up
                    # to one step; charge that overshoot to the next phase so
                    # long-run reuse matches the configuration.
                    exposure_budget -= state.get("exposure_carry", 0)
                state.update(phase="train", exposure_budget=exposure_budget)
                publish_checkpoint()
            if state["phase"] == "train":
                last_save = time.monotonic()

                def step(metrics):
                    nonlocal last_save
                    log(metrics)
                    if time.monotonic() - last_save >= args.checkpoint_seconds:
                        publish_checkpoint()
                        last_save = time.monotonic()

                trainer.train(
                    store,
                    exposure_budget=state["exposure_budget"],
                    should_stop=lambda: budget.stop()
                    or (
                        time.monotonic() >= budget.deadline - 60
                        if args.until is not None
                        else not budget.launch()
                    ),
                    on_step=step,
                )
                if torch.cuda.is_available():
                    # Return training activation memory before collection workers restart.
                    torch.cuda.empty_cache()
                if trainer.phase_exposures < state["exposure_budget"]:
                    publish_checkpoint()
                    break
                if cfg.learning.gradient_accumulation > 1:
                    state["exposure_carry"] = (
                        trainer.phase_exposures - state["exposure_budget"]
                    )
                state["phase"] = "evaluate"
                actor = args.output / f"actor-{state['iteration'] + 1:06d}.pt"
                state.update(actor=str(actor), checkpoint=str(actor))
                trainer.checkpoint(
                    actor, progress=dict(state), store=store, config_id=cfg.identifier
                )
                state["actor_id"] = file_hash(actor)
                if trainer.ema is not None:
                    ema_actor = args.output / f"ema-{state['iteration'] + 1:06d}.pt"
                    atomic_checkpoint(ema_actor, dict(model=trainer.ema_state_dict()))
                    state.update(ema_actor=str(ema_actor), ema_actor_id=file_hash(ema_actor))
                atomic_json(state_path, state)
            if state["phase"] == "evaluate":
                screen_path = args.output / f"screen-{state['iteration']:06d}.json"
                screen_deferred = (
                    time.monotonic() < next_screen
                    if args.screen_seconds is not None
                    else state.get("games_since_screen", 0) < args.screen_games
                )
                if screen_deferred and (not screen_path.exists()):
                    log(
                        dict(
                            decision="screen_deferred",
                            screen_seconds=args.screen_seconds,
                            screen_games=args.screen_games,
                            games_since_screen=state.get("games_since_screen", 0),
                        )
                    )
                    finish_iteration()
                    continue
                best, _ = load_runtime(cfg, Path(state["best"]), args.device)
                common = dict(
                    candidate=runtime,
                    best=best,
                    candidate_id=state["actor_id"],
                    best_id=state["best_id"],
                    seeds=monitor,
                    config=cfg,
                    max_positions=max_positions,
                    should_stop=(lambda: budget.stop() or not budget.launch())
                    if args.until
                    else budget.stop,
                )
                try:
                    screen = evaluate_pair_checkpoints(
                        **common, output=screen_path, pairs=cfg.run.screen_pairs
                    )
                    ema_actor = args.output / f"ema-{state['iteration'] + 1:06d}.pt"
                    ema_screen = None
                    if screen is not None and state.get("ema_actor") == str(ema_actor):
                        # Observation only: the EMA never replaces the collecting actor.
                        # Free the raw screen's decode workspace before the EMA builds its own.
                        getattr(runtime, "clear_caches", lambda: None)()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        ema_runtime, _ = load_runtime(cfg, ema_actor, args.device)
                        ema_screen = evaluate_pair_checkpoints(
                            **dict(common, candidate=ema_runtime,
                                   candidate_id=state["ema_actor_id"]),
                            output=args.output / f"screen-ema-{state['iteration']:06d}.json",
                            pairs=cfg.run.screen_pairs,
                        )
                        getattr(ema_runtime, "clear_caches", lambda: None)()
                        del ema_runtime
                        if ema_screen is None:
                            del best, common
                            break
                    confirmation = None
                    if (
                        screen
                        and screen["score"] > 0.5
                        and (screen["upper"] >= 0.45)
                        and (not args.defer_confirmation)
                        and (not args.observe_only_screen)
                    ):
                        confirmation = evaluate_pair_checkpoints(
                            **common,
                            output=args.output
                            / f"confirmation-{state['iteration']:06d}.json",
                            pairs=cfg.run.confirmation_pairs,
                        )
                        if confirmation is None:
                            del best, common
                            break
                except EvaluationProtocolError as exc:
                    state.update(
                        halted=True, halt_reason="evaluation_protocol", error=str(exc)
                    )
                    atomic_json(state_path, state)
                    log(dict(error=str(exc), decision="protocol_stop"))
                    del best, common
                    break
                del best, common
                # Return screen search memory (the decode workspace arena) before
                # collection workers restart, as the workers do after each phase.
                getattr(runtime, "clear_caches", lambda: None)()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if screen is None:
                    break
                recommended_action = decision(screen, confirmation)
                action = (
                    "observe_only" if args.observe_only_screen else recommended_action
                )
                log(
                    dict(
                        screen=screen,
                        **({"ema_screen": ema_screen} if ema_screen is not None else {}),
                        confirmation=confirmation,
                        decision=action,
                        recommended_decision=recommended_action,
                        confirmation_deferred=bool(
                            args.defer_confirmation and screen["score"] > 0.5
                        ),
                    )
                )
                if action == "rollback_stop":
                    state.update(
                        actor=state["best"], actor_id=state["best_id"], halted=True
                    )
                    atomic_json(state_path, state)
                    break
                if action == "promote":
                    state.update(best=state["actor"], best_id=state["actor_id"])
                next_screen = time.monotonic() + (args.screen_seconds or 0)
                state["games_since_screen"] = 0
                finish_iteration()
        print(
            f"Stopped at iteration {state['iteration']}, phase {state['phase']}; "
            + (
                "halted: investigate recorded failure"
                if state.get("halted")
                else "resume with --resume"
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
