"""Manually invoked, bounded collect/train/evaluate loop with restartable phases.

The external nightly schedule is deliberately not installed by this command.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
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
from imba_chess.self_play.trainer import Stage2Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize", type=Path)
    source.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, default=100000)
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
        help="Screen at phase boundaries after this interval instead of every N actors",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        help="Collection slots; execution-only override preserves resume config identity",
    )
    parser.add_argument(
        "--until",
        type=datetime.fromisoformat,
        help="Absolute deadline with timezone; overrides the configured duration",
    )
    parser.add_argument(
        "--screen-every",
        type=int,
        default=1,
        help="Screen every N trained actors; existing evaluation progress is always resumed",
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
    if args.concurrent_games is not None and args.concurrent_games < 1:
        parser.error("--concurrent-games must be positive")
    if args.screen_every < 1:
        parser.error("--screen-every must be positive")
    if args.checkpoint_seconds < 1:
        parser.error("--checkpoint-seconds must be positive")
    if args.keep_recovery_checkpoints < 1:
        parser.error("--keep-recovery-checkpoints must be positive")
    if args.screen_seconds is not None and args.screen_seconds < 1:
        parser.error("--screen-seconds must be positive")
    if args.until is not None and args.until.tzinfo is None:
        parser.error("--until needs an explicit timezone")
    cfg = load_config(args.config)
    seeds = load_seeds(args.seeds)
    monitor = [s for s in seeds if s.split == "monitor"]
    if len(monitor) < max(cfg.run.screen_pairs, cfg.run.confirmation_pairs):
        parser.error(
            "seed manifest needs enough distinct monitoring prefixes for screen and confirmation"
        )
    train_seeds = [s for s in seeds if s.split == "train"]
    if not train_seeds:
        parser.error("seed manifest has no training prefixes")
    seconds = (
        cfg.run.hours * 3600
        if args.until is None
        else args.until.timestamp() - time.time()
    )
    if seconds <= cfg.run.reserve_minutes * 60:
        parser.error("deadline must leave time beyond the configured reserve")
    state_path = args.output / "state.json"
    with (
        run_lock(args.output),
        StopBudget(
            seconds=seconds,
            reserve_seconds=cfg.run.reserve_minutes * 60,
            drain_seconds=cfg.run.drain_minutes * 60,
            hard_exit=True,
        ) as budget,
    ):
        if args.resume:
            state = json.loads(state_path.read_text())
            if state["config_id"] != cfg.identifier or state[
                "seed_manifest"
            ] != file_hash(args.seeds):
                raise ValueError("run configuration or seed manifest changed")
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
        runtime, max_positions = load_runtime(cfg, checkpoint, args.device)
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
            restored = trainer.resume(checkpoint, store=store, config_id=cfg.identifier)
            if (
                restored["iteration"] != state["iteration"]
                or restored["phase"] != state["phase"]
            ):
                raise ValueError("checkpoint and published phase disagree")
        else:
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
                str(Path(state[k]).resolve()) for k in ("actor", "best", "checkpoint")
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
                until=args.until.isoformat() if args.until else None,
                screen_every=args.screen_every,
                checkpoint_seconds=args.checkpoint_seconds,
                keep_recovery_checkpoints=args.keep_recovery_checkpoints,
                screen_seconds=args.screen_seconds,
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
        while state["iteration"] < args.max_iterations and (not budget.stop()):
            if state["phase"] == "collect":
                if not budget.launch():
                    break
                metrics = CollectionMetrics()
                for shard in store.manifest["shards"]:
                    if not (store.directory / shard["file"]).exists():
                        continue
                    for entry in shard["games"]:
                        game = store.read_game(entry["id"])
                        if (
                            game.get("iteration") == state["iteration"]
                            and game["actor_id"] == state["actor_id"]
                        ):
                            metrics.done(game)

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

                metrics = collect(
                    seeds=seeds,
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
                    **{"concurrent_games": args.concurrent_games}
                    if args.concurrent_games is not None
                    else {},
                )
                log(metrics.report())
                if (
                    metrics.counts["training_positions"]
                    < cfg.collection.fresh_positions
                ):
                    publish_checkpoint()
                    break
                trainer.begin_phase(store)
                state.update(
                    phase="train",
                    exposure_budget=int(
                        metrics.counts["training_positions"] * cfg.learning.reuse
                    ),
                )
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
                if trainer.phase_exposures < state["exposure_budget"]:
                    publish_checkpoint()
                    break
                state["phase"] = "evaluate"
                actor = args.output / f"actor-{state['iteration'] + 1:06d}.pt"
                state.update(actor=str(actor), checkpoint=str(actor))
                trainer.checkpoint(
                    actor, progress=dict(state), store=store, config_id=cfg.identifier
                )
                state["actor_id"] = file_hash(actor)
                atomic_json(state_path, state)
            if state["phase"] == "evaluate":
                screen_path = args.output / f"screen-{state['iteration']:06d}.json"
                screen_deferred = (
                    time.monotonic() < next_screen
                    if args.screen_seconds is not None
                    else (state["iteration"] + 1) % args.screen_every != 0
                )
                if screen_deferred and (not screen_path.exists()):
                    log(
                        dict(
                            decision="screen_deferred",
                            screen_every=args.screen_every,
                            screen_seconds=args.screen_seconds,
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
                if screen is None:
                    break
                recommended_action = decision(screen, confirmation)
                action = (
                    "observe_only" if args.observe_only_screen else recommended_action
                )
                log(
                    dict(
                        screen=screen,
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
