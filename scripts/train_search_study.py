"""Isolated, resumable collect/train forks for comparing search learning.

Uses the production multi-process collector and Stage2Trainer, with fixed local
human starts instead of a network stream. Never reads Stockfish scores as labels.
Normal self-play runs and their checkpoints are not modified or restarted.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import torch

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.eval.golden import digest, file_sha256
from imba_chess.self_play.collector import CollectionMetrics
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock, StopBudget
from imba_chess.self_play.seeds import load_seeds
from imba_chess.self_play.trainer import Stage2Trainer, atomic_checkpoint
from imba_chess.self_play.workers import CollectionWorkers
from compare_search_methods import source_digest


class FixedStarts:
    """The same phase/launch identity and prefix for all experimental arms."""

    durable_launches = False

    def __init__(self, seeds, games_per_phase, seed):
        self.seeds, self.games_per_phase, self.seed = seeds, games_per_phase, seed
        self.launches = 0

    def begin_phase(self, iteration, actor_id, completed, **kwargs):
        self.launches = 0

    def next_launch(self, iteration, actor_id):
        index = self.launches
        self.launches += 1
        seed = self.seeds[(iteration * self.games_per_phase + index) % len(self.seeds)]
        return seed, digest(["search-learning-study", self.seed, iteration, index, seed.seed_id])

    def launch_metadata(self, gid):
        return {}

    def reconcile(self, completed):
        pass

    def retire(self, gid, reason):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--initialize", type=Path, required=True)
    p.add_argument("--seeds", type=Path, default=Path("artifacts/corpus/v4_self_play_seeds_4096.json"))
    p.add_argument("--golden-positions", type=Path, required=True, help="only source identities are read; no engine labels")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phases", type=int, default=3)
    p.add_argument("--games-per-phase", type=int, default=320)
    p.add_argument("--exposures-per-phase", type=int, default=50000)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--concurrent-games", type=int, default=160)
    p.add_argument("--max-seconds", type=float, default=3600)
    p.add_argument("--no-compile-training", action="store_true")
    args = p.parse_args()
    if min(args.phases, args.games_per_phase, args.exposures_per_phase, args.workers, args.concurrent_games) < 1 or args.max_seconds <= 120:
        p.error("invalid experimental budgets")
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    # Drain each finite collection cohort so all arms train only outcome-labelled
    # games played entirely with one frozen actor. Every other recipe setting stays.
    cfg = replace(cfg, collection=replace(cfg.collection, carry_over_games=False,
                                          concurrent_games=args.concurrent_games))
    golden = json.loads(args.golden_positions.read_text())
    excluded_sources = {g["source_id"] for g in golden.get("games", []) if g["source"] == "human"}
    excluded_game_hashes = {g["game_id"] for g in golden.get("games", [])}
    # The source replay is read-only and used only to resolve parent human IDs.
    replay_source = golden.get("source_files", {}).get("replay")
    if replay_source:
        source = SelfPlayStore(replay_source, read_only=True)
        for g in golden.get("games", []):
            if g["source"] == "self_play" and g["source_id"] in source.index:
                excluded_sources.add(source.read_game(g["source_id"])["source_id"])
    seeds = [s for s in load_seeds(args.seeds, split="train")
             if s.source_id not in excluded_sources]
    random.Random(cfg.run.seed).shuffle(seeds)
    if len(seeds) < args.games_per_phase * args.phases:
        p.error("need distinct non-reference source games for every planned launch")
    identity = dict(schema_version=1, config_id=cfg.identifier, search=asdict(cfg.search),
                    code_sha256=digest([source_digest(), file_sha256(__file__)]),
                    checkpoint_sha256=file_sha256(args.initialize), seeds_sha256=file_sha256(args.seeds),
                    golden_positions_sha256=golden["positions_sha256"],
                    phases=args.phases, games_per_phase=args.games_per_phase,
                    exposures_per_phase=args.exposures_per_phase, workers=args.workers,
                    concurrent_games=args.concurrent_games, training_compiled=not args.no_compile_training,
                    excluded_reference_sources=len(excluded_sources),
                    actual_training_sources=[s.source_id for s in seeds[:args.games_per_phase * args.phases]])
    with run_lock(args.out), StopBudget(seconds=args.max_seconds, reserve_seconds=60, drain_seconds=30) as budget:
        identity_path = args.out / "manifest.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("learning study inputs changed; use another output directory")
        atomic_json(identity_path, identity)
        latest = args.out / "latest-state.pt"
        runtime, maximum = load_runtime(cfg, latest if latest.exists() else args.initialize, "cuda")
        trainer = Stage2Trainer(model=runtime.model, config=cfg.learning, move_vocab=runtime.move_vocab,
                                encoder=runtime.encoder, device="cuda", max_positions=maximum,
                                run_seed=cfg.run.seed, compile_model=not args.no_compile_training)
        store = SelfPlayStore(args.out / "replay", **asdict(cfg.replay))
        if latest.exists():
            state = trainer.resume(latest, store=store, config_id=cfg.identifier)
        else:
            trainer.initialize_optimization(args.initialize)
            state = dict(iteration=0, phase="collect", completed_phases=0)
        sampler = FixedStarts(seeds, args.games_per_phase, cfg.run.seed)
        reports = json.loads((args.out / "summary.json").read_text()) if (args.out / "summary.json").exists() else dict(phases=[])
        with runtime, CollectionWorkers(count=args.workers, config=cfg,
                checkpoint=latest if latest.exists() else args.initialize,
                device="cuda", directory=args.out / "collect-workers", cpu_threads=1) as workers:
            while state["iteration"] < args.phases and budget.launch():
                iteration = state["iteration"]
                entry = state.get("entry", dict(iteration=iteration))
                if state["phase"] == "collect":
                    metrics = CollectionMetrics()
                    started = time.perf_counter()
                    workers.collect(runtime=runtime, config=cfg, actor_id=f"study-{iteration}",
                        store=store, max_positions=maximum, start_sampler=sampler,
                        game_count=args.games_per_phase, skip_ids=store.seen,
                        should_launch=budget.launch, should_stop=budget.stop_collection,
                        iteration=iteration, metrics=metrics,
                        on_game=lambda game: trainer.prepare(game) if game["status"] == "completed" else None)
                    entry.update(collection_seconds=time.perf_counter() - started,
                                 collection=metrics.report())
                    if not budget.launch():
                        break
                    # Assert no complete golden source trajectory slipped into replay.
                    for gid in store.game_ids("train"):
                        game = store.read_game(gid)
                        if game["source_id"] in excluded_sources or digest(game["prefix_moves"] + game["moves"]) in excluded_game_hashes:
                            raise ValueError("reference source game leaked into experimental replay")
                    trainer.begin_phase(store)
                    state.update(phase="train", entry=entry)
                started, first_steps, first_exposures = time.perf_counter(), trainer.steps, trainer.exposures
                with (args.out / "train.jsonl").open("a") as log:
                    def on_step(metrics):
                        log.write(json.dumps(dict(iteration=iteration, **metrics)) + "\n")
                        log.flush()
                    trainer.train(store, exposure_budget=args.exposures_per_phase,
                                  should_stop=budget.stop, on_step=on_step)
                runtime.clear_caches()
                entry["training_seconds"] = entry.get("training_seconds", 0) + time.perf_counter() - started
                entry["optimizer_steps"] = entry.get("optimizer_steps", 0) + trainer.steps - first_steps
                entry["training_exposures"] = entry.get("training_exposures", 0) + trainer.exposures - first_exposures
                complete = trainer.phase_exposures >= args.exposures_per_phase
                if complete:
                    checkpoint = args.out / f"phase-{iteration + 1:03d}.pt"
                    atomic_checkpoint(checkpoint, dict(model=runtime.model.state_dict()))
                    entry.update(checkpoint=str(checkpoint), cumulative_exposures=trainer.exposures,
                                 cumulative_steps=trainer.steps)
                    reports["phases"].append(entry)
                    state = dict(iteration=iteration + 1, phase="collect", completed_phases=iteration + 1)
                trainer.checkpoint(latest, progress=state, store=store, config_id=cfg.identifier)
                atomic_json(args.out / "state.json", state)
                atomic_json(args.out / "summary.json", reports)
                print(json.dumps(dict(event="learning_phase", complete=complete, **entry)), flush=True)
                if not complete:
                    break
        reports.update(complete=state["iteration"] == args.phases,
                       total_training_exposures=trainer.exposures, optimizer_steps=trainer.steps)
        atomic_json(args.out / "summary.json", reports)


if __name__ == "__main__":
    main()
