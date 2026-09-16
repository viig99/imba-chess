"""Run history-cache correctness prerequisites, ablations and promotion gates.

All replay and compiler artifacts stay under a new output directory. Production
weights are read only. Cold variants always execute in separate processes.
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.self_play.collector import collect
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import file_hash, load_seeds
from imba_chess.self_play.trainer import Stage2Trainer
from scripts.profile_gumbel_pipeline import compare_targets, target_digest


def update_check(args):
    """Compare current/direct before and after one real scratch optimizer update."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config)
    cfg = replace(cfg, collection=replace(cfg.collection, concurrent_games=24))
    actor = file_hash(args.checkpoint)
    reference, limit = load_runtime(
        cfg, args.checkpoint, "cuda", history_cache_mode="current"
    )
    # Share identical updated weights, with separate executor ownership/buffers.
    from imba_chess.self_play.collector import InferenceRuntime

    candidate = InferenceRuntime(
        model=reference.model,
        move_vocab=reference.move_vocab,
        encoder=reference.encoder,
        device=reference.device,
        root_batch_tokens=cfg.collection.root_batch_tokens,
        one_query_per_game=True,
        cache_prefixes=True,
        decoder_mode="compiled",
        batch_projection=True,
        batch_inputs=True,
        batch_suffix=True,
        reuse_decode_buffers=True,
        native_gumbel=True,
        history_cache_mode="direct",
    )
    seeds = load_seeds(args.seeds)
    output = args.output / "update"

    def run(label, runtime):
        games = []
        store = SelfPlayStore(output / label / "replay", **asdict(cfg.replay))
        metrics = collect(
            seeds=seeds,
            runtime=runtime,
            config=cfg,
            actor_id=actor,
            store=store,
            max_positions=limit,
            game_count=4,
            on_game=games.append,
        )
        assert len(games) == 4 and all(g["status"] == "completed" for g in games)
        games.sort(key=lambda g: g["game_id"])
        atomic_json(output / label / "games.json", games)
        atomic_json(output / label / "metrics.json", metrics.report())
        return store, games

    def compare(a, b):
        for x, y in zip(a, b):
            for key in ("game_id", "status", "moves", "outcome_white"):
                assert x[key] == y[key], key
            compare_targets(x["targets"], y["targets"])
        return target_digest(
            [dict(id=g["game_id"], targets=g["targets"]) for g in a]
        ) == target_digest([dict(id=g["game_id"], targets=g["targets"]) for g in b])

    store, before = run("before_current", reference)
    _, before_candidate = run("before_direct", candidate)
    before_bitwise = compare(before, before_candidate)
    old = reference.model.prediction_head.weight.detach().clone()
    trainer = Stage2Trainer(
        model=reference.model,
        config=cfg.learning,
        move_vocab=reference.move_vocab,
        encoder=reference.encoder,
        device=reference.device,
        max_positions=limit,
        run_seed=cfg.run.seed,
    )
    reference.clear_caches()
    candidate.clear_caches()
    trainer.begin_phase(store)
    trainer.train(store, exposure_budget=1)
    delta = (old - reference.model.prediction_head.weight).abs().max().item()
    assert delta > 0 and trainer.steps > 0
    reference.model.eval()
    _, after = run("after_current", reference)
    _, after_candidate = run("after_direct", candidate)
    after_bitwise = compare(after, after_candidate)
    assert file_hash(args.checkpoint) == actor
    atomic_json(
        output / "summary.json",
        dict(
            passes=True,
            before_bitwise=before_bitwise,
            after_bitwise=after_bitwise,
            head_max_weight_change=delta,
            optimizer_steps=trainer.steps,
            checkpoint_sha256=actor,
            checkpoint_unchanged=True,
            config=asdict(cfg),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "seeds", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--update-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; no performance or promotion gates can run")
    if args.update_only:
        update_check(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    common = [
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--seeds",
        str(args.seeds),
    ]
    commands = []

    def run(script, options, *, cache="inductor_cache"):
        command = [sys.executable, script, *common, *options]
        commands.append(command)
        atomic_json(args.output / "commands.json", commands)
        env = os.environ.copy()
        # Every mode uses the same persisted kernel tuning decisions. Cold
        # passes disable the graph cache, not arithmetic-affecting tuning caches.
        env["TORCHINDUCTOR_CACHE_DIR"] = str((args.output / cache).resolve())
        subprocess.run(command, check=True, env=env)

    atomic_json(
        args.output / "identity.json",
        dict(
            config_sha256=file_hash(args.config),
            checkpoint_sha256=file_hash(args.checkpoint),
            seeds_sha256=file_hash(args.seeds),
            source_hashes={
                str(p): file_hash(p)
                for root, pattern in (
                    ("src/imba_chess", "*.py"),
                    ("native/imba_chess_native/src", "*.rs"),
                    ("scripts", "*history_cache.py"),
                    ("scripts", "profile_gumbel_pipeline.py"),
                )
                for p in Path(root).rglob(pattern)
            },
        ),
    )
    profile = "scripts/profile_gumbel_pipeline.py"
    for mode in ("current", "revision", "direct"):
        label = "cold_" + mode
        options = [
            "--output",
            str(args.output / label),
            "--variant",
            mode,
            "--cold-only",
            "--skip-profile",
            "--games",
            "32",
        ]
        if mode != "current":
            options += [
                "--reference-targets",
                str(args.output / "cold_current/baseline/targets.json"),
                "--reference-games",
                str(args.output / "cold_current/baseline/games.json"),
            ]
        run(profile, options)
        metrics = json.loads(
            (args.output / label / "baseline/metrics.json").read_text()
        )
        assert metrics[
            "single_game_tail"
        ], "cold check did not exercise a single-game tail"
    run(
        __file__,
        ["--output", str(args.output), "--update-only"],
        cache="update_inductor_cache",
    )
    for mode in ("current", "revision", "direct"):
        run(
            profile,
            [
                "--output",
                str(args.output / ("profile_" + mode)),
                "--variant",
                mode,
                "--games",
                "32",
                "--warmup-games",
                "32",
                "--reference-targets",
                str(args.output / "cold_current/baseline/targets.json"),
                "--reference-games",
                str(args.output / "cold_current/baseline/games.json"),
            ],
        )
    run(
        profile,
        [
            "--output",
            str(args.output / "promotion"),
            "--games",
            "128",
            "--warmup-games",
            "32",
            "--pairs",
            "3",
            "--thermal-cooldown",
            "--skip-profile",
        ],
    )
    gates = json.loads((args.output / "promotion/promotion.json").read_text())
    fusion = json.loads(
        (args.output / "profile_direct/fusion_trigger.json").read_text()
    )
    attribution = json.loads(
        (
            args.output / "profile_direct/cprofile/preparation_attribution.json"
        ).read_text()
    )
    fusion_trigger = fusion["threshold_met"] and attribution["threshold_met"]
    atomic_json(
        args.output / "decision.json",
        dict(
            cache_eligible_for_promotion=gates["direct"]["passes"],
            cold_and_updated_weight_checks=True,
            fusion_trigger_met=fusion_trigger,
            attempt_fusion=gates["direct"]["passes"] and fusion_trigger,
            default_changed=False,
        ),
    )


if __name__ == "__main__":
    main()
