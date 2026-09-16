"""Check optimized versus compatibility decoding before/after a scratch update.

Production weights are read only. Historical cache ablations live in Git history.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
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
    """Compare compatibility/direct before and after one real scratch optimizer update."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config)
    cfg = replace(cfg, collection=replace(cfg.collection, concurrent_games=24))
    actor = file_hash(args.checkpoint)
    reference, limit = load_runtime(
        cfg, args.checkpoint, "cuda", reuse_decode_buffers=False, native_gumbel=True
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

    store, before = run("before_compatibility", reference)
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
    _, after = run("after_compatibility", reference)
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; the scratch update check requires CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    update_check(args)


if __name__ == "__main__":
    main()
