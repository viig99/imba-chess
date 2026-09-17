from argparse import Namespace
from dataclasses import replace
import json

import torch

from imba_chess.self_play.benchmarks import benchmark_component
from tests.search_references import InferenceRuntime
from imba_chess.self_play.config import SelfPlayConfig, CollectionConfig
from imba_chess.self_play.seeds import Seed, source_split
from imba_chess.data.self_play_store import SelfPlayStore
from tests.test_self_play import VOCAB, ENCODER, tiny_model, mate_game


def test_component_entrypoints(tmp_path):
    torch.set_num_threads(1)
    cfg = SelfPlayConfig(collection=CollectionConfig(concurrent_games=2))
    cfg = replace(cfg, search=replace(cfg.search, simulations=2, max_depth=2))
    runtime = InferenceRuntime(
        model=tiny_model().eval(),
        move_vocab=VOCAB,
        encoder=ENCODER,
        device=torch.device("cpu"),
    )
    seeds = [
        Seed(str(i), "train-source", [], 0, source_split("train-source"), "c")
        for i in range(2)
    ]
    replay = SelfPlayStore(tmp_path / "replay", flush_games=1)
    replay.add(mate_game())
    manifest = tmp_path / "replay" / "manifest.json"
    original_manifest = manifest.read_bytes()
    original_mtime = manifest.stat().st_mtime_ns
    for component in ("controller", "root", "leaf", "search", "replay", "training"):
        output = tmp_path / component
        args = Namespace(
            component=component,
            games=2,
            repeats=0,
            output=output,
            replay=tmp_path / "replay",
            exposures=4,
            seconds=10,
            simulations="2",
            concurrency="2",
            candidates="2",
        )
        benchmark_component(args, cfg, runtime, seeds, 128, "actor")
        assert manifest.read_bytes() == original_manifest
        assert manifest.stat().st_mtime_ns == original_mtime
        rows = json.loads((output / "components.json").read_text())
        assert rows and all(r["component"] == component for r in rows)
        if component == "controller":
            assert len(rows) == 8
        if component == "search":
            assert rows[0]["positions"] == 2 and not rows[0]["errors"]
        if component == "training":
            assert rows[0]["exposures"] == 4
