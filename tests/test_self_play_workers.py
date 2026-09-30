"""Multi-process collection must match in-process collection and fail loudly."""

import dataclasses
import os

import pytest
import torch

from imba_chess.data.self_play_store import SelfPlayStore
from imba_chess.self_play.collector import CollectionMetrics, collect
from imba_chess.self_play.workers import CollectionWorkers, WorkerFailed, split_games
from tests.test_self_play import ScriptRuntime, mate_starts


class WeightedRuntime(ScriptRuntime):
    """Scripted mate line; root_value reports the weights this process searched with."""

    def __init__(self, executor=lambda payloads: payloads):
        self.executors = {"tick": executor}
        self.model = torch.nn.Linear(2, 1)

    def search(self, **kwargs):
        result = yield from super().search(**kwargs)
        return dataclasses.replace(result, root_value=float(self.model.weight.detach().sum()))


def scripted_loader(config, checkpoint, device):
    return WeightedRuntime(), 128


def _fail(payloads):
    raise RuntimeError("inference failed")


def failing_loader(config, checkpoint, device):
    return WeightedRuntime(_fail), 128


def _die(payloads):
    os._exit(3)


def dying_loader(config, checkpoint, device):
    return WeightedRuntime(_die), 128


def coordinator_runtime(weight):
    runtime = WeightedRuntime()
    with torch.no_grad():
        runtime.model.weight.fill_(weight)
    return runtime


def run_phase(collector, directory, runtime, **kwargs):
    starts, cfg = mate_starts(directory / "stream")
    store = SelfPlayStore(directory / "replay", flush_games=16)
    metrics = collector(
        runtime=runtime,
        config=cfg,
        actor_id="actor",
        store=store,
        max_positions=128,
        start_sampler=starts,
        **kwargs,
    )
    return store, metrics, starts, cfg


def workers_for(tmp_path, cfg, loader=scripted_loader, count=2):
    return CollectionWorkers(
        count=count,
        config=cfg,
        checkpoint=tmp_path / "unused.pt",
        device="cpu",
        directory=tmp_path / "workers",
        runtime_loader=loader,
    )


def test_split_games():
    assert split_games(64, 2) == [32, 32]
    assert split_games(10, 4) == [3, 3, 2, 2]
    with pytest.raises(ValueError):
        split_games(1, 2)


def test_workers_match_in_process_collection_and_follow_weights(tmp_path):
    runtime = coordinator_runtime(0.25)
    single, single_metrics, _, cfg = run_phase(
        collect, tmp_path / "single", runtime, game_count=6, concurrent_games=4
    )
    with workers_for(tmp_path, cfg) as workers:
        parallel, parallel_metrics, _, _ = run_phase(
            workers.collect, tmp_path / "parallel", runtime, game_count=6, concurrent_games=4
        )
        ids = sorted(single.game_ids())
        assert len(ids) == 6 and ids == sorted(parallel.game_ids())
        assert [single.read_game(g) for g in ids] == [parallel.read_game(g) for g in ids]
        for key in ("completed_games", "searched_positions", "neural_evaluations", "simulations"):
            assert parallel_metrics.counts[key] == single_metrics.counts[key] > 0
        assert all(
            t["root_value"] == 0.5 for g in ids for t in parallel.read_game(g)["targets"]
        )

        # A later phase reuses the processes and receives the new actor weights.
        updated, *_ = run_phase(
            workers.collect,
            tmp_path / "next",
            coordinator_runtime(-1.0),
            game_count=2,
            concurrent_games=4,
            iteration=1,
        )
        assert all(
            t["root_value"] == -2.0
            for g in updated.game_ids()
            for t in updated.read_game(g)["targets"]
        )


@pytest.mark.parametrize(
    "loader, message", [(failing_loader, "inference failed"), (dying_loader, "exit code 3")]
)
def test_worker_failure_aborts_the_phase_loudly(tmp_path, loader, message):
    starts, cfg = mate_starts(tmp_path / "stream")
    metrics = CollectionMetrics()
    unfinished = []
    workers = workers_for(tmp_path, cfg, loader)
    with pytest.raises(WorkerFailed, match=message):
        workers.collect(
            runtime=coordinator_runtime(0.0),
            config=cfg,
            actor_id="actor",
            store=SelfPlayStore(tmp_path / "replay"),
            max_positions=128,
            start_sampler=starts,
            game_count=2,
            concurrent_games=2,
            metrics=metrics,
            on_game=unfinished.append,
        )
    assert not workers.processes
    assert metrics.counts["unfinished_games"] == len(unfinished) >= 1
    assert all(game["targets"] == [] and message in game["error"] for game in unfinished)
    # Aborted launches stay journaled for reissue with the same game IDs.
    assert all(game["game_id"] in starts.state["pending"] for game in unfinished)


def test_stop_request_interrupts_worker_games(tmp_path):
    starts, cfg = mate_starts(tmp_path / "stream")
    games = []
    with workers_for(tmp_path, cfg) as workers:
        metrics = workers.collect(
            runtime=coordinator_runtime(0.0),
            config=cfg,
            actor_id="actor",
            store=(store := SelfPlayStore(tmp_path / "replay")),
            max_positions=128,
            start_sampler=starts,
            game_count=4,
            concurrent_games=4,
            should_stop=lambda: True,
            on_game=games.append,
        )
    assert metrics.counts["unfinished_games"] == len(games) >= 1
    assert all(g["termination"] == "interrupted" and g["targets"] == [] for g in games)
    assert not list(store.game_ids())
