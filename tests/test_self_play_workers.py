"""Multi-process collection must match in-process collection and fail loudly."""

import dataclasses
import os
import struct
import time

import pytest
import torch

from imba_chess.data.self_play_store import SelfPlayStore
from imba_chess.self_play.collector import CollectionMetrics, CollectionError, collect
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


def stalled_loader(config, checkpoint, device):
    time.sleep(60)
    return scripted_loader(config, checkpoint, device)


def stalled_executor_loader(config, checkpoint, device):
    def execute(payloads):
        checkpoint.with_suffix(".busy").touch()
        time.sleep(60)
        return payloads
    return WeightedRuntime(execute), 128


class InvalidSearchRuntime(WeightedRuntime):
    def search(self, **kwargs):
        raise ValueError("invalid evaluation WDL")


def invalid_search_loader(config, checkpoint, device):
    return InvalidSearchRuntime(), 128


def partial_message_worker(connection, *args):
    connection.send(("ready", 128))
    connection.recv()
    # Announce a game-sized frame but stall before sending its body.
    os.write(connection.fileno(), struct.pack("!i", 100000))
    time.sleep(60)


def partial_startup_worker(connection, *args):
    os.write(connection.fileno(), struct.pack("!i", 100000))
    time.sleep(60)


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


def workers_for(tmp_path, cfg, loader=scripted_loader, count=2, **kwargs):
    return CollectionWorkers(
        count=count,
        config=cfg,
        checkpoint=tmp_path / "unused.pt",
        device="cpu",
        directory=tmp_path / "workers",
        runtime_loader=loader,
        **kwargs,
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


def test_stop_request_prevents_new_worker_games(tmp_path):
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
            concurrent_games=4,
            should_stop=lambda: True,
            on_game=games.append,
        )
    assert metrics.counts["unfinished_games"] == len(games) == 0
    assert not starts.state["pending"]
    assert not list(store.game_ids())


@pytest.mark.parametrize("stop_requested", [False, True])
def test_partial_message_cannot_block_worker_watchdog(tmp_path, monkeypatch, stop_requested):
    import imba_chess.self_play.workers as worker_module
    monkeypatch.setattr(worker_module, "_worker_main", partial_message_worker)
    starts, cfg = mate_starts(tmp_path / "stream")
    workers = workers_for(tmp_path, cfg, progress_timeout=60 if stop_requested else 0.2,
                          shutdown_timeout=0.2)
    processes = list(workers.processes)
    began = time.monotonic()
    with pytest.raises(WorkerFailed, match="did not stop" if stop_requested else "progress timed out"):
        workers.collect(runtime=coordinator_runtime(0.0), config=cfg, actor_id="actor",
                        store=SelfPlayStore(tmp_path / "replay"), max_positions=128,
                        start_sampler=starts, concurrent_games=2,
                        should_stop=lambda: stop_requested)
    assert time.monotonic() - began < 3
    assert all(not process.is_alive() for process in processes)


def test_partial_startup_message_cannot_block_deadline(tmp_path, monkeypatch):
    import imba_chess.self_play.workers as worker_module
    monkeypatch.setattr(worker_module, "_worker_main", partial_startup_worker)
    _, cfg = mate_starts(tmp_path / "stream")
    with pytest.raises(WorkerFailed, match="startup timed out"):
        workers_for(tmp_path, cfg, startup_timeout=4)


def test_worker_search_validation_errors_abort_before_more_launches(tmp_path):
    starts, cfg = mate_starts(tmp_path / "stream")
    games = []
    workers = workers_for(tmp_path, cfg, invalid_search_loader)
    processes = list(workers.processes)
    store = SelfPlayStore(tmp_path / "replay")
    with pytest.raises(CollectionError, match="invalid evaluation WDL"):
        workers.collect(runtime=coordinator_runtime(0.0), config=cfg, actor_id="actor",
                        store=store, max_positions=128, start_sampler=starts,
                        concurrent_games=2, on_game=games.append)
    assert not workers.processes and all(not p.is_alive() for p in processes)
    assert 1 <= len(games) == len(starts.state["pending"]) <= 2
    assert not store.seen and all(g["targets"] == [] for g in games)


def test_worker_startup_has_a_deadline_and_reaps_children(tmp_path, monkeypatch):
    import multiprocessing
    starts, cfg = mate_starts(tmp_path / "stream")
    processes = []
    process_class = multiprocessing.get_context("spawn").Process
    original_start = process_class.start

    def track(process):
        original_start(process)
        processes.append(process)

    monkeypatch.setattr(process_class, "start", track)
    with pytest.raises(WorkerFailed, match="startup timed out"):
        workers_for(tmp_path, cfg, stalled_loader, startup_timeout=0.2)
    assert len(processes) == 2 and all(not p.is_alive() for p in processes)


@pytest.mark.parametrize("stop_requested", [False, True])
def test_stalled_workers_are_killed_and_launches_remain_retryable(tmp_path, stop_requested):
    starts, cfg = mate_starts(tmp_path / "stream")
    workers = workers_for(tmp_path, cfg, stalled_executor_loader,
                          progress_timeout=60 if stop_requested else 0.5,
                          shutdown_timeout=0.2)
    processes, games = list(workers.processes), []
    store = SelfPlayStore(tmp_path / "replay")
    with pytest.raises(WorkerFailed, match="did not stop" if stop_requested else "progress timed out"):
        workers.collect(runtime=coordinator_runtime(0.0), config=cfg, actor_id="actor",
                        store=store, max_positions=128, start_sampler=starts,
                        concurrent_games=2, on_game=games.append,
                        should_stop=lambda: stop_requested and (tmp_path / "unused.busy").exists())
    assert not workers.processes and all(not p.is_alive() for p in processes)
    assert len(games) == len(starts.state["pending"]) >= 1
    assert not store.seen and all(g["targets"] == [] for g in games)


def test_slow_coordinator_launches_do_not_trip_worker_watchdog(tmp_path):
    starts, cfg = mate_starts(tmp_path / "stream")
    original_launch = starts.next_launch

    def delayed_launch(*args):
        time.sleep(0.6)
        return original_launch(*args)

    starts.next_launch = delayed_launch
    with workers_for(tmp_path, cfg, progress_timeout=0.5) as workers:
        metrics = workers.collect(runtime=coordinator_runtime(0.0), config=cfg, actor_id="actor",
                                  store=SelfPlayStore(tmp_path / "replay"), max_positions=128,
                                  start_sampler=starts, concurrent_games=2, game_count=2)
    assert metrics.counts["completed_games"] == 2
