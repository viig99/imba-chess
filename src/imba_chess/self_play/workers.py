"""Multi-process self-play collection sharing one GPU.

One search thread per process is the collection bottleneck (the GPU idles
while Python searches), so several worker processes each play a share of the
concurrent games. The coordinating process keeps everything stateful: streamed
starts, regret state, replay writes, metrics and learning. Workers only play
games: they ask for a start, stream back per-position counters and finished
games, and load the coordinator's current weights at the start of every
phase, so every game in a phase is played by the same, current actor.

Protocol (one duplex pipe per worker; the coordinator only ever replies):
  worker -> coordinator  ("ready", max_positions) | ("launch",) |
                         ("position", counters, elapsed, bucket) |
                         ("game", game_id, game) | ("phase_done",) |
                         ("error", traceback)
  coordinator -> worker  ("phase", job) | ("exit",) | launch replies
                         (game_id, seed, metadata) or None
Any worker failure or exit aborts the phase and raises in the coordinator.
"""

import ctypes
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import torch

from imba_chess.eval.batch_scheduler import BatchScheduler
from .collector import CollectionPhase, launch_game, raise_game_error

POSITION_COUNTERS = ("neural_evaluations", "simulations", "terminal_hits", "depth_cutoffs")


def default_runtime_loader(config, checkpoint, device):
    from .runtime import load_runtime

    return load_runtime(
        config, checkpoint, device, dtype=config.collection.inference_dtype
    )


class WorkerFailed(RuntimeError):
    pass


def split_games(total, workers):
    """Concurrent games per worker, as even as possible."""
    if workers < 1 or total < workers:
        raise ValueError("need at least one concurrent game per collection worker")
    return [total // workers + (i < total % workers) for i in range(workers)]


class CollectionWorkers:
    """Persistent collection processes; use as a context manager.

    runtime_loader(config, checkpoint, device) -> (runtime, max_positions) runs in
    each worker and must be importable by name (processes are spawned). The
    loaded runtime's model receives the coordinator's weights every phase.
    """

    def __init__(
        self,
        *,
        count,
        config,
        checkpoint,
        device,
        directory,
        cpu_threads=1,
        runtime_loader=default_runtime_loader,
        startup_timeout=900.0,
        progress_timeout=300.0,
        shutdown_timeout=5.0,
        should_stop=lambda: False,
    ):
        if count < 2:
            raise ValueError("multi-worker collection needs at least two workers")
        self.count, self.config = count, config
        if any(not 0 < value < float("inf") for value in
               (startup_timeout, progress_timeout, shutdown_timeout)):
            raise ValueError("worker timeouts must be finite and positive")
        self.progress_timeout, self.shutdown_timeout = progress_timeout, shutdown_timeout
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.directory / "weights.pt"
        context = multiprocessing.get_context("spawn")
        self.stop = context.RawValue(ctypes.c_bool, False)
        self.processes, self.connections = [], []
        self._messages = queue.Queue(maxsize=count)
        self._reader_stop = threading.Event()
        self._readers = []
        try:
            for index in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker_main,
                    args=(child, config, str(checkpoint), device, cpu_threads,
                          runtime_loader, self.stop),
                    name=f"self-play-collector-{index}",
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            # Pipe readiness only promises some bytes, not a complete frame.
            # Receive off the coordinator thread so partial messages cannot
            # block startup, progress or stop deadlines. Bound buffered games.
            for index, connection in enumerate(self.connections):
                reader = threading.Thread(target=self._read_messages,
                                          args=(index, connection), daemon=True)
                reader.start()
                self._readers.append(reader)
            # Wait for every worker under one deadline, detecting failures in
            # any process even when an earlier worker is still loading.
            deadline = time.monotonic() + startup_timeout
            pending, limits = set(range(count)), set()
            while pending:
                if should_stop():
                    raise InterruptedError("stopped while loading collection workers")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerFailed(f"collection worker startup timed out: {sorted(pending)}")
                received = self._receive(timeout=min(0.5, remaining))
                if received is None:
                    continue
                index, message = received
                if message[0] == "error":
                    raise WorkerFailed(f"collection worker {index} failed:\n{message[1]}")
                if index not in pending or message[0] != "ready":
                    raise WorkerFailed(f"collection worker {index} sent unexpected startup message")
                limits.add(message[1])
                pending.remove(index)
        except BaseException:
            self.close(force=True)
            raise
        if len(limits) != 1:
            self.close()
            raise WorkerFailed("collection workers disagree on the context limit")
        (self.max_positions,) = limits

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read_messages(self, index, connection):
        while not self._reader_stop.is_set():
            try:
                message, error = connection.recv(), None
            except Exception as exc:
                message, error = None, exc
            while not self._reader_stop.is_set():
                try:
                    self._messages.put((index, message, error), timeout=0.1)
                    break
                except queue.Full:
                    continue
            if error is not None:
                return

    def _receive(self, *, timeout):
        try:
            index, message, error = self._messages.get(timeout=timeout)
        except queue.Empty:
            return None
        if error is not None:
            process = self.processes[index]
            process.join(timeout=min(0.5, self.shutdown_timeout))
            raise WorkerFailed(
                f"collection worker {index} exited (exit code {process.exitcode})"
            ) from error
        return index, message

    def _publish_weights(self, model):
        temporary = self.weights_path.with_suffix(".tmp")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
        os.replace(temporary, self.weights_path)

    def collect(
        self,
        *,
        runtime,
        config,
        actor_id,
        store,
        max_positions,
        start_sampler,
        concurrent_games=None,
        game_count=None,
        should_launch=lambda: True,
        should_stop=lambda: False,
        iteration=0,
        on_game=lambda game: None,
        skip_ids=(),
        metrics=None,
    ):
        """Same contract as collector.collect; runtime supplies the actor weights."""
        if config is not self.config and config.identifier != self.config.identifier:
            raise ValueError("collection workers were started for another configuration")
        if max_positions != self.max_positions:
            raise ValueError("collection workers use a different context limit")
        games = split_games(
            config.collection.concurrent_games if concurrent_games is None else concurrent_games,
            self.count,
        )
        phase = CollectionPhase(
            config=config,
            actor_id=actor_id,
            store=store,
            max_positions=max_positions,
            start_sampler=start_sampler,
            game_count=game_count,
            should_launch=lambda: not self.stop.value and not should_stop() and should_launch(),
            iteration=iteration,
            on_game=on_game,
            skip_ids=skip_ids,
            metrics=metrics,
        )
        try:
            if torch.cuda.is_available():
                # Training's cached activations stay reserved in this process;
                # the workers' search buffers live in other processes.
                torch.cuda.empty_cache()
            self._publish_weights(runtime.model)
            self.stop.value = False
            for connection, count in zip(self.connections, games):
                connection.send(("phase", dict(
                    weights=str(self.weights_path),
                    actor_id=actor_id,
                    config_id=config.identifier,
                    concurrent_games=count,
                )))
            running = set(range(self.count))
            last_progress = dict.fromkeys(running, time.monotonic())
            stop_deadline = None
            while running:
                now = time.monotonic()
                if not self.stop.value and should_stop():
                    self.stop.value = True
                    stop_deadline = now + self.shutdown_timeout
                if stop_deadline is not None and now >= stop_deadline:
                    raise WorkerFailed(f"collection workers did not stop: {sorted(running)}")
                stalled = [i for i in running if now - last_progress[i] >= self.progress_timeout]
                if stalled:
                    raise WorkerFailed(f"collection worker progress timed out: {sorted(stalled)}")
                deadline = min(last_progress[i] + self.progress_timeout for i in running)
                if stop_deadline is not None:
                    deadline = min(deadline, stop_deadline)
                received = self._receive(timeout=min(0.5, max(0.0, deadline - now)))
                if received is not None:
                    index, message = received
                    connection = self.connections[index]
                    kind = message[0]
                    handling_started = time.monotonic()
                    if kind == "launch":
                        connection.send(phase.next_launch())
                    elif kind == "position":
                        counters, elapsed, bucket = message[1:]
                        phase.metrics.position(SimpleNamespace(**counters), elapsed, bucket)
                    elif kind == "game":
                        phase.done(message[1], message[2])
                    elif kind == "phase_done":
                        running.discard(index)
                    elif kind == "error":
                        raise WorkerFailed(f"collection worker {index} failed:\n{message[1]}")
                    else:
                        raise WorkerFailed(f"collection worker {index} sent {kind!r}")
                    # Stream reads and replay publication may block the
                    # coordinator. Workers waiting on it are not stalled.
                    handled = time.monotonic()
                    for other in running:
                        last_progress[other] += handled - handling_started
                    last_progress[index] = handled
        except BaseException as exc:
            # Workers may be mid-phase; there is no partial recovery.
            self.close(force=True)
            phase.abort(exc)
            raise
        finally:
            phase.finish()
        return phase.metrics

    def close(self, *, force=False):
        self.stop.value = True
        self._reader_stop.set()
        if not force:
            for connection in self.connections:
                try:
                    connection.send(("exit",))
                except (BrokenPipeError, OSError):
                    pass
        deadline = time.monotonic() + (0.0 if force else self.shutdown_timeout)
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()  # Workers ignore SIGTERM; the coordinator drains them.
                process.join()
        for connection in self.connections:
            connection.close()
        for reader in self._readers:
            reader.join(timeout=0.5)
        self._readers = []
        self.processes, self.connections = [], []


def _worker_main(connection, config, checkpoint, device, cpu_threads, runtime_loader, stop):
    # The coordinator owns shutdown (its StopBudget drains on SIGINT/SIGTERM and
    # sets `stop`); a service-wide signal must not kill games mid-write.
    try:
        if sys.platform == "linux":
            # Hard coordinator deadlines bypass cleanup. A stuck GPU worker
            # must still die with its parent; SIGTERM is ignored below.
            parent = os.getppid()
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGKILL) != 0:  # PR_SET_PDEATHSIG
                raise OSError(ctypes.get_errno(), "cannot set worker parent-death signal")
            if os.getppid() != parent or parent == 1:
                return
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        torch.set_num_threads(cpu_threads)
        runtime, max_positions = runtime_loader(config, Path(checkpoint), device)
        connection.send(("ready", max_positions))
        while True:
            message = connection.recv()
            if message[0] == "exit":
                return
            job = message[1]
            weights = torch.load(job["weights"], map_location="cpu", weights_only=True)
            runtime.model.load_state_dict(weights)
            del weights
            getattr(runtime, "clear_caches", lambda: None)()
            _play_phase(connection, runtime, config, max_positions, job, stop)
            getattr(runtime, "clear_caches", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Return search memory for training/evaluation.
            connection.send(("phase_done",))
    except (EOFError, BrokenPipeError):
        return  # The coordinator is gone.
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        finally:
            os._exit(1)


def _play_phase(connection, runtime, config, max_positions, job, stop):
    def factory():
        while not stop.value:
            connection.send(("launch",))
            launch = connection.recv()
            if launch is None:
                return
            gid, seed, metadata = launch

            def on_position(result, elapsed, bucket=metadata.get("starting_bucket")):
                counters = {k: getattr(result, k) for k in POSITION_COUNTERS}
                connection.send(("position", counters, elapsed, bucket))

            yield gid, launch_game(
                runtime=runtime,
                config=config,
                config_id=job["config_id"],
                actor_id=job["actor_id"],
                seed=seed,
                game_id=gid,
                max_positions=max_positions,
                should_stop=lambda: stop.value,
                on_position=on_position,
            )

    BatchScheduler(
        game_factory=iter(factory()),
        executors=runtime.executors,
        concurrent_games=job["concurrent_games"],
        on_game_done=lambda gid, game: connection.send(("game", gid, game)),
        on_game_error=raise_game_error,
        completion_order=True,
    ).run()
