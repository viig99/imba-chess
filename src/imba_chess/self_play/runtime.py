"""Explicit runtime inputs shared by the self-play commands."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import random
import signal
import os
import threading
import time
import torch

from imba_chess.config import load_repo_config


def load_runtime(config, checkpoint, device, *, stats=None):
    from imba_chess.eval.inference_runtime import load_runtime as load_search_runtime

    random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    return load_search_runtime(
        repo_config=load_repo_config(Path(config.base_config)),
        checkpoint=checkpoint,
        device=device,
        algorithm="gumbel",
        root_batch_tokens=config.collection.root_batch_tokens,
        stats=stats,
    )


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"run directory is already locked: {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class StopBudget:
    def __init__(self, *, seconds, reserve_seconds=0, drain_seconds=0, hard_exit=False):
        if seconds <= 0:
            raise ValueError("time budget must be positive")
        self.hard_exit = hard_exit
        self.timer = None
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.collection_deadline = self.deadline - reserve_seconds
        self.drain_seconds = drain_seconds
        self.interrupted_at = None
        self.previous = {}

    def __enter__(self):
        if self.hard_exit:
            self.timer = threading.Timer(
                max(0, self.deadline - time.monotonic()), lambda: os._exit(124)
            )
            self.timer.daemon = True
            self.timer.start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self._signal)
        return self

    def _signal(self, signum, frame):
        if self.interrupted_at is None:
            self.interrupted_at = time.monotonic()
        else:
            self.deadline = time.monotonic()

    def launch(self):
        return (
            self.interrupted_at is None and time.monotonic() < self.collection_deadline
        )

    def stop_collection(self):
        now = time.monotonic()
        drain_start = (
            self.collection_deadline
            if self.interrupted_at is None
            else min(self.collection_deadline, self.interrupted_at)
        )
        return now >= min(self.deadline, drain_start + self.drain_seconds)

    def stop(self):
        return self.interrupted_at is not None or time.monotonic() >= self.deadline

    def __exit__(self, *args):
        if self.timer is not None:
            self.timer.cancel()
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
