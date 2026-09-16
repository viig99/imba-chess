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
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.position_evaluator import load_hstu_checkpoint
from .collector import InferenceRuntime


def load_runtime(
    config,
    checkpoint,
    device,
    *,
    optimized=True,
    one_query_per_game=None,
    cache_prefixes=None,
    decoder_mode=None,
    batch_projection=None,
    batch_inputs=None,
    batch_suffix=None,
    reuse_decode_buffers=None,
    native_gumbel=None,
):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu for correctness probes"
        )
    # Validated CUDA collector path; CPU and explicit reference mode retain
    # the original decoder. Individual overrides support controlled ablations.
    enabled = optimized and device.type == "cuda"
    if decoder_mode is None:
        decoder_mode = "compiled" if enabled else "current"
    if decoder_mode != "current" and config.search.max_depth > 32:
        raise ValueError("tensor decoder workspace supports search depth <= 32")
    options = {
        key: enabled if value is None else value
        for key, value in dict(
            one_query_per_game=one_query_per_game,
            cache_prefixes=cache_prefixes,
            batch_projection=batch_projection,
            batch_inputs=batch_inputs,
            batch_suffix=batch_suffix,
        ).items()
    }
    if reuse_decode_buffers is None:
        # Keep explicit decoder/packing ablations on their requested path.
        reuse_decode_buffers = (
            enabled and decoder_mode in ("tensor", "compiled") and all(options.values())
        )
    if native_gumbel is None:
        native_gumbel = enabled and reuse_decode_buffers
    repo = load_repo_config(Path(config.base_config))
    vocab = MoveVocab.load(repo.vocab.path)
    encoder = BoardStateEncoder(repo.board_state)
    random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    model, _ = load_hstu_checkpoint(
        checkpoint_path=Path(checkpoint),
        repo_config=repo,
        move_vocab=vocab,
        device=device,
        compile_model=False,
        require_value_head=True,
    )
    runtime = InferenceRuntime(
        model=model,
        move_vocab=vocab,
        encoder=encoder,
        device=device,
        root_batch_tokens=config.collection.root_batch_tokens,
        decoder_mode=decoder_mode,
        reuse_decode_buffers=reuse_decode_buffers,
        native_gumbel=native_gumbel,
        **options,
    )
    return runtime, model.config.max_position_embeddings


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
