"""Durable, prefetched starts using the supervised training game's input path.

HF does not checkpoint its shuffle buffer. We therefore checkpoint the filtered
UNSHUFFLED source cursor after each immutable block, and shuffle that block with
a deterministic seed. The collector journals a launch before using it; replay
publication acknowledges completed launches. Crashed/in-flight launches can be
reissued with identical game IDs, prefixes and exploration RNG.
"""

from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from imba_chess.config import load_repo_config
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.seeds import Seed, source_split, stable_hash

BUCKETS = ((0, 0), (1, 30), (31, 70), (71, 120))


def dataset_settings(base_config):
    cfg = asdict(load_repo_config(base_config).dataset)
    settings = {
        k: v
        for k, v in cfg.items()
        if k in inspect.signature(LichessDataset).parameters
    }
    settings.update(
        split="train", train_shuffle_buffer_size=0, parse_stockfish_evals=False
    )
    return settings


def open_source(settings):
    """Return a checkpointable source; share filtering for remote and local data."""
    dataset = LichessDataset(**settings)
    if settings.get("local_corpus_path"):
        # Local corpora are useful for offline verification; reapply the same
        # filters rather than assuming an arbitrary local file was filtered.
        from datasets import load_dataset

        rows = load_dataset(
            "parquet",
            data_files=settings["local_corpus_path"],
            split="train",
            streaming=True,
        )
    else:
        rows, _ = dataset.filtered_shuffled_rows(filter_rows=False)
        if rows is None:
            raise RuntimeError("streaming starts require a checkpointable source")
    # HF's filtered Arrow iterator in the installed version resumes at the
    # wrong row inside a record batch. Checkpoint its raw iterator instead;
    # apply the exact existing predicate before parsing/shuffling as usual.
    return dataset, FilteredSource(rows, dataset._game_filter)


class FilteredSource:
    def __init__(self, rows, predicate):
        self.rows, self.predicate = rows, predicate

    def __iter__(self):
        for row in self.rows:
            if self.predicate(row):
                yield row

    def state_dict(self):
        return self.rows.state_dict()

    def load_state_dict(self, state):
        self.rows.load_state_dict(state)


def make_block(dataset, rows, *, identity, number, run_seed):
    """One seed per accepted source, balanced by requesting a bucket first."""
    shuffled = list(rows)
    rng = random.Random(f"{run_seed}:block:{number}")
    rng.shuffle(shuffled)
    groups = [[], [], []]
    seen = set()
    requested = 0
    for game in dataset.stream_from_rows(shuffled, assume_prefiltered=True):
        source = game["game_id"]
        if not source or source in seen or source_split(source) != "train":
            continue
        seen.add(source)
        lower, upper = BUCKETS[requested + 1]
        moves = [p["move_uci"] for p in game["plays"]]
        upper = min(upper, len(moves) - 1)
        if upper < lower:
            continue
        ply = rng.randint(lower, upper)
        prefix = moves[:ply]
        seed = Seed(
            stable_hash(f"{identity}:{source}:{ply}"),
            source,
            prefix,
            ply,
            "train",
            identity,
        )
        try:
            seed.board()  # Reject illegal, terminal and claimable-draw prefixes.
        except ValueError:
            continue
        groups[requested].append(asdict(seed))
        requested = (requested + 1) % 3
    return groups


def produce(directory):
    directory = Path(directory)
    settings = json.loads((directory / "settings.json").read_text())
    # A killed parent must not leave a network producer behind (Linux runner).
    import ctypes

    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM
    if os.getppid() != parent or parent == 1:
        return
    from imba_chess.self_play.runtime import run_lock

    with run_lock(directory / "producer-lock"):
        files = sorted(directory.glob("block-*.json"))
        last = json.loads(files[-1].read_text()) if files else None
        number = len(files)
        source = iterator = dataset = None
        while True:
            consumer = json.loads((directory / "consumer.json").read_text())
            # Follow the furthest requested bucket, including blocks where that
            # bucket had no eligible starts; otherwise two empty blocks deadlock.
            consumed = max(consumer["bucket_blocks"])
            if number >= consumed + settings["prefetch_blocks"]:
                time.sleep(0.2)
                continue
            if last and last["exhausted"]:
                return
            if source is None:
                dataset, source = open_source(settings["dataset"])
                if last:
                    source.load_state_dict(last["source_state"])
                iterator = iter(source)
            rows, exhausted = [], False
            for _ in range(settings["block_rows"]):
                try:
                    rows.append(next(iterator))
                except StopIteration:
                    exhausted = True
                    break
            groups = make_block(
                dataset,
                rows,
                identity=settings["identity"],
                number=number,
                run_seed=settings["run_seed"],
            )
            last = dict(
                number=number,
                identity=settings["identity"],
                source_state=source.state_dict(),
                exhausted=exhausted,
                source_rows=len(rows),
                groups=groups,
            )
            atomic_json(directory / f"block-{number:08d}.json", last)
            print(
                json.dumps(
                    dict(
                        event="block_ready",
                        block=number,
                        rows=len(rows),
                        starts=list(map(len, groups)),
                    )
                ),
                flush=True,
            )
            number += 1


class StreamingStarts:
    def __init__(
        self, directory, config, *, should_stop=lambda: False, start_worker=True
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.should_stop = should_stop
        self.worker = self.log = None
        self.retry = []
        self.cached_blocks = {}
        self.wait_seconds = 0.0
        settings = dict(
            dataset=dataset_settings(config.base_config),
            **asdict(config.streaming),
            run_seed=config.run.seed,
            buckets=BUCKETS,
            schema_version=1,
        )
        settings["identity"] = stable_hash(json.dumps(settings, sort_keys=True))
        # Normalize tuples through JSON before comparing saved settings.
        settings = json.loads(json.dumps(settings))
        path = self.directory / "settings.json"
        if path.exists():
            if json.loads(path.read_text()) != settings:
                raise ValueError("stream settings changed; use a separate run")
        else:
            atomic_json(path, settings)
        self.identity = settings["identity"]
        self.path = self.directory / "consumer.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["identity"] != self.identity:
                raise ValueError("stream consumer identity mismatch")
        else:
            self.state = dict(
                identity=self.identity,
                block=0,
                bucket_blocks=[0, 0, 0],
                offsets=[0, 0, 0],
                sequence=0,
                pending={},
                launched=[0, 0, 0, 0],
                retired={},
            )
            self.save()
        if start_worker:
            self.log = (self.directory / "producer.log").open("ab", buffering=0)
            self.worker = subprocess.Popen(
                [sys.executable, "-u", "-m", __name__, str(self.directory)],
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def save(self):
        atomic_json(self.path, self.state)

    def close(self):
        if self.worker is not None:
            self.worker.terminate() if self.worker.poll() is None else None
            try:
                self.worker.wait(timeout=self.config.streaming.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
            self.worker = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _block(self, number=None):
        number = self.state["block"] if number is None else number
        if number in self.cached_blocks:
            return self.cached_blocks[number]
        path = self.directory / f"block-{number:08d}.json"
        start = time.monotonic()
        try:
            while not path.exists():
                if self.should_stop():
                    raise InterruptedError("stopped while waiting for streamed starts")
                if self.worker is not None and self.worker.poll() is not None:
                    raise RuntimeError(
                        f"start producer exited: see {self.directory / 'producer.log'}"
                    )
                if time.monotonic() - start > self.config.streaming.startup_timeout:
                    raise TimeoutError("stream start queue timed out")
                time.sleep(0.1)
            block = json.loads(path.read_text())
            if block["identity"] != self.identity:
                raise ValueError("stream block identity mismatch")
            self.cached_blocks = {
                k: v for k, v in self.cached_blocks.items() if k >= self.state["block"]
            }
            self.cached_blocks[number] = block
            return block
        finally:
            self.wait_seconds += time.monotonic() - start

    def warm(self):
        self._block()

    def begin_phase(self, iteration, actor_id, completed):
        self.reconcile(completed)
        for gid, record in list(self.state["pending"].items()):
            if (record["iteration"], record["actor_id"]) != (iteration, actor_id):
                raise ValueError(
                    "pending streamed launches belong to another actor/phase"
                )
        self.retry = sorted(
            self.state["pending"],
            key=lambda gid: self.state["pending"][gid]["sequence"],
        )

    def reconcile(self, completed):
        for gid in list(self.state["pending"]):
            if gid in completed:
                del self.state["pending"][gid]
        self.save()

    def retire(self, gid, reason):
        if gid in self.state["pending"]:
            del self.state["pending"][gid]
            counts = self.state["retired"]
            counts[reason] = counts.get(reason, 0) + 1
            self.save()

    def finish_phase(self):
        for gid in list(self.state["pending"]):
            self.retire(gid, "phase_finished_uncompleted")

    def next_launch(self, iteration, actor_id):
        if self.retry:
            gid = self.retry.pop(0)
            return Seed(**self.state["pending"][gid]["seed"]), gid
        seq = self.state["sequence"]
        order = list(range(4))
        random.Random(f"{self.config.run.seed}:mixture:{seq // 4}").shuffle(order)
        bucket = order[seq % 4]
        if bucket == 0:
            source = f"self-play-initial:{self.identity}:{seq}"
            while source_split(source) != "train":
                source += "x"
            seed = Seed(stable_hash(source), source, [], 0, "train", self.identity)
        else:
            while True:
                if self.should_stop():
                    raise InterruptedError("stopped while selecting streamed starts")
                block = self._block(self.state["bucket_blocks"][bucket - 1])
                group = block["groups"][bucket - 1]
                offset = self.state["offsets"][bucket - 1]
                if offset < len(group):
                    seed = Seed(**group[offset])
                    self.state["offsets"][bucket - 1] += 1
                    break
                if block["exhausted"]:
                    raise RuntimeError("human training stream exhausted")
                self.state["bucket_blocks"][bucket - 1] += 1
                self.state["offsets"][bucket - 1] = 0
                self.state["block"] = min(self.state["bucket_blocks"])
                self.save()
        gid = stable_hash(
            f"{self.identity}:{iteration}:{actor_id}:{seq}:{seed.seed_id}"
        )
        self.state["sequence"] += 1
        self.state["launched"][bucket] += 1
        self.state["pending"][gid] = dict(
            seed=asdict(seed), iteration=iteration, actor_id=actor_id, sequence=seq
        )
        self.save()  # Durable before the collector launches the game.
        return seed, gid

    def report(self):
        return dict(
            stream_launched=self.state["launched"],
            stream_pending=len(self.state["pending"]),
            stream_block=self.state["block"],
            stream_wait_seconds=self.wait_seconds,
            stream_retired=self.state["retired"],
        )


if __name__ == "__main__":
    produce(sys.argv[1])
