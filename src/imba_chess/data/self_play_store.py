"""Immutable game shards, atomic indexing, bounded active replay and crash recovery."""

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import uuid

import pyarrow as pa
import pyarrow.parquet as pq


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_game(game):
    if (
        game["schema_version"] != 1
        or game["status"] != "completed"
        or game["outcome_white"] not in (-1, 0, 1)
    ):
        raise ValueError("only completed, outcome-labeled games belong in replay")
    if (
        len(game["prefix_moves"]) != game["takeover_ply"]
        or not game["moves"]
        or len(game["moves"]) != len(game["targets"])
    ):
        raise ValueError("trajectory alignment mismatch")
    for target in game["targets"]:
        weight = target.get("policy_training_weight", 1.0)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("policy_training_weight must be finite and nonnegative")
        priors = target.get("root_log_priors")
        if priors is not None and (
            len(priors) != len(target["policy"]) or any(not math.isfinite(p) for p in priors)
        ):
            raise ValueError("actor log priors must be finite and aligned")
        ids, policy = target["legal_ids"], target["policy"]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(ids) != len(policy)
            or target["move_id"] not in ids
        ):
            raise ValueError("invalid sparse policy")
        if (
            any(not math.isfinite(p) or p < 0 for p in policy)
            or abs(sum(policy) - 1) > 1e-5
        ):
            raise ValueError("policy must be normalized")


class SelfPlayStore:
    """Single-writer store; the run lock is owned by the orchestrator.

    Inactive shard metadata remains indexed for deduplication. Garbage collection
    is explicit and must be called only with all checkpoint-pinned shard names.
    Read-only instances use the published manifest without recovery or trimming;
    they never create files or publish state. They do not pin shards against GC.
    """

    def __init__(
        self,
        directory,
        *,
        window_positions=50000,
        flush_games=16,
        flush_positions=2048,
        read_only=False,
    ):
        self.directory = Path(directory)
        self.read_only = read_only
        if not read_only:
            self.directory.mkdir(parents=True, exist_ok=True)
        if min(window_positions, flush_games, flush_positions) < 1:
            raise ValueError("replay bounds must be positive")
        self.window_positions, self.flush_games, self.flush_positions = (
            window_positions,
            flush_games,
            flush_positions,
        )
        self.pending = OrderedDict()
        self.pending_positions = 0
        path = self.directory / "manifest.json"
        self.manifest = (
            json.loads(path.read_text())
            if read_only or path.exists()
            else dict(schema_version=1, shards=[], active=[])
        )
        if self.manifest["schema_version"] != 1:
            raise ValueError("unsupported replay schema")
        indexed = {s["file"] for s in self.manifest["shards"]}
        self.seen = {g["id"] for s in self.manifest["shards"] for g in s["games"]}
        self.index = {
            g["id"]: (s["file"], g) for s in self.manifest["shards"] for g in s["games"]
        }
        if read_only:
            return
        for shard in sorted(self.directory.glob("shard-*.parquet")):
            if shard.name not in indexed:
                self._index_shard(shard)
        self._trim()
        self._publish()

    def _index_shard(self, path):
        entries = []
        for row in pq.ParquetFile(path).iter_batches(batch_size=16):
            for stored in row.to_pylist():
                game = json.loads(stored["trajectory"])
                validate_game(game)
                gid = game["game_id"]
                if gid in self.seen:
                    continue
                self.seen.add(gid)
                entry = dict(id=gid, positions=len(game["moves"]), split=game["split"])
                entries.append(entry)
                self.index[gid] = (path.name, entry)
                self.manifest["active"].append(gid)
        self.manifest["shards"].append(dict(file=path.name, games=entries))

    def _trim(self):
        active = self.manifest["active"]
        total = sum(self.index[g][1]["positions"] for g in active)
        while active and total > self.window_positions:
            total -= self.index[active.pop(0)][1]["positions"]

    def _require_writer(self):
        if self.read_only:
            raise PermissionError("replay store is read-only")

    def _publish(self):
        self._require_writer()
        atomic_json(self.directory / "manifest.json", self.manifest)

    def add(self, game):
        self._require_writer()
        validate_game(game)
        gid = game["game_id"]
        if gid in self.seen or gid in self.pending:
            return False
        if len(game["moves"]) > self.window_positions:
            raise ValueError("single game exceeds replay window")
        self.pending[gid] = game
        self.pending_positions += len(game["moves"])
        if (
            len(self.pending) >= self.flush_games
            or self.pending_positions >= self.flush_positions
        ):
            self.flush()
        return True

    def flush(self):
        self._require_writer()
        if not self.pending:
            return
        # A fixed Arrow schema avoids null-list inference changing between shards.
        table = pa.Table.from_pylist(
            [
                dict(
                    game_id=g["game_id"],
                    positions=len(g["moves"]),
                    trajectory=json.dumps(g, allow_nan=False),
                )
                for g in self.pending.values()
            ],
            schema=pa.schema(
                [
                    ("game_id", pa.string()),
                    ("positions", pa.int32()),
                    ("trajectory", pa.large_string()),
                ]
            ),
        )
        path = (
            self.directory
            / f'shard-{len(self.manifest["shards"]):08d}-{uuid.uuid4().hex}.parquet'
        )
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_directory(self.directory)
        self._index_shard(path)
        self._trim()
        self._publish()
        self.pending.clear()
        self.pending_positions = 0

    def game_ids(self, split="train"):
        return [
            g
            for g in self.manifest["active"]
            if split is None or self.index[g][1]["split"] == split
        ]

    def read_game(self, gid):
        filename, _ = self.index[gid]
        for batch in pq.ParquetFile(self.directory / filename).iter_batches(
            batch_size=16
        ):
            for row in batch.to_pylist():
                if row["game_id"] == gid:
                    return json.loads(row["trajectory"])
        raise KeyError(gid)

    def active_shards(self):
        return sorted({self.index[g][0] for g in self.manifest["active"]})

    def collect_garbage(self, *, pinned_shards):
        self._require_writer()
        keep = set(pinned_shards) | set(self.active_shards())
        for shard in self.manifest["shards"]:
            if shard["file"] not in keep:
                (self.directory / shard["file"]).unlink(missing_ok=True)
        sync_directory(self.directory)
