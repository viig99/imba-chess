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
        for key in ("search_wdl", "improved_wdl"):
            wdl = target.get(key)
            if wdl is not None and (
                len(wdl) != 3 or any(not math.isfinite(p) or p < 0 for p in wdl)
                or abs(sum(wdl) - 1) > 1e-5
            ):
                raise ValueError(f"{key} must be a normalized loss/draw/win distribution")
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


def game_summary(game):
    """Small durable collection record; no policies or inherited histories."""
    summary = dict(id=game["game_id"], positions=len(game["moves"]),
                   split=game["split"], status=game["status"],
                   actor_id=game["actor_id"], iteration=game.get("iteration"),
                   termination=game["termination"])
    if "starting_bucket" in game:
        summary["starting_bucket"] = game["starting_bucket"]
    return summary


class SelfPlayStore:
    """Single-writer store; the run lock is owned by the orchestrator.

    Games stay indexed, and deduplicated, while their shard file exists: evicted
    games of a kept shard still count for phase recovery. Garbage collection is
    explicit, must be called only with all checkpoint-pinned shard names, and
    forgets the shards it deletes. A re-delivered game (a launch retried after a
    crash) is always recent, so it is still indexed; launch ids never repeat.
    Read-only instances use the published manifest without recovery or trimming;
    they never create files or publish state. They do not pin shards against GC.

    Shards are the durable record: each is fsynced before it is indexed. The manifest
    is an index of them, rewritten only by explicit flushes (phase ends) and garbage
    collection, not by the automatic flushes of `add`; a writer reopening after a
    crash indexes the shards it is missing, so the recovered window is the same.
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
        self.unpublished = False
        path = self.directory / "manifest.json"
        self.manifest = (
            json.loads(path.read_text())
            if read_only or path.exists()
            else dict(schema_version=1, shards=[], active=[])
        )
        if self.manifest["schema_version"] != 1:
            raise ValueError("unsupported replay schema")
        # Shard numbers keep increasing after garbage collection forgets shards. Older
        # manifests listed every shard ever written, so their count is the next number.
        self.manifest.setdefault("next_shard", len(self.manifest["shards"]))
        indexed = {s["file"] for s in self.manifest["shards"]}
        self.seen = {g["id"] for s in self.manifest["shards"] for g in s["games"]}
        self.index = {
            g["id"]: (s["file"], g) for s in self.manifest["shards"] for g in s["games"]
        }
        if read_only:
            return
        # Manifests from before collection forgot deleted shards still list them.
        gone = [s for s in self.manifest["shards"] if not (self.directory / s["file"]).exists()]
        active = set(self.manifest["active"])
        if any(g["id"] in active for s in gone for g in s["games"]):
            raise FileNotFoundError("active replay shard missing")
        self._forget(gone)
        for shard in sorted(self.directory.glob("shard-*.parquet")):
            if shard.name not in indexed:
                self._index_shard(shard)
        self._trim()
        self._publish()

    def _index_shard(self, path):
        games = []
        for row in pq.ParquetFile(path).iter_batches(batch_size=16):
            for stored in row.to_pylist():
                game = json.loads(stored["trajectory"])
                validate_game(game)
                games.append(game)
        self._index_games(path.name, games)

    def _index_games(self, name, games):
        entries = []
        for game in games:
            gid = game["game_id"]
            if gid in self.seen:
                continue
            self.seen.add(gid)
            entry = game_summary(game)
            entries.append(entry)
            self.index[gid] = (name, entry)
            self.manifest["active"].append(gid)
        self.manifest["shards"].append(dict(file=name, games=entries))
        self.manifest["next_shard"] = max(self.manifest["next_shard"], int(name.split("-")[1]) + 1)

    def _forget(self, shards):
        files = {s["file"] for s in shards}
        for shard in shards:
            for g in shard["games"]:
                self.seen.discard(g["id"])
                del self.index[g["id"]]
        self.manifest["shards"] = [s for s in self.manifest["shards"] if s["file"] not in files]

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
        self.unpublished = False

    def add(self, game, *, before_flush=None):
        """before_flush, if given, runs just before an automatic flush writes."""
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
            self.flush(publish=False, before=before_flush)
        return True

    def flush(self, publish=True, *, before=None):
        """Write pending games as one fsynced shard, calling `before` first;
        publish=False leaves the manifest for a later explicit flush (see the
        class docstring)."""
        self._require_writer()
        if not self.pending:
            if publish and self.unpublished:
                self._publish()
            return
        if before is not None:
            before()
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
            / f'shard-{self.manifest["next_shard"]:08d}-{uuid.uuid4().hex}.parquet'
        )
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_directory(self.directory)
        # Validated by add; index from memory rather than reading the shard back.
        self._index_games(path.name, self.pending.values())
        self._trim()
        self.pending.clear()
        self.pending_positions = 0
        self.unpublished = True
        if publish:
            self._publish()

    def game_ids(self, split="train"):
        return [
            g
            for g in self.manifest["active"]
            if split is None or self.index[g][1]["split"] == split
        ]

    def phase_summaries(self, *, iteration, actor_id):
        """Restore collection counts from metadata, including evicted games.

        Legacy manifests are enriched once, reading each surviving legacy shard
        once. Read-only readers never publish or modify the manifest.
        """
        required = {"actor_id", "iteration", "termination", "status"}
        summaries, changed = [], False
        for shard in self.manifest["shards"]:
            legacy = {entry["id"]: entry for entry in shard["games"]
                      if not required <= entry.keys()}
            recovered = {}
            if legacy:
                path = self.directory / shard["file"]
                if path.exists():
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                        for row in batch.to_pylist():
                            if row["game_id"] in legacy:
                                summary = game_summary(json.loads(row["trajectory"]))
                                recovered[summary["id"]] = summary
                                if not self.read_only:
                                    legacy[summary["id"]].update(summary)
                                    changed = True
            for entry in shard["games"]:
                summary = recovered.get(entry["id"], entry)
                if summary.get("iteration") == iteration and summary.get("actor_id") == actor_id:
                    summaries.append(summary)
        if changed:
            self._publish()
        return summaries

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
        deleted = [s for s in self.manifest["shards"] if s["file"] not in keep]
        for shard in deleted:
            (self.directory / shard["file"]).unlink(missing_ok=True)
        sync_directory(self.directory)
        if deleted:
            self._forget(deleted)
            self._publish()
