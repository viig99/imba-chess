from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def iter_shard(items: Iterable[T], shard_id: int, num_shards: int) -> Iterator[T]:
    """Yield only items that belong to the given shard."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    for index, item in enumerate(items):
        if index % num_shards == shard_id:
            yield item

