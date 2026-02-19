from __future__ import annotations

import inspect
from typing import Any, Iterator, Optional

from .sharding import iter_shard

try:
    from torch.utils.data import IterableDataset, get_worker_info

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

    class IterableDataset:  # type: ignore[override]
        pass

    def get_worker_info():  # type: ignore[no-untyped-def]
        return None


class TorchLichessIterableDataset(IterableDataset):
    """Torch IterableDataset adapter over LichessDataset with rank+worker sharding."""

    def __init__(
        self,
        dataset: Any,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[Any]:
        rank, world_size = self._resolve_distributed_context()

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        shard_id = (rank * num_workers) + worker_id
        num_shards = world_size * num_workers

        stream_fn = getattr(self.dataset, "stream")
        stream_signature = inspect.signature(stream_fn)
        if "shard_id" in stream_signature.parameters and "num_shards" in stream_signature.parameters:
            return stream_fn(shard_id=shard_id, num_shards=num_shards)
        return iter_shard(
            self.dataset.stream(), shard_id=shard_id, num_shards=num_shards
        )

    def _resolve_distributed_context(self) -> tuple[int, int]:
        if (self.rank is None) != (self.world_size is None):
            raise ValueError("rank and world_size must both be set or both be None")

        if self.rank is not None and self.world_size is not None:
            if self.world_size < 1:
                raise ValueError(f"world_size must be >= 1, got {self.world_size}")
            if self.rank < 0 or self.rank >= self.world_size:
                raise ValueError(
                    f"rank must be in [0, {self.world_size}), got {self.rank}"
                )
            return self.rank, self.world_size

        if TORCH_AVAILABLE:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()

        return 0, 1
