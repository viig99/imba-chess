from __future__ import annotations

from types import SimpleNamespace

import pytest

from imba_chess.data.sharding import iter_shard
from imba_chess.data.torch_iterable import TorchLichessIterableDataset


class DummyDataset:
    def __init__(self, values):
        self.values = values

    def stream(self):
        return iter(self.values)


class DummyDatasetWithShard:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def stream(self, *, shard_id=None, num_shards=None):
        self.calls.append((shard_id, num_shards))
        return iter(self.values[shard_id::num_shards])


def test_iter_shard_partitions_evenly():
    values = list(range(12))
    shard_0 = list(iter_shard(values, shard_id=0, num_shards=3))
    shard_1 = list(iter_shard(values, shard_id=1, num_shards=3))
    shard_2 = list(iter_shard(values, shard_id=2, num_shards=3))

    assert shard_0 == [0, 3, 6, 9]
    assert shard_1 == [1, 4, 7, 10]
    assert shard_2 == [2, 5, 8, 11]


def test_iter_shard_validates_inputs():
    with pytest.raises(ValueError):
        list(iter_shard([1, 2], shard_id=0, num_shards=0))
    with pytest.raises(ValueError):
        list(iter_shard([1, 2], shard_id=2, num_shards=2))


def test_torch_iterable_uses_rank_and_worker_sharding(monkeypatch):
    import imba_chess.data.torch_iterable as module

    # rank=1/world=2 and worker=1/workers=3 => shard_id=4, num_shards=6
    monkeypatch.setattr(
        module,
        "get_worker_info",
        lambda: SimpleNamespace(id=1, num_workers=3),
    )

    dataset = TorchLichessIterableDataset(
        dataset=DummyDataset(list(range(12))),
        rank=1,
        world_size=2,
    )
    assert list(dataset) == [4, 10]


def test_torch_iterable_defaults_single_shard(monkeypatch):
    import imba_chess.data.torch_iterable as module

    monkeypatch.setattr(module, "get_worker_info", lambda: None)

    dataset = TorchLichessIterableDataset(dataset=DummyDataset([10, 20, 30]))
    assert list(dataset) == [10, 20, 30]


def test_torch_iterable_prefers_dataset_native_shard_stream(monkeypatch):
    import imba_chess.data.torch_iterable as module

    monkeypatch.setattr(
        module,
        "get_worker_info",
        lambda: SimpleNamespace(id=1, num_workers=2),
    )

    base_dataset = DummyDatasetWithShard(list(range(12)))
    dataset = TorchLichessIterableDataset(
        dataset=base_dataset,
        rank=1,
        world_size=2,
    )
    values = list(dataset)

    assert values == [3, 7, 11]
    assert base_dataset.calls == [(3, 4)]


def test_torch_iterable_validates_rank_world_size():
    with pytest.raises(ValueError):
        list(TorchLichessIterableDataset(dataset=DummyDataset([]), rank=0, world_size=None))
    with pytest.raises(ValueError):
        list(TorchLichessIterableDataset(dataset=DummyDataset([]), rank=1, world_size=1))
