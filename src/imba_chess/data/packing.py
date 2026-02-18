from __future__ import annotations

from typing import Iterable, Iterator

from .collate import collate_jagged_batch
from .types import EventSequence, JaggedBatch

try:
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover
    class IterableDataset:  # type: ignore[override]
        pass


def iter_max_tokens_batches(
    events: Iterable[EventSequence],
    *,
    max_tokens_per_batch: int,
) -> Iterator[JaggedBatch]:
    if max_tokens_per_batch < 1:
        raise ValueError("max_tokens_per_batch must be >= 1")

    current_batch: list[EventSequence] = []
    current_tokens = 0

    for sample in events:
        seq_len = len(sample["seq_token_id"])

        if seq_len > max_tokens_per_batch:
            if current_batch:
                yield collate_jagged_batch(current_batch)
                current_batch = []
                current_tokens = 0
            yield collate_jagged_batch([sample])
            continue

        if current_batch and (current_tokens + seq_len) > max_tokens_per_batch:
            yield collate_jagged_batch(current_batch)
            current_batch = [sample]
            current_tokens = seq_len
        else:
            current_batch.append(sample)
            current_tokens += seq_len

    if current_batch:
        yield collate_jagged_batch(current_batch)


class MaxTokensJaggedBatchDataset(IterableDataset):
    """Pack event sequences into jagged batches constrained by max token count."""

    def __init__(self, event_dataset: IterableDataset, max_tokens_per_batch: int) -> None:
        self.event_dataset = event_dataset
        self.max_tokens_per_batch = max_tokens_per_batch

    def __iter__(self) -> Iterator[JaggedBatch]:
        yield from iter_max_tokens_batches(
            self.event_dataset,
            max_tokens_per_batch=self.max_tokens_per_batch,
        )
