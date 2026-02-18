from __future__ import annotations

from typing import Any, Iterator, Optional

from ..config import RepoConfig
from .event_builder import EventBuilder
from .move_vocab import MoveVocab, load_or_create_static_move_vocab
from .packing import MaxTokensJaggedBatchDataset
from .types import EventSequence

try:
    from torch.utils.data import DataLoader, IterableDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    DataLoader = Any  # type: ignore[misc,assignment]

    class IterableDataset:  # type: ignore[override]
        pass


class ChessEventIterableDataset(IterableDataset):
    """Converts game rows into BOS+ply event sequences."""

    def __init__(self, game_iterable_dataset: Any, event_builder: EventBuilder) -> None:
        self.game_iterable_dataset = game_iterable_dataset
        self.event_builder = event_builder

    def __iter__(self) -> Iterator[EventSequence]:
        for game in self.game_iterable_dataset:
            yield self.event_builder.build_game(game)


def build_event_dataloader(
    *,
    lichess_dataset: Any,
    config: Optional[RepoConfig] = None,
    move_vocab: Optional[MoveVocab] = None,
) -> Any:
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("torch is required to build DataLoader")

    runtime = config or RepoConfig()

    resolved_move_vocab = move_vocab or load_or_create_static_move_vocab(
        path=runtime.vocab.path,
        include_unk=runtime.vocab.include_unk,
    )
    game_iterable_dataset = lichess_dataset.as_torch_iterable(
        rank=runtime.dataloader.rank,
        world_size=runtime.dataloader.world_size,
    )
    event_builder = EventBuilder(resolved_move_vocab)
    event_dataset = ChessEventIterableDataset(game_iterable_dataset, event_builder)
    packed_dataset = MaxTokensJaggedBatchDataset(
        event_dataset=event_dataset,
        max_tokens_per_batch=runtime.dataloader.max_tokens_per_batch,
    )

    return DataLoader(
        packed_dataset,
        batch_size=None,
        num_workers=runtime.dataloader.num_workers,
        pin_memory=runtime.dataloader.pin_memory,
    )
