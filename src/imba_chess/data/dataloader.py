from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from .collate import collate_batch
from .event_builder import EventBuilder
from .move_vocab import MoveVocab

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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for game in self.game_iterable_dataset:
            yield self.event_builder.build_game(game)


def build_event_dataloader(
    *,
    lichess_dataset: Any,
    move_vocab: MoveVocab,
    batch_size: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Any:
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("torch is required to build DataLoader")

    game_iterable_dataset = lichess_dataset.as_torch_iterable(rank=rank, world_size=world_size)
    event_builder = EventBuilder(move_vocab)
    event_dataset = ChessEventIterableDataset(game_iterable_dataset, event_builder)

    return DataLoader(
        event_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )

