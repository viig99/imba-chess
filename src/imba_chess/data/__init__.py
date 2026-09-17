"""Data utilities. Lazy exports avoid loading torch and datasets for board or move-vocabulary helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_event_dataloader",
    "ChessEventIterableDataset",
    "MaxTokensJaggedBatchDataset",
    "collate_jagged_batch",
    "LichessDataset",
    "EventBuilder",
    "EVENT_TOKEN_ID",
    "BOS_TOKEN_ID",
    "TARGET_IGNORE_INDEX",
    "EventSequence",
    "JaggedBatch",
    "MoveVocab",
    "MoveVocabConfig",
    "load_or_create_static_move_vocab",
    "DEFAULT_STATIC_MOVE_VOCAB_PATH",
    "BoardTokenConfig",
    "BoardState",
    "TorchLichessIterableDataset",
    "winpercent_wdl",
]

# name -> submodule (relative to this package) it is defined in.
_SOURCE_MODULE = {
    "collate_jagged_batch": "collate",
    "ChessEventIterableDataset": "dataloader",
    "build_event_dataloader": "dataloader",
    "BOS_TOKEN_ID": "event_builder",
    "EVENT_TOKEN_ID": "event_builder",
    "EventBuilder": "event_builder",
    "TARGET_IGNORE_INDEX": "event_builder",
    "LichessDataset": "lichess_dataset",
    "BoardState": "models",
    "BoardTokenConfig": "models",
    "DEFAULT_STATIC_MOVE_VOCAB_PATH": "move_vocab",
    "MoveVocab": "move_vocab",
    "MoveVocabConfig": "move_vocab",
    "load_or_create_static_move_vocab": "move_vocab",
    "MaxTokensJaggedBatchDataset": "packing",
    "winpercent_wdl": "stockfish_evals",
    "TorchLichessIterableDataset": "torch_iterable",
    "EventSequence": "types",
    "JaggedBatch": "types",
}


def __getattr__(name: str) -> Any:
    module_name = _SOURCE_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
