"""Data utilities for imba_chess.

Lazy re-exports (PEP 562 module `__getattr__`) rather than eager top-level
imports: `lichess_dataset.py` imports the `datasets` package, which in this
environment transitively imports torch, and `torch_iterable.py` /
`dataloader.py` are themselves torch-oriented. Eagerly importing them here
would make `import imba_chess.data.<anything>` -- including
`board_state.py`, `move_vocab.py`, `models.py`, `event_builder.py`, all of
which the torch-free multiprocess eval actor worker needs
(`src/imba_chess/eval/actor_worker.py`) -- transitively import torch merely
by touching this package's `__init__`, regardless of what the target
submodule itself imports. Lazy attribute resolution keeps
`from imba_chess.data import LichessDataset` (etc.) working exactly as
before while letting `import imba_chess.data.board_state` /
`import imba_chess.data.move_vocab` stay genuinely torch-free.
"""

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
    "compute_blended_value_target",
    "POLICY_KL_MAX_ARMS",
    "arm_vocab_ids_and_qhat",
    "RolloutRow",
    "assert_rollout_checkpoint_consistency",
    "load_rollout_lookup",
    "write_rollout_parquet",
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
    "POLICY_KL_MAX_ARMS": "policy_target_kl",
    "arm_vocab_ids_and_qhat": "policy_target_kl",
    "RolloutRow": "rollout_store",
    "assert_rollout_checkpoint_consistency": "rollout_store",
    "load_rollout_lookup": "rollout_store",
    "write_rollout_parquet": "rollout_store",
    "TorchLichessIterableDataset": "torch_iterable",
    "EventSequence": "types",
    "JaggedBatch": "types",
    "compute_blended_value_target": "value_target_blend",
}


def __getattr__(name: str) -> Any:
    module_name = _SOURCE_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)
