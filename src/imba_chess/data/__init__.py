"""Data utilities for imba_chess."""

from .collate import collate_jagged_batch
from .dataloader import (
    ChessEventIterableDataset,
    build_event_dataloader,
)
from .event_builder import BOS_TOKEN_ID, EVENT_TOKEN_ID, EventBuilder, TARGET_IGNORE_INDEX
from .lichess_dataset import LichessDataset
from .models import BoardState, BoardTokenConfig
from .move_vocab import (
    DEFAULT_STATIC_MOVE_VOCAB_PATH,
    MoveVocab,
    MoveVocabConfig,
    load_or_create_static_move_vocab,
)
from .packing import MaxTokensJaggedBatchDataset
from .rollout_store import RolloutRow, load_rollout_lookup, write_rollout_parquet
from .torch_iterable import TorchLichessIterableDataset
from .types import EventSequence, JaggedBatch
from .value_target_blend import compute_blended_value_target

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
    "RolloutRow",
    "load_rollout_lookup",
    "write_rollout_parquet",
]
