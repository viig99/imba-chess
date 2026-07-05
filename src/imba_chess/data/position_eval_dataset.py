"""Streaming dataset over Lichess/chess-position-evaluations.

Yields flat per-position samples with soft WDL targets (side-to-move POV)
for training the standalone ValueNet. Verified parquet schema: fen str,
line str, depth uint8, knodes int32, cp int16 nullable, mate int8 nullable
(cp/mate mutually exclusive; the dataset card's prose is partly wrong).
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, Iterable, Iterator

import chess
import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info

from ..model.value_net import board_material_count, cp_to_wdl
from .board_state import BoardStateEncoder

_MATE_TARGET_WIN = (0.0025, 0.0025, 0.995)
_MATE_TARGET_LOSS = (0.995, 0.0025, 0.0025)


class PositionEvalDataset(IterableDataset):
    def __init__(
        self,
        *,
        split: str = "train",
        depth_min: int = 12,
        dataset_name: str = "Lichess/chess-position-evaluations",
        shuffle_buffer_size: int = 10_000,
        seed: int = 0,
        val_permille: int = 50,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0 <= val_permille <= 1000:
            raise ValueError("val_permille must be in [0, 1000]")
        self.split = split
        self.depth_min = int(depth_min)
        self.dataset_name = dataset_name
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.val_permille = int(val_permille)
        self._encoder = BoardStateEncoder()

    def _in_val(self, fen: str) -> bool:
        return zlib.crc32(fen.encode("utf-8")) % 1000 < self.val_permille

    def _row_to_sample(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor] | None:
        depth = row.get("depth")
        cp = row.get("cp")
        mate = row.get("mate")
        if depth is None or int(depth) < self.depth_min:
            return None
        if cp is None and mate is None:
            return None
        fen = row.get("fen")
        if not fen:
            return None
        if self._in_val(fen) != (self.split == "val"):
            return None
        try:
            board = chess.Board(fen)
        except ValueError:
            return None

        stm_sign = 1 if board.turn == chess.WHITE else -1
        if mate is not None:
            mate_stm = int(mate) * stm_sign
            if mate_stm == 0:
                return None
            target = _MATE_TARGET_WIN if mate_stm > 0 else _MATE_TARGET_LOSS
        else:
            target = cp_to_wdl(int(cp) * stm_sign, board_material_count(board))

        state = self._encoder.encode(board)
        return {
            "piece_ids": torch.tensor(state.piece_ids, dtype=torch.long),
            "turn_id": torch.tensor(state.turn_id, dtype=torch.long),
            "castle_id": torch.tensor(state.castle_id, dtype=torch.long),
            "ep_file_id": torch.tensor(state.ep_file_id, dtype=torch.long),
            "wdl_target": torch.tensor(target, dtype=torch.float32),
        }

    def samples_from_rows(
        self, rows: Iterable[Dict[str, Any]]
    ) -> Iterator[Dict[str, torch.Tensor]]:
        for row in rows:
            sample = self._row_to_sample(row)
            if sample is not None:
                yield sample

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rows = load_dataset(self.dataset_name, split="train", streaming=True)
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            rows = rows.shard(num_shards=worker.num_workers, index=worker.id)
        if self.split == "train" and self.shuffle_buffer_size > 0:
            rows = rows.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        yield from self.samples_from_rows(rows)
