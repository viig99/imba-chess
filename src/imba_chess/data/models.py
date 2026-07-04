from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BoardTokenConfig:
    en_passant: Literal["legal", "fen", "xfen"] = "legal"
    halfmove_max: int = 100
    halfmove_bucket_size: int = 2
    fullmove_max: int = 200
    fullmove_bucket_size: int = 2


@dataclass(frozen=True)
class BoardState:
    piece_ids: list[int]
    turn_id: int
    castle_id: int
    ep_file_id: int
    halfmove_bucket_id: int
    fullmove_bucket_id: int
