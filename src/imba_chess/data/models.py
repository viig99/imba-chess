from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


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


@dataclass(frozen=True)
class PlayerInfo:
    name: str
    elo: int


@dataclass(frozen=True)
class GamePlayers:
    white: PlayerInfo
    black: PlayerInfo


@dataclass(frozen=True)
class GameMetadata:
    event: str
    termination: str
    eco: str
    opening: str
    time_control: str
    utc_date: str
    utc_time: str


@dataclass(frozen=True)
class PlayRecord:
    play_id: int
    move_uci: str
    move_san: str
    state: BoardState
    time_remaining_seconds: Optional[float]
    time_taken_seconds: Optional[float]
    played_by_color: Literal["white", "black"]
    played_by: str
    played_by_elo: int
    opponent_player: str
    opponent_elo: int
    outcome_for_player: Literal["win", "loss", "draw"]


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    result: str
    winner_side: Optional[Literal["white", "black"]]
    winner_player: Optional[str]
    winner_elo: Optional[int]
    loser_player: Optional[str]
    loser_elo: Optional[int]
    average_elo: float
    num_plies: int
    players: GamePlayers
    metadata: GameMetadata
    plays: list[PlayRecord]

