from __future__ import annotations

from typing import Any, TypedDict


class EventSequence(TypedDict):
    game_id: str
    game_result_white: int
    seq_token_id: list[int]
    piece_ids: list[list[int]]
    turn_id: list[int]
    castle_id: list[int]
    ep_file_id: list[int]
    halfmove_bucket_id: list[int]
    fullmove_bucket_id: list[int]
    prev_move_id: list[int]
    target_move_id: list[int]
    played_by_elo: list[int]


class JaggedBatch(TypedDict):
    game_id: list[str]
    game_result_white: Any
    num_games: int
    total_tokens: int
    seq_lens: Any
    seq_offsets: Any
    piece_ids: Any
    seq_token_id: Any
    turn_id: Any
    castle_id: Any
    ep_file_id: Any
    halfmove_bucket_id: Any
    fullmove_bucket_id: Any
    prev_move_id: Any
    target_move_id: Any
    played_by_elo: Any
