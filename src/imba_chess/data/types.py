from __future__ import annotations

from typing import Any, TypedDict


class _RolloutEventFields(TypedDict, total=False):
    value_target_soft: list[list[float]]
    has_rollout_value_target: list[int]
    policy_kl_arm_ids: list[list[int]]
    policy_kl_arm_qhat: list[list[float]]
    policy_kl_arm_mask: list[list[bool]]
    has_rollout_policy_target: list[int]


class EventSequence(_RolloutEventFields):
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


class _RolloutJaggedFields(TypedDict, total=False):
    value_target_soft: Any
    has_rollout_value_target: Any
    policy_kl_arm_ids: Any
    policy_kl_arm_qhat: Any
    policy_kl_arm_mask: Any
    has_rollout_policy_target: Any


class JaggedBatch(_RolloutJaggedFields):
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
