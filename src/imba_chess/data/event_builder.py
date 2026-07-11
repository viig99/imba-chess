from __future__ import annotations

from typing import Any, Dict

from .move_vocab import MoveVocab
from .rollout_store import RolloutRow
from .types import EventSequence
from .value_target_blend import compute_blended_value_target

EVENT_TOKEN_ID = 0
BOS_TOKEN_ID = 1
TARGET_IGNORE_INDEX = -100


def _result_to_game_result_white(value: Any) -> int:
    text = str(value).strip()
    if text == "1-0":
        return 1
    if text == "0-1":
        return -1
    if text == "1/2-1/2":
        return 0
    raise ValueError(f"Unsupported game result for value target: {text!r}")


class EventBuilder:
    """Build BOS+ply event sequences for next-move prediction."""

    def __init__(
        self,
        move_vocab: MoveVocab,
        *,
        rollout_lookup: Dict[tuple[str, int], RolloutRow] | None = None,
        beta: float = 0.0,
    ) -> None:
        self.move_vocab = move_vocab
        self.rollout_lookup = rollout_lookup
        self.beta = beta

    def build_game(self, game: Dict[str, Any]) -> EventSequence:
        game_result_white = _result_to_game_result_white(game["result"])

        seq_token_id = [BOS_TOKEN_ID]
        piece_ids = [[0] * 64]
        turn_id = [0]
        castle_id = [0]
        ep_file_id = [0]
        halfmove_bucket_id = [0]
        fullmove_bucket_id = [0]
        prev_move_id = [self.move_vocab.start_id]
        target_move_id = [TARGET_IGNORE_INDEX]
        played_by_elo = [0]

        previous_move = self.move_vocab.start_id
        for play in game["plays"]:
            state = play["state"]
            current_move = self.move_vocab.encode(play["move_uci"])
            current_played_by_elo = int(play.get("played_by_elo", 0))

            seq_token_id.append(EVENT_TOKEN_ID)
            piece_ids.append(list(state["piece_ids"]))
            turn_id.append(int(state["turn_id"]))
            castle_id.append(int(state["castle_id"]))
            ep_file_id.append(int(state["ep_file_id"]))
            halfmove_bucket_id.append(int(state["halfmove_bucket_id"]))
            fullmove_bucket_id.append(int(state["fullmove_bucket_id"]))
            prev_move_id.append(previous_move)
            target_move_id.append(current_move)
            played_by_elo.append(current_played_by_elo)

            previous_move = current_move

        result: EventSequence = {
            "game_id": game["game_id"],
            "game_result_white": game_result_white,
            "seq_token_id": seq_token_id,
            "piece_ids": piece_ids,
            "turn_id": turn_id,
            "castle_id": castle_id,
            "ep_file_id": ep_file_id,
            "halfmove_bucket_id": halfmove_bucket_id,
            "fullmove_bucket_id": fullmove_bucket_id,
            "prev_move_id": prev_move_id,
            "target_move_id": target_move_id,
            "played_by_elo": played_by_elo,
        }
        if self.rollout_lookup is not None:
            value_target_soft, has_rollout_value_target = self._build_rollout_value_targets(game)
            result["value_target_soft"] = value_target_soft
            result["has_rollout_value_target"] = has_rollout_value_target
        return result

    def _build_rollout_value_targets(
        self, game: Dict[str, Any]
    ) -> tuple[list[list[float]], list[int]]:
        assert self.rollout_lookup is not None
        num_tokens = len(game["plays"]) + 1
        value_target_soft: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(num_tokens)]
        has_rollout_value_target = [0] * num_tokens
        game_id = game["game_id"]

        for ply_idx in range(len(game["plays"])):
            row = self.rollout_lookup.get((game_id, ply_idx))
            if row is None:
                continue
            token_idx = ply_idx + 1
            value_target_soft[token_idx] = compute_blended_value_target(
                root_wdl_unsearched=row.root_wdl_unsearched,
                backed_value=row.best_arm_backed_value,
                real_outcome_stm=row.real_outcome_stm,
                beta=self.beta,
            )
            has_rollout_value_target[token_idx] = 1

        return value_target_soft, has_rollout_value_target
