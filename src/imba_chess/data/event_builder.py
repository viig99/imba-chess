from __future__ import annotations

from typing import Any, Dict

from .move_vocab import MoveVocab
from .serialize import as_plain_dict
from .types import EventSequence

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

    def __init__(self, move_vocab: MoveVocab) -> None:
        self.move_vocab = move_vocab

    def build_game(self, game: Dict[str, Any]) -> EventSequence:
        data = as_plain_dict(game)
        plays = [as_plain_dict(play) for play in data["plays"]]
        game_result_white = _result_to_game_result_white(
            data.get("result", data.get("Result", ""))
        )

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
        for play in plays:
            state = as_plain_dict(play["state"])
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

        return {
            "game_id": data["game_id"],
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
