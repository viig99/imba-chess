from __future__ import annotations

import dataclasses
from typing import Any, Dict

from .move_vocab import MoveVocab

PAD_SEQ_TOKEN_ID = 0
BOS_SEQ_TOKEN_ID = 1


def _as_dict(value: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


class EventBuilder:
    """Build BOS+ply event sequences for next-move prediction."""

    def __init__(self, move_vocab: MoveVocab) -> None:
        self.move_vocab = move_vocab

    def build_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        data = _as_dict(game)
        plays = [_as_dict(play) for play in data["plays"]]

        seq_token_id = [BOS_SEQ_TOKEN_ID]
        piece_ids = [[0] * 64]
        turn_id = [0]
        castle_id = [0]
        ep_file_id = [0]
        halfmove_bucket_id = [0]
        fullmove_bucket_id = [0]
        prev_move_id = [self.move_vocab.start_id]
        target_move_id = [self.move_vocab.pad_id]
        attention_mask = [1]
        loss_mask = [0]

        previous_move = self.move_vocab.start_id
        for play in plays:
            state = _as_dict(play["state"])
            current_move = self.move_vocab.encode(play["move_uci"])

            seq_token_id.append(PAD_SEQ_TOKEN_ID)
            piece_ids.append(list(state["piece_ids"]))
            turn_id.append(int(state["turn_id"]))
            castle_id.append(int(state["castle_id"]))
            ep_file_id.append(int(state["ep_file_id"]))
            halfmove_bucket_id.append(int(state["halfmove_bucket_id"]))
            fullmove_bucket_id.append(int(state["fullmove_bucket_id"]))
            prev_move_id.append(previous_move)
            target_move_id.append(current_move)
            attention_mask.append(1)
            loss_mask.append(1)

            previous_move = current_move

        return {
            "game_id": data["game_id"],
            "seq_token_id": seq_token_id,
            "piece_ids": piece_ids,
            "turn_id": turn_id,
            "castle_id": castle_id,
            "ep_file_id": ep_file_id,
            "halfmove_bucket_id": halfmove_bucket_id,
            "fullmove_bucket_id": fullmove_bucket_id,
            "prev_move_id": prev_move_id,
            "target_move_id": target_move_id,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

