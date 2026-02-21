from __future__ import annotations

from typing import List

from .types import EventSequence, JaggedBatch

def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required for collate_jagged_batch") from exc
    return torch


def collate_jagged_batch(batch: List[EventSequence]) -> JaggedBatch:
    """Flatten event sequences into jagged tensors with seq_lens/seq_offsets."""
    if not batch:
        raise ValueError("collate_jagged_batch received an empty batch")

    torch = _require_torch()

    scalar_keys = [
        "seq_token_id",
        "turn_id",
        "castle_id",
        "ep_file_id",
        "halfmove_bucket_id",
        "fullmove_bucket_id",
        "prev_move_id",
        "target_move_id",
        "played_by_elo",
    ]
    per_game_scalar_keys = ["game_result_white"]

    flat_scalars = {key: [] for key in scalar_keys}
    flat_piece_ids: list[list[int]] = []
    seq_lens: list[int] = []
    per_game_scalars = {key: [] for key in per_game_scalar_keys}

    for sample in batch:
        seq_len = len(sample["seq_token_id"])
        game_id = sample.get("game_id", "<unknown>")
        piece_ids = sample["piece_ids"]
        if len(piece_ids) != seq_len:
            raise ValueError(
                f"Sample {game_id} has piece_ids length {len(piece_ids)} "
                f"but seq_token_id length {seq_len}"
            )
        for key in scalar_keys:
            values = sample[key]
            if len(values) != seq_len:
                raise ValueError(
                    f"Sample {game_id} has {key} length {len(values)} "
                    f"but seq_token_id length {seq_len}"
                )
        seq_lens.append(seq_len)
        flat_piece_ids.extend(piece_ids)
        for key in scalar_keys:
            flat_scalars[key].extend(sample[key])
        for key in per_game_scalar_keys:
            per_game_scalars[key].append(sample[key])

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)

    output: JaggedBatch = {
        "game_id": [sample["game_id"] for sample in batch],
        "num_games": len(batch),
        "total_tokens": offsets[-1],
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "seq_offsets": torch.tensor(offsets, dtype=torch.long),
        "piece_ids": torch.tensor(flat_piece_ids, dtype=torch.long),
    }

    for key in scalar_keys:
        output[key] = torch.tensor(flat_scalars[key], dtype=torch.long)
    for key in per_game_scalar_keys:
        output[key] = torch.tensor(per_game_scalars[key], dtype=torch.long)

    return output
