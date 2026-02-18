from __future__ import annotations

from typing import Any, Dict, List


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required for collate_batch") from exc
    return torch


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch of event sequences and return a dict of torch tensors."""
    if not batch:
        raise ValueError("collate_batch received an empty batch")

    torch = _require_torch()

    batch_size = len(batch)
    max_len = max(len(sample["seq_token_id"]) for sample in batch)

    tensor_keys = [
        "seq_token_id",
        "piece_ids",
        "turn_id",
        "castle_id",
        "ep_file_id",
        "halfmove_bucket_id",
        "fullmove_bucket_id",
        "prev_move_id",
        "target_move_id",
        "attention_mask",
        "loss_mask",
    ]

    output: Dict[str, Any] = {
        "game_id": [sample["game_id"] for sample in batch],
    }

    # Initialize padded buffers.
    for key in tensor_keys:
        if key == "piece_ids":
            output[key] = torch.zeros((batch_size, max_len, 64), dtype=torch.long)
        else:
            output[key] = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Fill per sample.
    for row, sample in enumerate(batch):
        seq_len = len(sample["seq_token_id"])
        for key in tensor_keys:
            values = sample[key]
            if key == "piece_ids":
                output[key][row, :seq_len, :] = torch.tensor(values, dtype=torch.long)
            else:
                output[key][row, :seq_len] = torch.tensor(values, dtype=torch.long)

    return output

