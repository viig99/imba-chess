#!/usr/bin/env python3
"""Report held-out value loss for a checkpoint on the val/test split.

train.py's Ignite evaluators call model(..., return_loss=False) and only
track policy metrics (cross-entropy, hr@k, MRR) -- value_loss is never
computed on held-out data anywhere in the existing pipeline, only on live
training batches. This script fills that gap for comparing candidate
[expert_iteration] beta values against held-out value loss, per the plan's
tuning methodology (Stage 2).

Usage: python scripts/eval_value_loss.py --config <toml> --checkpoint <pt>
       [--split val|test] [--max-games N]
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from pathlib import Path

import torch
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import LichessDataset, build_event_dataloader, load_or_create_static_move_vocab
from imba_chess.eval.position_evaluator import load_hstu_checkpoint
from imba_chess.model import create_batch_block_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-games", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_config = load_repo_config(args.config)

    device_arg = args.device or repo_config.training.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype or repo_config.training.dtype]

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path, include_unk=repo_config.vocab.include_unk
    )
    model, _ = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=False,
        require_value_head=True,
    )

    if args.split == "val":
        dataset_cfg = replace(repo_config.dataset, val_max_games=args.max_games)
    else:
        dataset_cfg = replace(repo_config.dataset, test_max_games=args.max_games)
    dataset_cfg = replace(dataset_cfg, split=args.split)
    lichess_dataset = LichessDataset(
        min_avg_elo=dataset_cfg.min_avg_elo,
        min_time_control_sec=dataset_cfg.min_time_control_sec,
        split=args.split,
        dataset_name=dataset_cfg.dataset_name,
        train_start_month=dataset_cfg.train_start_month,
        train_end_month=dataset_cfg.train_end_month,
        val_start_month=dataset_cfg.val_start_month,
        val_end_month=dataset_cfg.val_end_month,
        test_start_month=dataset_cfg.test_start_month,
        test_end_month=dataset_cfg.test_end_month,
        val_max_games=dataset_cfg.val_max_games,
        test_max_games=dataset_cfg.test_max_games,
        cache_dir=dataset_cfg.cache_dir,
        parquet_batch_size=dataset_cfg.parquet_batch_size,
        max_seq_len=dataset_cfg.max_seq_len,
        board_state_config=repo_config.board_state,
    )
    loader = build_event_dataloader(
        lichess_dataset=lichess_dataset,
        config=replace(repo_config, dataset=dataset_cfg),
        move_vocab=move_vocab,
    )

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype) if use_amp else contextlib.nullcontext()
    )

    total_value_loss = 0.0
    total_policy_loss = 0.0
    total_games = 0
    total_tokens = 0
    num_batches = 0
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        for batch in tqdm(loader, desc=f"value-loss[{args.split}]", unit="batch"):
            seq_offsets = batch["seq_offsets"].to(device=device, dtype=torch.long, non_blocking=True)
            block_mask = create_batch_block_mask(
                seq_offsets=seq_offsets, total_tokens=int(batch["total_tokens"]), device=device
            )
            output = model(batch, block_mask=block_mask, return_loss=True)
            games = int(batch["num_games"])
            tokens = int(batch["total_tokens"])
            total_value_loss += float(output["value_loss"].item()) * games
            total_policy_loss += float(output["policy_loss"].item()) * games
            total_games += games
            total_tokens += tokens
            num_batches += 1
            if total_games >= args.max_games:
                break

    print(f"\nsplit={args.split} games={total_games} tokens={total_tokens} batches={num_batches}")
    print(f"  mean value_loss: {total_value_loss / max(1, total_games):.6f}")
    print(f"  mean policy_loss: {total_policy_loss / max(1, total_games):.6f}")


if __name__ == "__main__":
    main()
