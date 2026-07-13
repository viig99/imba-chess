#!/usr/bin/env python3
"""Stratify held-out value-head quality by game-progress bucket.

value_weight_alpha discounts early-game tokens in the TRAINING loss
(progress**alpha). This script checks the resulting effect on held-out
predictions directly: does the value head end up sharp/confident near the
end of the game and comparatively flat/hedged early on, more than the loss
weighting alone would explain? Reuses the same per-token target formula as
HSTUChessModel.forward (value_target from game_result_white + turn_id), but
recomputes it unweighted per progress decile instead of relying on the
model's single aggregate value_loss.

Usage: python scripts/eval_value_by_progress.py --config <toml> --checkpoint <pt>
       [--split val|test] [--max-games N]
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import LichessDataset, build_event_dataloader, load_or_create_static_move_vocab
from imba_chess.eval.position_evaluator import load_hstu_checkpoint
from imba_chess.model import create_batch_block_mask

NUM_BUCKETS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-games", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=None,
        help="Override dataloader.max_tokens_per_batch (the default config's "
        "batch size is sized for a 4090 and OOMs on smaller GPUs).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_config = load_repo_config(args.config)
    if args.max_tokens_per_batch is not None:
        repo_config = replace(
            repo_config,
            dataloader=replace(
                repo_config.dataloader, max_tokens_per_batch=args.max_tokens_per_batch
            ),
        )

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

    ignore_index = int(repo_config.model.ignore_index)
    bucket_loss_sum = torch.zeros(NUM_BUCKETS)
    bucket_entropy_sum = torch.zeros(NUM_BUCKETS)
    bucket_confidence_sum = torch.zeros(NUM_BUCKETS)
    bucket_count = torch.zeros(NUM_BUCKETS)
    total_games = 0

    model.eval()
    with torch.inference_mode(), autocast_ctx:
        for batch in tqdm(loader, desc=f"value-by-progress[{args.split}]", unit="batch"):
            seq_offsets = batch["seq_offsets"].to(device=device, dtype=torch.long, non_blocking=True)
            block_mask = create_batch_block_mask(
                seq_offsets=seq_offsets, total_tokens=int(batch["total_tokens"]), device=device
            )
            output = model(batch, block_mask=block_mask, return_loss=True)
            value_logits = output["value_logits"].float()

            counts = seq_offsets[1:] - seq_offsets[:-1]
            token_game_id = torch.repeat_interleave(
                torch.arange(counts.numel(), device=device), counts
            )
            token_pos = torch.arange(value_logits.shape[0], device=device) - seq_offsets.index_select(
                0, token_game_id
            )
            seq_len = counts.index_select(0, token_game_id).clamp_min(1)
            progress = token_pos.to(torch.float32) / (seq_len.to(torch.float32) - 1.0).clamp_min(1.0)

            game_result_white = batch["game_result_white"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            z_token = game_result_white.index_select(0, token_game_id)
            turn_id = batch["turn_id"].to(device=device, dtype=torch.long, non_blocking=True)
            y = torch.where(turn_id == 0, z_token, -z_token)
            value_target = (y + 1).clamp(min=0, max=2)

            target_move_id = batch["target_move_id"].to(device=device, dtype=torch.long)
            valid_mask = target_move_id != ignore_index

            per_token_loss = F.cross_entropy(value_logits, value_target, reduction="none")
            probs = F.softmax(value_logits, dim=-1)
            confidence = probs.max(dim=-1).values
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

            bucket_idx = (progress * NUM_BUCKETS).long().clamp(max=NUM_BUCKETS - 1)
            for b in range(NUM_BUCKETS):
                mask = valid_mask & (bucket_idx == b)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                bucket_loss_sum[b] += per_token_loss[mask].sum().item()
                bucket_entropy_sum[b] += entropy[mask].sum().item()
                bucket_confidence_sum[b] += confidence[mask].sum().item()
                bucket_count[b] += n

            total_games += int(batch["num_games"])
            if total_games >= args.max_games:
                break

    print(f"\nsplit={args.split} games={total_games}")
    print(f"{'progress bucket':<18}{'tokens':>10}{'mean loss':>12}{'mean entropy':>14}{'mean confidence':>17}")
    for b in range(NUM_BUCKETS):
        n = bucket_count[b].item()
        if n == 0:
            continue
        lo, hi = b / NUM_BUCKETS, (b + 1) / NUM_BUCKETS
        print(
            f"[{lo:.1f}, {hi:.1f})".ljust(18)
            + f"{int(n):>10}"
            + f"{bucket_loss_sum[b].item() / n:>12.4f}"
            + f"{bucket_entropy_sum[b].item() / n:>14.4f}"
            + f"{bucket_confidence_sum[b].item() / n:>17.4f}"
        )


if __name__ == "__main__":
    main()
