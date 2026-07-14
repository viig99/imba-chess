#!/usr/bin/env python3
"""Held-out policy-KL loss diagnostic for Phase 1b sigma-sweep probes.

Loads a checkpoint and a rollout parquet, holds out a deterministic
fraction of the rollout ROWS (not the original train/val split -- rollouts
only exist for train-split games, see scripts/generate_search_rollouts.py),
and reports the mean policy_kl_loss (the same masked-softmax-KL formula
HSTUChessModel.forward computes during training) over that held-out slice
at a given --sigma. Cheap, fast first-pass filter for comparing sigma
candidates before spending live-eval compute -- see docs/superpowers/specs/
2026-07-13-phase1b-policy-distillation-design.md Part 5.

Usage: python scripts/eval_policy_kl_loss.py --checkpoint <path> \
    --rollout-path <path> --sigma 1.0
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from dataclasses import replace
from pathlib import Path

import torch
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import (
    LichessDataset,
    build_event_dataloader,
    load_or_create_static_move_vocab,
    load_rollout_lookup,
)
from imba_chess.eval.position_evaluator import load_hstu_checkpoint


def _is_holdout_row(game_id: str, ply: int, holdout_fraction: float) -> bool:
    """Deterministic held-out split by hashing (game_id, ply).

    See this script's module docstring for why this splits rollout ROWS
    rather than reusing the original Lichess train/val split.
    """
    digest = hashlib.sha256(f"{game_id}:{ply}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < holdout_fraction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-path", type=Path, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--max-games", type=int, default=2000)
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
    if not 0.0 <= args.holdout_fraction <= 1.0:
        raise ValueError("--holdout-fraction must be in [0, 1]")

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
        require_value_head=False,
    )
    # load_hstu_checkpoint builds the model config via
    # build_hstu_chess_config(repo_config.model, ...), which defaults
    # policy_kl_sigma=1.0 and never sees --sigma. policy_kl_sigma is a
    # forward()-time scalar only (never baked into any layer's shape/
    # weights), so overriding it on the already-constructed frozen config is
    # safe and avoids needing load_hstu_checkpoint to grow a new parameter.
    model.config = replace(model.config, policy_kl_sigma=float(args.sigma))

    full_rollout_lookup = load_rollout_lookup(args.rollout_path)
    holdout_lookup = {
        key: row
        for key, row in full_rollout_lookup.items()
        if _is_holdout_row(key[0], key[1], args.holdout_fraction)
    }
    print(
        f"Held out {len(holdout_lookup)} / {len(full_rollout_lookup)} rollout rows "
        f"(holdout_fraction={args.holdout_fraction})"
    )
    if not holdout_lookup:
        raise ValueError(
            "No rows fell into the held-out split -- increase --holdout-fraction "
            "or check --rollout-path points at a non-empty parquet."
        )

    # No train_max_games field exists on DatasetConfig/LichessDataset (only
    # val_max_games/test_max_games do -- train is meant to be streamed, not
    # capped up front); --max-games is instead enforced below via the
    # games_seen counter breaking out of the batch loop.
    dataset_cfg = repo_config.dataset
    lichess_dataset = LichessDataset(
        min_avg_elo=dataset_cfg.min_avg_elo,
        min_time_control_sec=dataset_cfg.min_time_control_sec,
        split="train",
        dataset_name=dataset_cfg.dataset_name,
        train_start_month=dataset_cfg.train_start_month,
        train_end_month=dataset_cfg.train_end_month,
        cache_dir=dataset_cfg.cache_dir,
        parquet_batch_size=dataset_cfg.parquet_batch_size,
        max_seq_len=dataset_cfg.max_seq_len,
        board_state_config=repo_config.board_state,
    )
    loader = build_event_dataloader(
        lichess_dataset=lichess_dataset,
        config=repo_config,
        move_vocab=move_vocab,
        rollout_lookup=holdout_lookup,
        rollout_beta=0.0,
    )

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype) if use_amp else contextlib.nullcontext()
    )

    total_weighted_loss = 0.0
    total_weight = 0.0
    games_seen = 0
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        for batch in tqdm(loader, desc="policy-kl-eval", unit="batch"):
            output = model(batch, return_loss=True)
            has_target = batch.get("has_rollout_policy_target")
            if has_target is None or not bool(has_target.any().item()):
                games_seen += int(batch["num_games"])
                if games_seen >= args.max_games:
                    break
                continue
            weight = float(has_target.sum().item())
            total_weighted_loss += float(output["policy_kl_loss"].item()) * weight
            total_weight += weight
            games_seen += int(batch["num_games"])
            if games_seen >= args.max_games:
                break

    if total_weight == 0.0:
        raise ValueError(
            "No held-out rollout-covered tokens were seen -- check --max-games "
            "is large enough to reach the held-out (game_id, ply) pairs."
        )
    mean_policy_kl_loss = total_weighted_loss / total_weight
    print(f"sigma={args.sigma} games_seen={games_seen} held_out_tokens={int(total_weight)}")
    print(f"mean policy_kl_loss={mean_policy_kl_loss:.6f}")


if __name__ == "__main__":
    main()
