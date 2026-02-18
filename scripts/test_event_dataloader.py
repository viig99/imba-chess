#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import (
    LichessDataset,
    build_event_dataloader,
    load_or_create_static_move_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build event dataloader and print tensor batch shapes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to repo config TOML.",
    )
    parser.add_argument("--num-batches", type=int, default=2, help="How many batches to print.")
    return parser.parse_args()


def make_dataset(config) -> LichessDataset:
    return LichessDataset(
        min_avg_elo=config.dataset.min_avg_elo,
        split=config.dataset.split,
        dataset_name=config.dataset.dataset_name,
        train_start_month=config.dataset.train_start_month,
        train_end_month=config.dataset.train_end_month,
        val_start_month=config.dataset.val_start_month,
        val_end_month=config.dataset.val_end_month,
        test_start_month=config.dataset.test_start_month,
        test_end_month=config.dataset.test_end_month,
        val_max_games=config.dataset.val_max_games,
        test_max_games=config.dataset.test_max_games,
        cache_dir=config.dataset.cache_dir,
        parquet_batch_size=config.dataset.parquet_batch_size,
        max_seq_len=config.dataset.max_seq_len,
        return_dataclasses=config.dataset.return_dataclasses,
        board_state_config=config.board_state,
    )


def main() -> None:
    args = parse_args()
    config = load_repo_config(args.config)

    move_vocab = load_or_create_static_move_vocab(
        path=config.vocab.path,
        include_unk=config.vocab.include_unk,
    )
    print(f"Move vocab size: {len(move_vocab)}")

    stream_source = make_dataset(config)
    loader = build_event_dataloader(
        lichess_dataset=stream_source,
        config=config,
        move_vocab=move_vocab,
    )

    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx}")
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, (int, float, bool, str)):
                print(f"  {key}: {type(value).__name__}={value}")
            else:
                print(f"  {key}: {type(value).__name__}, len={len(value)}")
        if (batch_idx + 1) >= args.num_batches:
            break


if __name__ == "__main__":
    main()
