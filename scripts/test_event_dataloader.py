#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import islice

from imba_chess.data import LichessDataset, MoveVocab, build_event_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build move vocab + event dataloader and print tensor batch shapes."
    )
    parser.add_argument("--num-games-vocab", type=int, default=128, help="Games used to build move vocab.")
    parser.add_argument("--batch-size", type=int, default=4, help="Dataloader batch size.")
    parser.add_argument("--num-batches", type=int, default=2, help="How many batches to print.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--min-avg-elo", type=int, default=2000, help="Average Elo threshold.")
    parser.add_argument("--split", default="train", help="HF split.")
    parser.add_argument("--dataset-name", default="Lichess/standard-chess-games", help="HF dataset name.")
    parser.add_argument("--cache-dir", default=None, help="Optional cache dir.")
    parser.add_argument("--parquet-batch-size", type=int, default=2048, help="Parquet streaming batch size.")
    return parser.parse_args()


def make_dataset(args: argparse.Namespace) -> LichessDataset:
    return LichessDataset(
        min_avg_elo=args.min_avg_elo,
        split=args.split,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        parquet_batch_size=args.parquet_batch_size,
    )


def main() -> None:
    args = parse_args()

    # Pass 1: build move vocab from a bounded game sample.
    vocab_source = make_dataset(args)
    vocab_games = list(islice(vocab_source.stream(), args.num_games_vocab))
    if not vocab_games:
        raise RuntimeError("No games found to build move vocab.")

    move_vocab = MoveVocab.build_from_games(vocab_games)
    print(f"Move vocab size: {len(move_vocab)}")

    # Pass 2: stream events into torch DataLoader.
    stream_source = make_dataset(args)
    loader = build_event_dataloader(
        lichess_dataset=stream_source,
        move_vocab=move_vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx}")
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__}, len={len(value)}")
        if (batch_idx + 1) >= args.num_batches:
            break


if __name__ == "__main__":
    main()

