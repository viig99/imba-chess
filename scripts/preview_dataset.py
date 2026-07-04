#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import LichessDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and preview parsed game rows from LichessDataset."
    )
    parser.add_argument(
        "--num-games",
        "--num-samples",
        dest="num_games",
        type=int,
        default=20,
        help="Number of parsed game rows to emit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to repo config TOML.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional output file path. If omitted, writes to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_config = load_repo_config(args.config)

    dataset = LichessDataset(
        min_avg_elo=repo_config.dataset.min_avg_elo,
        split=repo_config.dataset.split,
        dataset_name=repo_config.dataset.dataset_name,
        train_start_month=repo_config.dataset.train_start_month,
        train_end_month=repo_config.dataset.train_end_month,
        val_start_month=repo_config.dataset.val_start_month,
        val_end_month=repo_config.dataset.val_end_month,
        test_start_month=repo_config.dataset.test_start_month,
        test_end_month=repo_config.dataset.test_end_month,
        val_max_games=repo_config.dataset.val_max_games,
        test_max_games=repo_config.dataset.test_max_games,
        cache_dir=repo_config.dataset.cache_dir,
        parquet_batch_size=repo_config.dataset.parquet_batch_size,
        max_seq_len=repo_config.dataset.max_seq_len,
        board_state_config=repo_config.board_state,
    )

    emitted = 0
    out_file = None
    try:
        if args.output_jsonl is not None:
            args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            out_file = args.output_jsonl.open("w", encoding="utf-8")

        for sample in dataset.stream():
            payload = dataclasses.asdict(sample) if dataclasses.is_dataclass(sample) else sample
            line = json.dumps(payload, ensure_ascii=True)
            if out_file is not None:
                out_file.write(line + "\n")
            else:
                print(line)

            emitted += 1
            if emitted >= args.num_games:
                break
    finally:
        if out_file is not None:
            out_file.close()

    destination = str(args.output_jsonl) if args.output_jsonl else "stdout"
    print(f"Emitted {emitted} games to {destination}.")


if __name__ == "__main__":
    main()
