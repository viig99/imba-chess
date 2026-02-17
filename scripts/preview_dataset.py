#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

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
        "--min-avg-elo",
        type=int,
        default=2000,
        help="Keep only games where (WhiteElo + BlackElo)/2 >= this value.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face dataset split to stream from.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Lichess/standard-chess-games",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for Hugging Face metadata/temp files.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=2048,
        help="Parquet streaming batch size (lower uses less peak memory).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional output file path. If omitted, writes to stdout.",
    )
    parser.add_argument(
        "--return-dataclasses",
        action="store_true",
        help="Return dataclass objects from dataset internals (converted to JSON for output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LichessDataset(
        min_avg_elo=args.min_avg_elo,
        split=args.split,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        parquet_batch_size=args.parquet_batch_size,
        return_dataclasses=args.return_dataclasses,
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
