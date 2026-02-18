#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from imba_chess.data import MoveVocab, MoveVocabConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save a static UCI move vocabulary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/move_vocab_static_uci.json"),
        help="Output path for vocab JSON.",
    )
    parser.add_argument(
        "--include-unk",
        action="store_true",
        help="Include <unk> token (disabled by default for static vocab).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MoveVocabConfig(include_unk=args.include_unk)
    vocab = MoveVocab.build_static(config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(args.output)
    print(f"Saved vocab with {len(vocab)} tokens to {args.output}")


if __name__ == "__main__":
    main()

