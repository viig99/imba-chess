#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data import MoveVocab, MoveVocabConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save a static UCI move vocabulary."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to repo config TOML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_config = load_repo_config(args.config)
    config = MoveVocabConfig(include_unk=repo_config.vocab.include_unk)
    vocab = MoveVocab.build_static(config=config)
    output_path = Path(repo_config.vocab.path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output_path)
    print(f"Saved vocab with {len(vocab)} tokens to {output_path}")


if __name__ == "__main__":
    main()
