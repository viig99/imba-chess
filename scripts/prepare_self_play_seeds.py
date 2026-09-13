"""Prepare offline takeover prefixes from a provenance-tagged training corpus."""

import argparse
import json
from pathlib import Path
from imba_chess.self_play.seeds import prepare_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--min-ply", type=int, default=20)
    parser.add_argument("--max-ply", type=int, default=120)
    args = parser.parse_args()
    provenance = json.loads(
        args.corpus.with_suffix(args.corpus.suffix + ".provenance.json").read_text()
    )
    seeds = prepare_seeds(
        args.corpus,
        args.output,
        provenance=provenance,
        run_seed=args.seed,
        max_games=args.max_games,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
    )
    print(
        f'Prepared {len(seeds)} prefixes: {sum(s.split == "monitor" for s in seeds)} monitoring'
    )


if __name__ == "__main__":
    main()
