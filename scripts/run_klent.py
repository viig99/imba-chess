"""Run (or resume) KLENT self-play training into an output directory."""

import argparse
from pathlib import Path

from imba_chess.klent.config import load_klent_config
from imba_chess.klent.run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Write an eval-loadable actor snapshot every N iterations.")
    args = parser.parse_args()
    run(load_klent_config(args.config), output=args.output, device=args.device,
        save_every=args.save_every)


if __name__ == "__main__":
    main()
