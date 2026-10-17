"""Write the EMA weights of a full self-play checkpoint as a weights-only checkpoint.

run_self_play.py keeps the EMA inside each full trainer state (actor-NNNNNN.pt, state-*.pt)
instead of writing ema-NNNNNN.pt at every save; screens still write their own EMA file.
This produces the file an EMA evaluation loads, e.g. for scripts/run_stockfish_eval.py,
with the same contents the run used to write. The source is not modified.
"""

import argparse
from pathlib import Path
import torch
from imba_chess.self_play.trainer import atomic_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="full self-play checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Memory-mapped: only the EMA tensors are read.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state.get("stage2_schema") != 1 or "ema_model" not in state:
        raise ValueError(f"{args.checkpoint} is not a full self-play checkpoint with EMA weights")
    atomic_checkpoint(args.output, dict(model=state["ema_model"]))
    print(f"wrote {args.output}: EMA of {args.checkpoint}")


if __name__ == "__main__":
    main()
