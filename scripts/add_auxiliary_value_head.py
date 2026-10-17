"""Turn a stage-1 checkpoint into a stage-2 initialization with zeroed auxiliary value heads.

The stage-2 recipe (config/self_play.toml) trains three auxiliary TD value horizons, so its model
config (config/imba_chess_v4_aux3.toml) has an auxiliary head that stage-1 checkpoints lack, and
checkpoint loading is strict. This writes the stage-1 weights plus that head, zero-initialized,
as a weights-only checkpoint for run_self_play.py --initialize. The source is not modified.
Applied to the flattened ckpt34 checkpoint at step 53250 it reproduces
artifacts/checkpoints_keep/rl_initial_53250_aux3.pt, the initialization of the 2026-10 lineage.
"""

import argparse
from pathlib import Path
import torch
from imba_chess.config import load_repo_config
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.model import HSTUChessModel, build_hstu_chess_config
from imba_chess.model.checkpoint import load_initial_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="stage-1 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("config/imba_chess_v4_aux3.toml"),
                        help="the base_config of the stage-2 config")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = load_repo_config(args.model_config)
    model_config = build_hstu_chess_config(repo.model, move_vocab_size=len(MoveVocab.load(repo.vocab.path)))
    if not model_config.enable_auxiliary_value_head:
        raise ValueError(f"{args.model_config} has no auxiliary value head")
    model = HSTUChessModel(model_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_initial_weights(model, checkpoint, allow_missing_prefixes=("auxiliary_value_head.",))
    head = model.auxiliary_value_head
    if head.weight.count_nonzero() or head.bias.count_nonzero():
        raise ValueError("checkpoint already has auxiliary value head weights")
    torch.save({"model": model.state_dict()}, args.output)
    print(f"wrote {args.output}: auxiliary head {tuple(head.weight.shape)} zeroed")


if __name__ == "__main__":
    main()
