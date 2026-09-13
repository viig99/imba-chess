"""Initialize stage 2 from weights or resume its full training state."""

import argparse
from dataclasses import asdict
from pathlib import Path
from imba_chess.data.self_play_store import SelfPlayStore
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock, StopBudget
from imba_chess.self_play.trainer import Stage2Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exposures", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.exposures < 1 or args.seconds <= 0:
        parser.error("exposures and seconds must be positive")
    cfg = load_config(args.config)
    with run_lock(args.output.parent), StopBudget(seconds=args.seconds) as stop:
        runtime, positions = load_runtime(
            cfg, args.resume or args.initialize, args.device
        )
        store = SelfPlayStore(args.replay, read_only=True, **asdict(cfg.replay))
        trainer = Stage2Trainer(
            model=runtime.model,
            config=cfg.learning,
            move_vocab=runtime.move_vocab,
            encoder=runtime.encoder,
            device=runtime.device,
            max_positions=positions,
            run_seed=cfg.run.seed,
        )
        if args.resume:
            trainer.resume(args.resume, store=store, config_id=cfg.identifier)
        else:
            trainer.begin_phase(store)
        trainer.train(
            store, exposure_budget=args.exposures, should_stop=stop.stop, on_step=print
        )
        trainer.checkpoint(
            args.output,
            progress=dict(phase="train", exposure_budget=args.exposures),
            store=store,
            config_id=cfg.identifier,
        )


if __name__ == "__main__":
    main()
