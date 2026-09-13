"""Collect bounded, outcome-labeled Gumbel continuations from prepared seeds."""

import argparse
from dataclasses import asdict
from pathlib import Path
from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.self_play.collector import collect
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock, StopBudget
from imba_chess.self_play.seeds import load_seeds, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--decoder-mode",
        choices=["current", "compiled"],
        default=None,
        help="Default: compiled on CUDA, current (eager) on CPU",
    )
    args = parser.parse_args()
    if args.decoder_mode == "compiled" and args.device.split(":")[0] != "cuda":
        parser.error("compiled decoding requires --device cuda")
    if args.games < 1 or args.seconds <= 0:
        parser.error("games and seconds must be positive")
    cfg = load_config(args.config)
    with run_lock(args.output), StopBudget(seconds=args.seconds) as stop:
        runtime, positions = load_runtime(
            cfg,
            args.checkpoint,
            args.device,
            **(
                {"decoder_mode": args.decoder_mode}
                if args.decoder_mode is not None
                else {}
            ),
        )
        store = SelfPlayStore(args.output / "replay", **asdict(cfg.replay))
        metrics = collect(
            seeds=load_seeds(args.seeds),
            runtime=runtime,
            config=cfg,
            actor_id=file_hash(args.checkpoint),
            store=store,
            max_positions=positions,
            game_count=args.games,
            should_launch=stop.launch,
            should_stop=stop.stop,
            skip_ids=store.seen,
        )
        report = metrics.report()
        atomic_json(args.output / "metrics.json", report)
        print(report)


if __name__ == "__main__":
    main()
