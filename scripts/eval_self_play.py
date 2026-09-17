"""Restartable Gumbel checkpoint pairs or independent Stockfish evaluation."""

import argparse
import json
from pathlib import Path
from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.config import load_config
from imba_chess.self_play.evaluation import (
    evaluate_pair_checkpoints,
    EvaluationProtocolError,
)
from imba_chess.self_play.runtime import load_runtime, run_lock, StopBudget
from imba_chess.self_play.seeds import (
    load_seeds,
    file_hash,
    Seed,
    source_split,
    stable_hash,
)
from imba_chess.self_play.stockfish import StockfishRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--best", type=Path)
    opponent.add_argument("--stockfish", type=Path)
    starts = parser.add_mutually_exclusive_group(required=True)
    starts.add_argument("--seeds", type=Path)
    starts.add_argument("--initial-board", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.pairs < 1 or args.seconds <= 0:
        parser.error("bounds must be positive")
    if args.initial_board:
        if not args.stockfish:
            parser.error(
                "initial-board mode is reserved for the independent Stockfish protocol"
            )
        seeds = []
        for i in range(args.pairs):
            source = f"initial-eval-{i}"
            while source_split(source) != "monitor":
                source += "x"
            seeds.append(
                Seed(stable_hash(source), source, [], 0, "monitor", "initial-board")
            )
    else:
        seeds = load_seeds(args.seeds, "monitor")
    with (
        run_lock(args.output.parent),
        StopBudget(seconds=args.seconds, hard_exit=True) as stop,
    ):
        runtime, positions = load_runtime(cfg, args.checkpoint, args.device)
        if args.stockfish:
            best = StockfishRuntime(runtime, path=str(args.stockfish))
            best_id = stable_hash(json.dumps(best.protocol, sort_keys=True))
            atomic_json(args.output.with_suffix(".protocol.json"), best.protocol)
        else:
            best, _ = load_runtime(cfg, args.best, args.device)
            best_id = file_hash(args.best)
        try:
            result = evaluate_pair_checkpoints(
                candidate=runtime,
                best=best,
                candidate_id=file_hash(args.checkpoint),
                best_id=best_id,
                seeds=seeds,
                config=cfg,
                max_positions=positions,
                output=args.output,
                pairs=args.pairs,
                should_stop=stop.stop,
            )
            print(
                result
                if result is not None
                else "Evaluation unfinished; rerun the same command to resume."
            )
        except EvaluationProtocolError as exc:
            raise SystemExit(str(exc)) from exc
        finally:
            if args.stockfish:
                best.close()


if __name__ == "__main__":
    main()
