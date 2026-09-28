#!/usr/bin/env python
"""Head-to-head between two self-play arms, one of which may use a frozen evaluator.

Two separate reference matches carry both arms' sampling error; playing the arms
against each other makes one game one paired observation. This reuses the
production paired-opening protocol in `evaluate_pair_checkpoints` (colour
reversal, restartable progress, protocol-failure detection) rather than
reimplementing it, and composes the frozen evaluator exactly as the run did.
"""

import argparse
import json
from pathlib import Path

from imba_chess.eval.composed_runtime import ComposedRuntime
from imba_chess.self_play.config import load_config
from imba_chess.self_play.evaluation import evaluate_pair_checkpoints
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import file_hash, load_seeds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-evaluator", type=Path,
                   help="Frozen value network the candidate played with, if any")
    p.add_argument("--opponent", type=Path, required=True)
    p.add_argument("--opponent-evaluator", type=Path)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = [s for s in load_seeds(args.seeds) if s.split == "monitor"]
    if len(seeds) < args.pairs:
        p.error(f"need {args.pairs} monitor prefixes, have {len(seeds)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with run_lock(args.output.parent):
        limits, runtimes = set(), {}
        for role, ckpt, evaluator in (
            ("candidate", args.candidate, args.candidate_evaluator),
            ("opponent", args.opponent, args.opponent_evaluator),
        ):
            runtime, maximum = load_runtime(cfg, ckpt, args.device)
            limits.add(maximum)
            if evaluator is not None:
                frozen, frozen_max = load_runtime(cfg, evaluator, args.device)
                limits.add(frozen_max)
                runtime = ComposedRuntime(runtime, frozen)
            runtimes[role] = runtime
        if len(limits) != 1:
            raise ValueError("checkpoint context limits differ")

        identity = {
            role: file_hash(ckpt) + (f"+{file_hash(ev)}" if ev else "")
            for role, ckpt, ev in (
                ("candidate", args.candidate, args.candidate_evaluator),
                ("opponent", args.opponent, args.opponent_evaluator),
            )
        }
        interval = evaluate_pair_checkpoints(
            candidate=runtimes["candidate"],
            best=runtimes["opponent"],
            candidate_id=identity["candidate"],
            best_id=identity["opponent"],
            seeds=seeds,
            config=cfg,
            max_positions=limits.pop(),
            output=args.output,
            pairs=args.pairs,
        )
        print(json.dumps(dict(identity=identity, interval=interval), indent=2))


if __name__ == "__main__":
    main()
