"""Measure KLENT self-play and training throughput, and project a position budget.

Runs the real pipeline (KlentRun.collect, then KlentTrainer.train_epoch) for a
short burst per slot count, reports where self-play time goes (CPU prepare,
GPU decode + sampling, CPU apply) and the projected hours for --budget
positions (one fitting epoch per position, as in training).
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from imba_chess.klent.config import load_klent_config
from imba_chess.klent.run import KlentRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--slots", type=int, nargs="+", default=[256])
    parser.add_argument("--steps", type=int, default=600,
                        help="Decode steps per slot; long enough to reach long contexts.")
    parser.add_argument("--train-games", type=int, default=400)
    parser.add_argument("--budget", type=float, default=75e6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    base = load_klent_config(args.config)
    for slots in args.slots:
        cfg = replace(base, slots=slots)
        run = KlentRun(cfg, device=args.device)
        run._phase()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        games, metrics = run.collect(slots * args.steps, bootstrap=cfg.bootstrap)
        seconds = time.perf_counter() - start
        selfplay_peak = torch.cuda.max_memory_allocated() / 2**30
        positions_per_second = metrics["selfplay/positions_played"] / seconds
        result = dict(
            slots=slots,
            positions_per_second=round(positions_per_second),
            selfplay_peak_gib=round(selfplay_peak, 2),
            mean_plies=round(metrics["selfplay/mean_plies"], 1),
            games=metrics["selfplay/games"],
            **{k: round(v / seconds, 3) for k, v in metrics.items() if k.startswith("time/")},
        )
        if games:
            torch.cuda.reset_peak_memory_stats()
            sample = games[: args.train_games]
            rng = np.random.default_rng(0)
            # First pass compiles; time the second.
            run.trainer.train_epoch(sample, rng, policy_weight=1.0, value_weight=cfg.value_weight)
            train = run.trainer.train_epoch(sample, rng, policy_weight=1.0,
                                            value_weight=cfg.value_weight)
            tokens_per_position = train["train/tokens"] / sum(len(g["move_id"]) for g in sample)  # per supervised position
            train_positions_per_second = train["train/tokens_per_second"] / tokens_per_position
            hours_selfplay = args.budget / positions_per_second / 3600
            hours_train = args.budget / train_positions_per_second / 3600
            result.update(
                train_tokens_per_second=round(train["train/tokens_per_second"]),
                train_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                projected_hours_selfplay=round(hours_selfplay, 2),
                projected_hours_train=round(hours_train, 2),
                projected_hours_total=round(hours_selfplay + hours_train, 2),
            )
        print(json.dumps(result), flush=True)
        del run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
