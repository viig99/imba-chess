"""Bounded CPU/CUDA trace of real, sequential Gumbel searches across games."""

import argparse
from dataclasses import replace
from pathlib import Path
import random
import torch
from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.self_play.benchmarks import histories
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime, run_lock
from imba_chess.self_play.seeds import load_seeds, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    config = replace(cfg.search, simulations=args.simulations)
    seeds = load_seeds(args.seeds)[: args.games]
    with run_lock(args.output):
        runtime, _ = load_runtime(cfg, args.checkpoint, "cuda")
        actor = file_hash(args.checkpoint)

        def execute():
            def factory():
                for seed, board, history in histories(seeds, runtime):
                    yield (
                        seed.seed_id,
                        runtime.search(
                            board=board,
                            history=history,
                            actor_id=actor,
                            game_id=seed.seed_id,
                            config=config,
                            rng=random.Random(seed.seed_id),
                            should_stop=lambda: False,
                        ),
                    )

            def labeled(kind, fn):
                def call(payloads):
                    with torch.profiler.record_function(kind):
                        return fn(payloads)

                return call

            results, errors = ([], [])
            BatchScheduler(
                game_factory=iter(factory()),
                concurrent_games=args.games,
                executors={k: labeled(k, v) for k, v in runtime.executors.items()},
                completion_order=True,
                on_game_done=lambda gid, r: results.append(r),
                on_game_error=lambda gid, e: errors.append(str(e)),
            ).run()
            assert not errors, errors
            assert len(results) == len(seeds)

        execute()
        execute()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.profiler.record_function("gumbel_searches"):
                execute()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / "trace.json"))
        events = prof.key_averages()
        (args.output / "cpu.txt").write_text(
            events.table(sort_by="self_cpu_time_total", row_limit=35)
        )
        (args.output / "cuda.txt").write_text(
            events.table(sort_by="self_device_time_total", row_limit=35)
        )
        atomic_json(
            args.output / "metadata.json",
            dict(
                checkpoint=actor,
                config=cfg.identifier,
                seeds=[s.seed_id for s in seeds],
                simulations=args.simulations,
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
            ),
        )
        print((args.output / "cpu.txt").read_text())


if __name__ == "__main__":
    main()
