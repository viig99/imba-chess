"""Measure the shared human-game input path without running GPU self-play.

Example:
  .venv/bin/python scripts/bench_self_play_input.py --source remote \
    --output artifacts/self_play_validation/input_stream/remote.json

The local mode reads an already filtered corpus. Remote mode retains the
training configuration's month order, column projection, filters and shuffle.
Parsing repetitions reuse identical captured rows; they exclude network time.
This is an input-capacity benchmark, not a benchmark of a finished start sampler.
"""

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import resource
import statistics
import time

import pyarrow.parquet as pq

from imba_chess.config import load_repo_config
from imba_chess.data.lichess_dataset import LichessDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/imba_chess_v4.toml")
    )
    parser.add_argument("--source", choices=("local", "remote"), required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus/v4_self_play_train_4096.parquet"),
    )
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 2 or args.repeats < 1:
        parser.error("rows must be >= 2 and repeats must be positive")
    cfg = asdict(load_repo_config(args.config).dataset)
    accepted = inspect.signature(LichessDataset).parameters
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    kwargs.update(split="train", local_corpus_path=None, parse_stockfish_evals=False)
    dataset = LichessDataset(**kwargs)
    result = dict(
        source=args.source,
        dataset=kwargs,
        requested_rows=args.rows,
        corpus=str(args.corpus) if args.source == "local" else None,
        timing_scope="after Python imports; one process; no GPU",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(event, **values):
        result.update(values)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(event=event, **values)), flush=True)

    save("starting", status="running")
    start = time.perf_counter()
    try:
        if args.source == "remote":
            rows, prefiltered = dataset.filtered_shuffled_rows()
            if rows is None:
                raise RuntimeError("empty remote stream")
        else:
            handle = pq.ParquetFile(args.corpus)

            def local_rows():
                for batch in handle.iter_batches(batch_size=dataset.parquet_batch_size):
                    yield from batch.to_pylist()

            rows, prefiltered = local_rows(), True
        setup = time.perf_counter() - start
        save("resolved", setup_seconds=setup, prefiltered=prefiltered)
        iterator, captured, waits = iter(rows), [], []
        first = None
        for i in range(args.rows):
            before = time.perf_counter()
            try:
                row = next(iterator)
            except StopIteration:
                break
            waits.append(time.perf_counter() - before)
            captured.append(row)
            if first is None:
                first = time.perf_counter() - start
                save("first_row", first_row_seconds=first)
            if (i + 1) % 256 == 0:
                print(
                    json.dumps(
                        dict(
                            event="rows",
                            count=i + 1,
                            elapsed_seconds=time.perf_counter() - start,
                        )
                    ),
                    flush=True,
                )
        elapsed = time.perf_counter() - start
        if len(captured) < 2:
            raise RuntimeError("fewer than two source rows")
        save(
            "read",
            rows=len(captured),
            input_seconds=elapsed,
            raw_rows_per_second_after_first=(len(captured) - 1) / sum(waits[1:]),
            max_next_row_seconds=max(waits),
            raw_unique_sources=len({r.get("Site") for r in captured}),
        )
        parsing = []
        for repeat in range(args.repeats):
            before = time.perf_counter()
            games, plies, eligible = 0, 0, [0, 0, 0]
            for game in dataset.stream_from_rows(
                captured, assume_prefiltered=prefiltered
            ):
                games += 1
                length = len(game["plays"])
                plies += length
                # Length coverage only, not terminal/draw eligibility or seed selection.
                for j, lower in enumerate((1, 31, 71)):
                    eligible[j] += length > lower
            seconds = time.perf_counter() - before
            parsing.append(
                dict(
                    repeat=repeat,
                    games=games,
                    plies=plies,
                    seconds=seconds,
                    games_per_second=games / seconds,
                    games_reaching_bucket_lower_bounds=eligible,
                )
            )
            print(json.dumps(dict(event="parsed", **parsing[-1])), flush=True)
        save(
            "complete",
            status="complete",
            parsing=parsing,
            median_parsed_games_per_second=statistics.median(
                p["games_per_second"] for p in parsing
            ),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
    except Exception as exc:
        save(
            "error",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - start,
        )
        raise


if __name__ == "__main__":
    main()
