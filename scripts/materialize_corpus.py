"""Materialize the seeded Lichess stream to a local parquet, in stream order.

Why: `artifacts/hf_cache` is empty, so every run re-streams from HuggingFace.
That costs ~6s per run and, worse, it STALLS -- a 4-game diagnostic hung for 8
minutes with 0% CPU and all 38 threads in `futex_do_wait` on the HF socket while
huggingface.co itself answered in 0.13s. Corpus streaming now blocks
measurement, not just throughput (handoff section 8 lever 5, section 13).

Alignment safety: rows are captured from `LichessDataset.filtered_shuffled_rows()`
-- i.e. AFTER `.filter()` and AFTER `.shuffle(seed=train_month_shuffle_seed)` --
so the local file holds exactly the rows `stream()` would parse, in exactly that
order. Replaying it via `stream_local()` therefore preserves the `(game_id, ply)`
keys that join rollouts to training. This is gated by a bit-identical rollout
diff, not assumed.

Run:
  .venv/bin/python scripts/materialize_corpus.py \
    --config config/imba_chess_exit_seeded_rollout.toml \
    --output artifacts/corpus/seed42_train.parquet --max-rows 50000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config  # noqa: E402
from imba_chess.data.lichess_dataset import LichessDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=50_000)
    ap.add_argument("--chunk-rows", type=int, default=5_000)
    args = ap.parse_args()

    cfg = load_repo_config(args.config).dataset
    if cfg.train_month_shuffle_seed is None:
        raise ValueError(
            "config leaves train_month_shuffle_seed unset -- the stream order would "
            "be OS-entropy-seeded and a materialized copy would align with nothing. "
            "Use a seed-pinned config (see scripts/rollout_nightly_start.sh)."
        )

    dataset = LichessDataset(
        min_avg_elo=cfg.min_avg_elo,
        min_time_control_sec=cfg.min_time_control_sec,
        split="train",
        dataset_name=cfg.dataset_name,
        train_start_month=cfg.train_start_month,
        train_end_month=cfg.train_end_month,
        cache_dir=cfg.cache_dir,
        parquet_batch_size=cfg.parquet_batch_size,
        max_seq_len=cfg.max_seq_len,
        shuffle_train_month_files_on_start=cfg.shuffle_train_month_files_on_start,
        train_month_shuffle_seed=cfg.train_month_shuffle_seed,
        train_shuffle_buffer_size=cfg.train_shuffle_buffer_size,
    )

    rows, prefiltered = dataset.filtered_shuffled_rows()
    if rows is None:
        raise RuntimeError("no data files resolved for this config's month window")
    print(
        f"streaming (prefiltered={prefiltered}, shuffle_seed="
        f"{cfg.train_month_shuffle_seed}, buffer={cfg.train_shuffle_buffer_size}) "
        f"-> {args.output}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    buf: list[dict] = []
    written = 0
    t0 = time.perf_counter()

    def _flush() -> None:
        nonlocal writer, buf, written
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
        written += len(buf)
        rate = written / max(1e-9, time.perf_counter() - t0)
        print(f"  {written:,} rows ({rate:,.0f}/s)", flush=True)
        buf = []

    try:
        for row in rows:
            buf.append(dict(row))
            if len(buf) >= args.chunk_rows:
                _flush()
            if written >= args.max_rows:
                break
        _flush()
    finally:
        if writer is not None:
            writer.close()

    if written == 0:
        raise RuntimeError("no rows written -- refusing to leave an empty corpus")
    tmp.replace(args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"\nwrote {written:,} rows to {args.output} ({size_mb:.1f} MB) "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    print("Replay with LichessDataset.stream_local(path) / --local-corpus.")
    sys.stdout.flush()
    sys.stderr.flush()
    # Same unconditional hard exit as the other drivers (590838a): a native
    # thread touches the GIL during finalization and CPython aborts with
    # "PyGILState_Release: thread state must be current". Observed here too --
    # the parquet was already written and renamed, but the process core-dumped.
    os._exit(0)


if __name__ == "__main__":
    main()
