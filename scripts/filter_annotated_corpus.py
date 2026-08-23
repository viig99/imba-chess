#!/usr/bin/env python3
"""Keep only the games that carry Stockfish `[%eval]` annotations.

Phase B trains two arms that differ ONLY in the value target. Restricting both
to annotated games makes the distill arm's coverage 100% of plies instead of
~13%, which is the maximum-signal form of the test: if a Stockfish-derived
value target cannot move the head when every single target is one, it will not
move it diluted.

The restriction is not free -- annotation is strongly self-selected by rating
(9.6% of games at 2000-2199 average Elo, 64.3% at 2600-2799), so this subset
is a stronger population than the full stream. That is exactly why the control
arm must train on THIS SAME file with beta=0: both arms then see the identical
(stronger) games and the only difference left is the target.

Row order is preserved, so a corpus filtered from a seed-pinned materialization
still replays in stream order.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk-rows", type=int, default=20_000)
    args = ap.parse_args()

    handle = pq.ParquetFile(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    buf: list[dict] = []
    seen = kept = 0
    t0 = time.perf_counter()

    def flush() -> None:
        nonlocal writer, buf
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=handle.schema_arrow)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
        buf = []

    for batch in handle.iter_batches(batch_size=args.chunk_rows):
        for row in batch.to_pylist():
            seen += 1
            if "[%eval" in (row.get("movetext") or ""):
                kept += 1
                buf.append(row)
        if len(buf) >= args.chunk_rows:
            flush()
    flush()
    if writer is None:
        raise RuntimeError(f"no annotated games found in {args.corpus}")
    writer.close()
    os.replace(tmp, args.output)

    dt = time.perf_counter() - t0
    print(f"scanned {seen} games, kept {kept} annotated ({kept / max(1, seen):.1%}) "
          f"-> {args.output} in {dt:.1f}s")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
