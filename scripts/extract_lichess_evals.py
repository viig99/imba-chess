#!/usr/bin/env python3
"""Extract Lichess's Stockfish `[%eval]` annotations as per-ply value targets.

Roughly 13% of the games in this corpus carry a server-side Stockfish
evaluation on EVERY ply (measured: 13.3% of games, 13.05% of all plies, and
within an annotated game the median is 100% of plies). That is ~10x the ply
coverage of the search rollouts and costs no GPU at all, which makes it the
natural teacher for the value head -- see docs/GENERATION_PERF_HANDOFF.md and
the value-plateau finding (value_loss fell 1.7% over the last 75% of training
while policy_loss fell 8.5%).

Two alignment facts this script exists to get right, either of which silently
corrupts every downstream number if missed:

1. **The one-ply shift.** PGN puts a comment AFTER the move it follows, so
   `1. Nf3 { [%eval 0.14] }` is the evaluation of the position *after* Nf3.
   LichessDataset._extract_plays stores `plays[i]["state"]` as the position
   *before* move i. So the eval trailing ply i describes the position before
   ply i+1, and must be keyed `ply = i + 1` to line up with the
   `(game_id, ply)` convention rollout_store/event_builder already use.

2. **Perspective.** Lichess evals are from White's point of view; the value
   head is side-to-move. The position before ply j has White to move iff j is
   even, so the sign flips on odd plies.

Ply enumeration mirrors `_extract_plays` exactly -- same `chess.pgn` mainline
walk, same `max_seq_len` truncation -- so ply indices are directly comparable.
Mate scores are kept separately rather than clipped into the centipawn scale;
the calibration step decides what to do with them.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.stockfish_evals import (
    RESULT_TO_WHITE_SCORE as _RESULT_TO_WHITE_SCORE,
    eval_from_comment as _eval_from_comment,
)


def extract_game(
    movetext: str, game_id: str, result: str, max_seq_len: int | None
) -> list[dict]:
    game = chess.pgn.read_game(io.StringIO(movetext))
    # Same rejection as LichessDataset._parse_row: a mid-parse break leaves a
    # truncated prefix carrying a result it never reached.
    if game is None or game.errors:
        return []
    white_score = _RESULT_TO_WHITE_SCORE.get(result)
    if white_score is None:
        return []

    # Walk the mainline exactly as _extract_plays does, recording each move's
    # trailing eval against the NEXT ply index (see module docstring).
    trailing: list[tuple[float | None, int | None]] = []
    node = game
    n_plies = 0
    while node.variations and (max_seq_len is None or n_plies < max_seq_len):
        node = node.variations[0]
        trailing.append(_eval_from_comment(node.comment or ""))
        n_plies += 1

    rows: list[dict] = []
    for i, (cp, mate) in enumerate(trailing):
        if cp is None and mate is None:
            continue
        ply = i + 1
        if ply >= n_plies:
            # The final position has no ply of its own to attach to.
            continue
        stm_is_white = ply % 2 == 0
        sign = 1.0 if stm_is_white else -1.0
        rows.append(
            {
                "game_id": game_id,
                "ply": ply,
                "cp_white": cp,
                "mate_white": mate,
                "stm_is_white": stm_is_white,
                "cp_stm": None if cp is None else cp * sign,
                "mate_stm": None if mate is None else int(mate * sign),
                "real_outcome_stm": int(white_score * sign),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="parquet from scripts/materialize_corpus.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    cfg = load_repo_config(args.config).dataset
    handle = pq.ParquetFile(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")

    writer: pq.ParquetWriter | None = None
    buf: list[dict] = []
    games = annotated = rows_out = 0
    t0 = time.perf_counter()

    def flush() -> None:
        nonlocal writer, buf
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
        buf = []

    for batch in handle.iter_batches(batch_size=cfg.parquet_batch_size):
        for row in batch.to_pylist():
            games += 1
            rows = extract_game(
                str(row.get("movetext") or ""),
                str(row.get("Site") or ""),
                str(row.get("Result") or ""),
                cfg.max_seq_len,
            )
            if rows:
                annotated += 1
                rows_out += len(rows)
                buf.extend(rows)
            if len(buf) >= 50_000:
                flush()
            if args.max_games and games >= args.max_games:
                break
        if args.max_games and games >= args.max_games:
            break
    flush()
    if writer is not None:
        writer.close()
        os.replace(tmp, args.output)

    dt = time.perf_counter() - t0
    print(f"games scanned    : {games}")
    print(f"games annotated  : {annotated} ({annotated / max(1, games):.1%})")
    print(f"eval rows written: {rows_out}  -> {args.output}  ({dt:.1f}s)")
    # Same unconditional hard exit as the other drivers (590838a): the parquet
    # is written and renamed above, and interpreter finalization here has hit
    # `PyGILState_Release: thread state must be current`. os._exit skips
    # stdout flushing, so flush explicitly or the report above is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
