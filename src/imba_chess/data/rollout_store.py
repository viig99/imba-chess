from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class RolloutRow:
    game_id: str
    ply: int
    human_move_uci: str
    human_move_backed_value: float | None
    real_outcome_stm: int
    best_arm_move_uci: str
    best_arm_backed_value: float
    root_wdl_unsearched: tuple[float, float, float]
    arm_move_uci: tuple[str, ...]
    arm_backed_value: tuple[float, ...]
    arm_evals_spent: tuple[int, ...]
    arm_log_prior: tuple[float, ...]
    search_budget: int
    search_top_m: int
    search_max_depth: int
    checkpoint: str
    search_refutation_top_r: int = 2
    search_expand_top: int = 3
    search_lam: float = 0.05


_ROLLOUT_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string()),
        pa.field("ply", pa.int64()),
        pa.field("human_move_uci", pa.string()),
        pa.field("human_move_backed_value", pa.float64()),
        pa.field("real_outcome_stm", pa.int64()),
        pa.field("best_arm_move_uci", pa.string()),
        pa.field("best_arm_backed_value", pa.float64()),
        pa.field("root_wdl_unsearched", pa.list_(pa.float64())),
        pa.field("arm_move_uci", pa.list_(pa.string())),
        pa.field("arm_backed_value", pa.list_(pa.float64())),
        pa.field("arm_evals_spent", pa.list_(pa.int64())),
        pa.field("arm_log_prior", pa.list_(pa.float64())),
        pa.field("search_budget", pa.int64()),
        pa.field("search_top_m", pa.int64()),
        pa.field("search_max_depth", pa.int64()),
        pa.field("checkpoint", pa.string()),
        pa.field("search_refutation_top_r", pa.int64()),
        pa.field("search_expand_top", pa.int64()),
        pa.field("search_lam", pa.float64()),
    ]
)


def _row_to_record(row: RolloutRow) -> dict:
    return {
        "game_id": row.game_id,
        "ply": row.ply,
        "human_move_uci": row.human_move_uci,
        "human_move_backed_value": row.human_move_backed_value,
        "real_outcome_stm": row.real_outcome_stm,
        "best_arm_move_uci": row.best_arm_move_uci,
        "best_arm_backed_value": row.best_arm_backed_value,
        "root_wdl_unsearched": list(row.root_wdl_unsearched),
        "arm_move_uci": list(row.arm_move_uci),
        "arm_backed_value": list(row.arm_backed_value),
        "arm_evals_spent": list(row.arm_evals_spent),
        "arm_log_prior": list(row.arm_log_prior),
        "search_budget": row.search_budget,
        "search_top_m": row.search_top_m,
        "search_max_depth": row.search_max_depth,
        "checkpoint": row.checkpoint,
        "search_refutation_top_r": row.search_refutation_top_r,
        "search_expand_top": row.search_expand_top,
        "search_lam": row.search_lam,
    }


def write_rollout_parquet(rows: list[RolloutRow], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_row_to_record(row) for row in rows], schema=_ROLLOUT_SCHEMA)
    # Write to a sibling temp file then atomically rename into place, so a
    # process killed mid-write (e.g. a scheduled overnight stop) can never
    # leave a truncated/corrupt file at `output_path` -- callers always see
    # either the previous complete write or the new one, never a partial one.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, output_path)


def load_rollout_lookup(path: str | Path) -> dict[tuple[str, int], RolloutRow]:
    table = pq.read_table(path)
    lookup: dict[tuple[str, int], RolloutRow] = {}
    for record in table.to_pylist():
        row = RolloutRow(
            game_id=record["game_id"],
            ply=int(record["ply"]),
            human_move_uci=record["human_move_uci"],
            human_move_backed_value=(
                float(record["human_move_backed_value"])
                if record["human_move_backed_value"] is not None
                else None
            ),
            real_outcome_stm=int(record["real_outcome_stm"]),
            best_arm_move_uci=record["best_arm_move_uci"],
            best_arm_backed_value=float(record["best_arm_backed_value"]),
            root_wdl_unsearched=tuple(float(v) for v in record["root_wdl_unsearched"]),
            arm_move_uci=tuple(record["arm_move_uci"]),
            arm_backed_value=tuple(float(v) for v in record["arm_backed_value"]),
            arm_evals_spent=tuple(int(v) for v in record["arm_evals_spent"]),
            arm_log_prior=tuple(float(v) for v in record["arm_log_prior"]),
            search_budget=int(record["search_budget"]),
            search_top_m=int(record["search_top_m"]),
            search_max_depth=int(record["search_max_depth"]),
            checkpoint=record["checkpoint"],
            search_refutation_top_r=int(record["search_refutation_top_r"]),
            search_expand_top=int(record["search_expand_top"]),
            search_lam=float(record["search_lam"]),
        )
        lookup[(row.game_id, row.ply)] = row
    return lookup
