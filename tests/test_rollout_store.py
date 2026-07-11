import pytest

pytest.importorskip("pyarrow")

from imba_chess.data.rollout_store import (
    RolloutRow,
    load_rollout_lookup,
    write_rollout_parquet,
)


def _row(game_id: str, ply: int) -> RolloutRow:
    return RolloutRow(
        game_id=game_id,
        ply=ply,
        human_move_uci="e2e4",
        human_move_backed_value=0.1,
        real_outcome_stm=1,
        best_arm_move_uci="d2d4",
        best_arm_backed_value=0.3,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=("d2d4", "e2e4", "", ""),
        arm_backed_value=(0.3, 0.1, 0.0, 0.0),
        arm_evals_spent=(120, 80, 0, 0),
        arm_log_prior=(-0.5, -0.7, 0.0, 0.0),
        search_budget=2048,
        search_top_m=4,
        search_max_depth=8,
        checkpoint="artifacts/checkpoints/best_hr10_checkpoint_23.pt",
    )


def test_write_then_load_round_trips(tmp_path):
    path = tmp_path / "rollouts.parquet"
    rows = [_row("g1", 3), _row("g1", 7), _row("g2", 0)]

    write_rollout_parquet(rows, path)
    lookup = load_rollout_lookup(path)

    assert set(lookup.keys()) == {("g1", 3), ("g1", 7), ("g2", 0)}
    restored = lookup[("g1", 3)]
    assert restored == rows[0]


def test_load_handles_null_human_move_backed_value(tmp_path):
    path = tmp_path / "rollouts.parquet"
    row = _row("g1", 0)
    row_with_null = RolloutRow(**{**row.__dict__, "human_move_backed_value": None})

    write_rollout_parquet([row_with_null], path)
    lookup = load_rollout_lookup(path)

    assert lookup[("g1", 0)].human_move_backed_value is None
