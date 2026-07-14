import pytest

pytest.importorskip("pyarrow")

from imba_chess.data.rollout_store import (
    RolloutRow,
    assert_rollout_checkpoint_consistency,
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
        search_refutation_top_r=2,
        search_expand_top=3,
        search_lam=0.05,
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


def test_assert_rollout_checkpoint_consistency_noop_on_empty_lookup():
    assert_rollout_checkpoint_consistency({}, resume_checkpoint=None) is None
    assert_rollout_checkpoint_consistency({}, resume_checkpoint="anything.pt") is None


def test_assert_rollout_checkpoint_consistency_passes_on_matching_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint_23.pt"
    checkpoint_path.write_text("dummy")
    row = _row("g1", 0)
    row = RolloutRow(**{**row.__dict__, "checkpoint": str(checkpoint_path)})
    lookup = {("g1", 0): row}

    # No exception, and resolving via a different relative/absolute spelling
    # of the same file still matches.
    assert_rollout_checkpoint_consistency(lookup, resume_checkpoint=checkpoint_path)
    assert_rollout_checkpoint_consistency(lookup, resume_checkpoint=str(checkpoint_path))


def test_assert_rollout_checkpoint_consistency_raises_on_mismatch(tmp_path):
    checkpoint_a = tmp_path / "checkpoint_a.pt"
    checkpoint_b = tmp_path / "checkpoint_b.pt"
    row = _row("g1", 0)
    row = RolloutRow(**{**row.__dict__, "checkpoint": str(checkpoint_a)})
    lookup = {("g1", 0): row}

    with pytest.raises(ValueError, match="Rollout checkpoint mismatch"):
        assert_rollout_checkpoint_consistency(lookup, resume_checkpoint=checkpoint_b)


def test_assert_rollout_checkpoint_consistency_raises_when_resume_missing(tmp_path):
    checkpoint_path = tmp_path / "checkpoint_23.pt"
    row = _row("g1", 0)
    row = RolloutRow(**{**row.__dict__, "checkpoint": str(checkpoint_path)})
    lookup = {("g1", 0): row}

    with pytest.raises(ValueError, match="no --resume checkpoint"):
        assert_rollout_checkpoint_consistency(lookup, resume_checkpoint=None)
