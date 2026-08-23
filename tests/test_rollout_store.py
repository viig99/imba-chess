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


# ── external teachers are exempt from the checkpoint-consistency rule ────────

def _teacher_row(checkpoint: str, ply: int = 0):
    from imba_chess.data.rollout_store import RolloutRow
    return RolloutRow(
        game_id="g", ply=ply, human_move_uci="", human_move_backed_value=None,
        real_outcome_stm=0, best_arm_move_uci="", best_arm_backed_value=0.0,
        root_wdl_unsearched=(0.3, 0.4, 0.3), arm_move_uci=(), arm_backed_value=(),
        arm_evals_spent=(), arm_log_prior=(), search_budget=0, search_top_m=0,
        search_max_depth=0, checkpoint=checkpoint,
    )


def test_external_teacher_rows_skip_the_checkpoint_check(tmp_path):
    """Lichess Stockfish evals were produced by no checkpoint at all, so there
    is nothing for them to be consistent or inconsistent with."""
    from imba_chess.data.rollout_store import (
        EXTERNAL_TEACHER_PREFIX,
        assert_rollout_checkpoint_consistency,
    )
    lookup = {("g", 0): _teacher_row(f"{EXTERNAL_TEACHER_PREFIX}lichess-stockfish-eval")}
    ckpt = tmp_path / "some_checkpoint.pt"
    ckpt.write_bytes(b"")
    assert_rollout_checkpoint_consistency(lookup, ckpt)      # any checkpoint
    assert_rollout_checkpoint_consistency(lookup, None)      # or none at all


def test_a_real_rollout_still_has_to_match(tmp_path):
    """The exemption must not weaken the rule it sits next to."""
    import pytest as _pytest
    from imba_chess.data.rollout_store import assert_rollout_checkpoint_consistency
    other = tmp_path / "other.pt"; other.write_bytes(b"")
    resume = tmp_path / "resume.pt"; resume.write_bytes(b"")
    with _pytest.raises(ValueError, match="checkpoint mismatch"):
        assert_rollout_checkpoint_consistency({("g", 0): _teacher_row(str(other))}, resume)
    with _pytest.raises(ValueError, match="no --resume checkpoint"):
        assert_rollout_checkpoint_consistency({("g", 0): _teacher_row(str(other))}, None)


def test_mixing_external_and_checkpoint_rows_still_validates_the_checkpoint_ones(tmp_path):
    import pytest as _pytest
    from imba_chess.data.rollout_store import (
        EXTERNAL_TEACHER_PREFIX,
        assert_rollout_checkpoint_consistency,
    )
    other = tmp_path / "other.pt"; other.write_bytes(b"")
    resume = tmp_path / "resume.pt"; resume.write_bytes(b"")
    lookup = {
        ("g", 0): _teacher_row(f"{EXTERNAL_TEACHER_PREFIX}lichess-stockfish-eval", 0),
        ("g", 1): _teacher_row(str(other), 1),
    }
    with _pytest.raises(ValueError, match="checkpoint mismatch"):
        assert_rollout_checkpoint_consistency(lookup, resume)
