import pytest

from imba_chess.data.value_target_blend import compute_blended_value_target


def test_beta_zero_reproduces_real_outcome_one_hot():
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.9,
        real_outcome_stm=1,
        beta=0.0,
    )
    assert result == pytest.approx([0.0, 0.0, 1.0])


def test_beta_one_reproduces_searched_vec_when_not_clamped():
    # p_win_raw = (1 - 0.3 + 0.2) / 2 = 0.45, p_loss_raw = (1 - 0.3 - 0.2) / 2 = 0.25
    # both non-negative -> no clamping, sums to 1 exactly.
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.2,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert result == pytest.approx([0.25, 0.3, 0.45])


def test_draw_mass_preserved_when_not_clamped():
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.2,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert result[1] == pytest.approx(0.3)


def test_sums_to_one_for_intermediate_beta():
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.2,
        real_outcome_stm=-1,
        beta=0.4,
    )
    assert sum(result) == pytest.approx(1.0)


def test_monotone_in_backed_value():
    low = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=-0.1,
        real_outcome_stm=0,
        beta=1.0,
    )
    high = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.4,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert high[2] > low[2]  # p_win increases
    assert high[0] < low[0]  # p_loss decreases


def test_clamps_and_renormalizes_when_backed_value_is_extreme():
    # p_win_raw = (1 - 0.9 - 0.8) / 2 = -0.35 -> clamped to 0.
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.05, 0.9, 0.05),
        backed_value=-0.8,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert result[2] == pytest.approx(0.0)
    assert all(value >= 0.0 for value in result)
    assert sum(result) == pytest.approx(1.0)
    # Draw mass is NOT preserved in the clamped case (documents the deviation
    # from the common-case invariant tested above).
    assert result[1] != pytest.approx(0.9)


def test_beta_out_of_range_raises():
    with pytest.raises(ValueError, match="beta"):
        compute_blended_value_target(
            root_wdl_unsearched=(0.2, 0.3, 0.5),
            backed_value=0.0,
            real_outcome_stm=0,
            beta=1.5,
        )


def test_invalid_real_outcome_raises():
    with pytest.raises(ValueError, match="real_outcome_stm"):
        compute_blended_value_target(
            root_wdl_unsearched=(0.2, 0.3, 0.5),
            backed_value=0.0,
            real_outcome_stm=2,
            beta=0.5,
        )
