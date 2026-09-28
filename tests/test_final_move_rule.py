import pytest

from imba_chess.eval.gumbel_search import final_move_index


def test_gumbel_rule_keeps_search_choice():
    assert final_move_index([3, 5], [0.1, 0.2], "gumbel") is None


def test_most_visited_breaks_ties_by_mean_q_and_skips_unvisited():
    assert final_move_index([4, 9, 0], [0.9, -0.2, 1.0], "most_visited") == 1
    assert final_move_index([9, 9, 0], [-0.2, 0.3, 1.0], "most_visited") == 1


def test_lcb_prefers_well_visited_move_when_means_are_close():
    # 0.10 - 1.96*sqrt(0.99/100) = -0.095 beats 0.15 - 1.96*sqrt(0.9775/20) = -0.283.
    assert final_move_index([100, 20], [0.10, 0.15], "lcb") == 0


def test_lcb_overrides_visits_for_a_clearly_better_move():
    assert final_move_index([100, 40], [-0.5, 0.5], "lcb") == 1


def test_lcb_ignores_children_below_the_visit_floor():
    assert final_move_index([100, 10], [0.0, 0.99], "lcb") == 0
    assert final_move_index([100, 10], [0.0, 0.99], "lcb", lcb_min_visit_prop=0.1) == 1


def test_lcb_with_zero_z_is_best_mean_among_eligible():
    assert final_move_index([50, 30, 5], [0.1, 0.3, 0.9], "lcb", lcb_z=0.0) == 1


def test_lcb_bound_uses_bhatia_davis_sd():
    # Equal visits: 0.9 has sd 0.436 -> 0.9 - 2*0.436/5 = 0.726; 0.8 has sd 0.6
    # -> 0.8 - 2*0.6/5 = 0.56. A certain draw-free win (q=1) has sd 0 and wins.
    assert final_move_index([25, 25], [0.8, 0.9], "lcb", lcb_z=2.0) == 1
    assert final_move_index([25, 25], [0.95, 1.0], "lcb", lcb_z=50.0) == 1


def test_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError):
        final_move_index([1], [0.0], "best")
    with pytest.raises(ValueError):
        final_move_index([0, 0], [0.0, 0.0], "most_visited")
    with pytest.raises(ValueError):
        final_move_index([1, 2], [0.0], "lcb")
    with pytest.raises(ValueError):
        final_move_index([1, 2], [0.0, 0.0], "lcb", lcb_z=-1.0)
