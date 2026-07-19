from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The module under test imports torch (via imba_chess.eval.position_evaluator
# and scripts/eval_vs_stockfish.py) at module load time, mirroring the
# existing tests/test_eval_vs_stockfish.py convention. These tests exercise
# only pure stats/rounding helpers -- no engine process, no GPU, no model
# checkpoint is touched.
pytest.importorskip("torch")


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "calibrate_stockfish_nodes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "calibrate_stockfish_nodes_script", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load calibrate_stockfish_nodes.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


# ---------------------------------------------------------------------------
# percentile / median
# ---------------------------------------------------------------------------


def test_percentile_single_element_returns_it_for_any_p(module):
    assert module.percentile([42.0], 0.0) == 42.0
    assert module.percentile([42.0], 50.0) == 42.0
    assert module.percentile([42.0], 100.0) == 42.0


def test_percentile_exact_boundary_five_elements(module):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    # rank = (5-1) * p/100 lands exactly on an index for p=25/50/75 with n=5.
    assert module.percentile(values, 25.0) == 2.0
    assert module.percentile(values, 50.0) == 3.0
    assert module.percentile(values, 75.0) == 4.0
    assert module.percentile(values, 0.0) == 1.0
    assert module.percentile(values, 100.0) == 5.0


def test_percentile_interpolates_between_two_elements(module):
    values = [1.0, 2.0, 3.0, 4.0]
    # rank = (4-1) * 0.25 = 0.75 -> interpolate between index 0 and 1.
    assert module.percentile(values, 25.0) == pytest.approx(1.75)
    # rank = 3 * 0.75 = 2.25 -> interpolate between index 2 and 3.
    assert module.percentile(values, 75.0) == pytest.approx(3.25)


def test_percentile_unsorted_input_is_sorted_first(module):
    assert module.percentile([3.0, 1.0, 2.0], 50.0) == 2.0


def test_percentile_rejects_empty_list(module):
    with pytest.raises(ValueError):
        module.percentile([], 50.0)


def test_percentile_rejects_out_of_range_p(module):
    with pytest.raises(ValueError):
        module.percentile([1.0, 2.0], -1.0)
    with pytest.raises(ValueError):
        module.percentile([1.0, 2.0], 100.1)


def test_median_matches_percentile_50_odd_and_even(module):
    assert module.median([5.0, 1.0, 3.0]) == module.percentile([5.0, 1.0, 3.0], 50.0)
    assert module.median([5.0, 1.0, 3.0]) == 3.0
    assert module.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_median_single_element(module):
    assert module.median([7.0]) == 7.0


# ---------------------------------------------------------------------------
# round_to_sig_figs
# ---------------------------------------------------------------------------


def test_round_to_sig_figs_zero(module):
    assert module.round_to_sig_figs(0, 2) == 0.0
    assert module.round_to_sig_figs(0.0, 3) == 0.0


def test_round_to_sig_figs_rounds_up_non_tie_cases(module):
    # 149500 = 1.495e5 -> 1 decimal-place round of the mantissa (1.495) is
    # not a tie (hundredths digit is 9, forcing a round-up regardless of
    # tie-breaking rule) -> 1.50e5.
    assert module.round_to_sig_figs(149500, 2) == 150000.0
    # 94999 = 9.4999e4 -> mantissa rounds up to 9.5e4 -> 95000.
    assert module.round_to_sig_figs(94999, 2) == 95000.0


def test_round_to_sig_figs_exact_tie_rounds_away_from_zero(module):
    # 125 = 1.25e2 is an exact tie between 1.2e2 and 1.3e2 at 2 sig figs;
    # ROUND_HALF_UP resolves it up to 130.
    assert module.round_to_sig_figs(125, 2) == 130.0
    # 135 = 1.35e2 is likewise an exact tie -> rounds up to 140.
    assert module.round_to_sig_figs(135, 2) == 140.0


def test_round_to_sig_figs_negative_preserves_sign(module):
    assert module.round_to_sig_figs(-149500, 2) == -150000.0
    assert module.round_to_sig_figs(-125, 2) == -130.0


def test_round_to_sig_figs_value_already_within_precision(module):
    assert module.round_to_sig_figs(7, 2) == 7.0
    assert module.round_to_sig_figs(100, 2) == 100.0
    assert module.round_to_sig_figs(99, 2) == 99.0


def test_round_to_sig_figs_more_sig_figs_than_digits_is_identity(module):
    assert module.round_to_sig_figs(42, 5) == 42.0


def test_round_to_sig_figs_rejects_non_positive_sig_figs(module):
    with pytest.raises(ValueError):
        module.round_to_sig_figs(100, 0)
    with pytest.raises(ValueError):
        module.round_to_sig_figs(100, -1)


# ---------------------------------------------------------------------------
# build_nodes_stats
# ---------------------------------------------------------------------------


def test_build_nodes_stats_shape_and_values(module):
    nodes = [1000, 2000, 3000, 4000, 5000]
    stats = module.build_nodes_stats(nodes)

    assert stats["nodes"] == nodes
    assert stats["count"] == 5
    assert stats["median"] == 3000.0
    assert stats["p25"] == 2000.0
    assert stats["p75"] == 4000.0
    assert stats["recommended_stockfish_nodes"] == 3000
    assert isinstance(stats["recommended_stockfish_nodes"], int)


def test_build_nodes_stats_recommendation_rounds_to_two_sig_figs(module):
    # median of [149000, 150000] = 149500 -> 2 sig figs -> 150000.
    stats = module.build_nodes_stats([149000, 150000])
    assert stats["median"] == 149500.0
    assert stats["recommended_stockfish_nodes"] == 150000


def test_build_nodes_stats_rejects_empty_list(module):
    with pytest.raises(ValueError):
        module.build_nodes_stats([])


def test_build_nodes_stats_single_move(module):
    stats = module.build_nodes_stats([12345])
    assert stats["count"] == 1
    assert stats["median"] == 12345.0
    assert stats["p25"] == 12345.0
    assert stats["p75"] == 12345.0
    assert stats["recommended_stockfish_nodes"] == 12000
