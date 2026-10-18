import pytest

from imba_chess.eval.search_study import policy_metrics, paired_regret_interval


def observation(policy=(0.25, 0.75), move="a2a4"):
    import math
    return dict(position_id="p", variant="eval", legal_ucis=["a2a3", "a2a4"],
                policy=list(policy), visits=[1, 3], move_uci=move,
                root_log_priors=[math.log(0.5), math.log(0.5)],
                root_wdl=(0.2, 0.5, 0.3), search_wdl=(0.1, 0.5, 0.4))


REFERENCE = dict(root=dict(expectation=0.65), moves={"a2a3": dict(expectation=0.4), "a2a4": dict(expectation=0.8)})


def test_expected_target_regret_and_value_perspective():
    result = policy_metrics(observation(), REFERENCE)
    assert result["target_regret"] == pytest.approx(0.1)
    assert result["move_regret"] == 0
    assert result["good_mass"] == 0.75
    assert result["root_wdl_mae"] == pytest.approx(0.1)
    assert result["search_wdl_mae"] == 0
    assert "played_q_mae" not in result and "improved_wdl_mae" not in result
    result = policy_metrics(dict(observation(), improved_wdl=(0.1, 0.4, 0.5)), REFERENCE)
    assert result["improved_wdl_mae"] == pytest.approx(0.05)
    # Played edge a2a4 has q = 0.2, expected score 0.6 against the 0.65 root.
    result = policy_metrics(dict(observation(), qvalues=[-0.5, 0.2]), REFERENCE)
    assert result["played_q_mae"] == pytest.approx(0.05)
    assert result["played_q_mse"] == pytest.approx(0.0025)
    with pytest.raises(ValueError, match="invalid search policy"):
        policy_metrics(observation(policy=(1, 1)), REFERENCE)


def test_paired_interval_uses_source_game_clusters():
    a, b = observation(policy=(0.5, 0.5)), observation(policy=(0, 1))
    report = paired_regret_interval([a], [b], {"p": REFERENCE}, [dict(position_id="p", game_id="g")], samples=100)
    assert report["candidate_minus_baseline"] == pytest.approx(-0.2)
    assert report["lower"] == report["upper"] == pytest.approx(-0.2)
    assert report["source_games"] == 1
    with pytest.raises(ValueError, match="identical"):
        paired_regret_interval([a], [], {"p": REFERENCE}, [])
