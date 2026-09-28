import math

from scripts.audit_blunders import analyse


def _scores(**exp):
    return {m: dict(expectation=e) for m, e in exp.items()}


def _result(chosen, priors, visits, q, root_value=0.0):
    return dict(move_uci=chosen, root_log_priors=[math.log(p) for p in priors],
                visits=visits, qvalues=q, root_value=root_value)


def test_policy_miss_when_no_good_move_is_a_candidate():
    moves = ["a", "b", "c"]
    r = analyse(_result("a", [.6, .3, .1], [5, 5, 0], [0.1, 0.0, 0.0]), moves, _scores(a=.2, b=.2, c=.9), top_m=2)
    assert r["cause"] == "policy_miss" and r["best_good_prior_rank"] == 3


def test_value_misorder_when_search_q_prefers_worse_move():
    moves = ["a", "b"]
    r = analyse(_result("a", [.5, .5], [8, 8], [0.3, 0.1]), moves, _scores(a=.3, b=.8), top_m=2)
    assert r["cause"] == "value_misorder" and r["best_good_move"] == "b"


def test_selection_override_and_good_choice():
    moves = ["a", "b"]
    r = analyse(_result("a", [.9, .1], [8, 8], [0.1, 0.3]), moves, _scores(a=.3, b=.8), top_m=2)
    assert r["cause"] == "selection_override"
    r = analyse(_result("b", [.9, .1], [8, 8], [0.1, 0.3]), moves, _scores(a=.79, b=.8), top_m=2)
    assert r["cause"] == "chose_good_move" and r["regret"] == 0
