"""The stepwise generator core must be call-for-call identical to the sync API.

_RecordingEvaluator wraps a real evaluator and logs every evaluate() batch
(handles + cozy-board FENs). Driving the generator by hand must produce the
same chosen move, same rows, and the same sequence of evaluate() batches as
the sync wrapper — proving the wrapper/generator refactor changed nothing.
"""

import random

import chess
import pytest

from imba_chess.eval import search
from tests.test_search import _ArmValueEvaluator, _MaterialEvaluator


class _RecordingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.calls: list[list[str]] = []

    def extend(self, handle, move_uci, move_vocab_id=None):
        return self.inner.extend(handle, move_uci)

    def evaluate(self, batch):
        self.calls.append([cozy_board.fen() for _, cozy_board in batch])
        return self.inner.evaluate(batch)


def _drive_by_hand(gen, evaluator):
    try:
        request = next(gen)
        while True:
            request = gen.send(evaluator.evaluate(request.batch))
    except StopIteration as stop:
        return stop.value


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
])
@pytest.mark.parametrize("coverage,q", [(False, 0), (True, 0), (False, 2), (True, 2)])
def test_halving_generator_matches_sync_wrapper(fen, coverage, q):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_log_priors = [-1.0 - 0.01 * i for i in range(len(legal_moves))]
    config = search.HalvingConfig(budget=64, top_m=8, max_depth=3,
                                 tactical_coverage=coverage, quiescence_plies=q)

    sync_eval = _RecordingEvaluator(_MaterialEvaluator())
    sync_result = search.select_value_search_halving(
        evaluator=sync_eval, root_handle=None, board=board,
        legal_moves=legal_moves, legal_log_priors=legal_log_priors,
        config=config, rng=random.Random(7),
    )

    gen_eval = _RecordingEvaluator(_MaterialEvaluator())
    gen = search._halving_stepwise(
        root_handle=None, board=board, legal_moves=legal_moves,
        legal_log_priors=legal_log_priors, config=config, rng=random.Random(7),
        extend=gen_eval.extend,
    )
    gen_result = _drive_by_hand(gen, gen_eval)

    assert gen_result == sync_result
    assert gen_eval.calls == sync_eval.calls


def test_d2_and_rerank_wrappers_unchanged_behavior():
    # _ArmValueEvaluator's value for a position depends only on which root
    # move started the line (handle[0].uci()), keyed against the dict passed
    # to its constructor -- so, as in test_search.py's own halving tests, the
    # legal_moves list must be restricted to exactly the moves in that dict
    # (a full board.legal_moves() top_k=4 cut would surface unrelated knight/
    # pawn moves not present in arm_values and KeyError).
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    priors = [-1.0, -1.0]
    evaluator = _RecordingEvaluator(_ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6}))
    idx, rows = search.select_value_search_d2(
        evaluator=evaluator, root_handle=None, board=board,
        legal_moves=legal_moves, legal_log_priors=priors, top_k=4, lam=0.05,
    )
    assert legal_moves[idx].uci() == "e2e4"
    assert evaluator.calls  # evaluator was exercised through the wrapper


@pytest.mark.parametrize("seed", list(range(24)))
def test_budget_starvation_falls_back_to_highest_prior_arm(seed):
    """The starvation fallback must mean what it says.

    `_halving_stepwise` ends with:

        best = max(survivors, key=score)
        if best.score == -inf:
            # Budget starvation: fall back to the highest-prior candidate.
            best = arms[0]          # <-- the bug

    `arms` is built from `picks` = `order[:top_m]`. With
    gumbel_root_sampling=False `order` is _prior_order, so arms[0] genuinely
    is the highest-prior candidate. With gumbel_root_sampling=True (the
    DEFAULT in scripts/generate_search_rollouts.py) `order` is a Gumbel-Top-k
    permutation, so arms[0] is an arbitrary draw and the fallback silently
    returned a random move while claiming to return the best-prior one.

    The checkable invariant is over the CANDIDATES, not all legal moves:
    Gumbel need not sample the globally highest-prior move at all, so the
    fallback can only be the best prior among the arms that exist -- which
    `rows` reports for every arm.

    budget=0 forces the branch: the round loop breaks on
    `spent >= config.budget` before any eval, so every arm keeps score -inf.
    """
    board = chess.Board(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
    )
    legal_moves = list(board.legal_moves)
    legal_log_priors = [
        -1.0 - 0.13 * ((i * 7) % len(legal_moves)) for i in range(len(legal_moves))
    ]

    gen = search._halving_stepwise(
        extend=lambda handle, uci: None,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=search.HalvingConfig(budget=0, top_m=8, gumbel_root_sampling=True),
        rng=random.Random(seed),
    )
    try:
        next(gen)
        raise AssertionError("budget=0 must not request any evaluation")
    except StopIteration as stop:
        best_local_idx, rows = stop.value

    assert all(row["backed_value"] is None for row in rows), "budget=0 must score nothing"
    chosen_prior = legal_log_priors[best_local_idx]
    best_arm_prior = max(row["policy_log_prob"] for row in rows)
    assert chosen_prior == best_arm_prior, (
        f"starvation fallback returned {legal_moves[best_local_idx].uci()} "
        f"(prior {chosen_prior:.3f}) but the best-prior candidate among the "
        f"{len(rows)} arms has prior {best_arm_prior:.3f}"
    )


def test_budget_starvation_is_unchanged_for_deterministic_inference():
    """gumbel off (the inference setting) must keep bit-identical behavior:
    the fallback is the globally highest-prior legal move, i.e. old arms[0]."""
    board = chess.Board(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
    )
    legal_moves = list(board.legal_moves)
    legal_log_priors = [
        -1.0 - 0.13 * ((i * 7) % len(legal_moves)) for i in range(len(legal_moves))
    ]
    gen = search._halving_stepwise(
        extend=lambda handle, uci: None,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=search.HalvingConfig(budget=0, top_m=8, gumbel_root_sampling=False),
    )
    try:
        next(gen)
        raise AssertionError("budget=0 must not request any evaluation")
    except StopIteration as stop:
        best_local_idx, _rows = stop.value

    assert best_local_idx == max(
        range(len(legal_moves)), key=legal_log_priors.__getitem__
    )
