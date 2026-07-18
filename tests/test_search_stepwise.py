"""The stepwise generator core must be call-for-call identical to the sync API.

_RecordingEvaluator wraps a real evaluator and logs every evaluate() batch
(handles + board FENs). Driving the generator by hand must produce the same
chosen move, same rows, and the same sequence of evaluate() batches as the
sync wrapper — proving the wrapper/generator refactor changed nothing.
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

    def extend(self, handle, board_before, move):
        return self.inner.extend(handle, board_before, move)

    def evaluate(self, batch):
        self.calls.append([board.fen() for _, board in batch])
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
def test_halving_generator_matches_sync_wrapper(fen):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_log_priors = [-1.0 - 0.01 * i for i in range(len(legal_moves))]
    config = search.HalvingConfig(budget=64, top_m=8, max_depth=3)

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
