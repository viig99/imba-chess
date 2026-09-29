"""The stepwise generator core must be call-for-call identical to the sync API.

_RecordingEvaluator wraps a real evaluator and logs every evaluate() batch
(handles + cozy-board FENs). Driving the generator by hand must produce the
same chosen move, same rows, and the same sequence of evaluate() batches as
the sync wrapper — proving the wrapper/generator refactor changed nothing.
"""

import chess
from imba_chess.eval import search
from tests.test_search import _MaterialEvaluator


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


def test_halving_generator_matches_sync_wrapper():
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_log_priors = [-1.0 - 0.01 * i for i in range(len(legal_moves))]
    config = search.HalvingConfig(budget=64, top_m=8, max_depth=3)
    sync_eval = _RecordingEvaluator(_MaterialEvaluator())
    sync_result = search.select_value_search_halving(
        evaluator=sync_eval,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )
    gen_eval = _RecordingEvaluator(_MaterialEvaluator())
    gen = search._halving_stepwise(
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
        extend=gen_eval.extend,
    )
    gen_result = _drive_by_hand(gen, gen_eval)
    assert gen_result == sync_result
    assert gen_eval.calls == sync_eval.calls


def test_budget_starvation_falls_back_to_highest_prior_move():
    """budget=0 scores nothing; the fallback is the globally highest-prior
    legal move."""
    board = chess.Board(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
    )
    legal_moves = list(board.legal_moves)
    legal_log_priors = [
        -1.0 - 0.13 * (i * 7 % len(legal_moves)) for i in range(len(legal_moves))
    ]
    gen = search._halving_stepwise(
        extend=lambda handle, uci: None,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=search.HalvingConfig(budget=0, top_m=8),
    )
    try:
        next(gen)
        raise AssertionError("budget=0 must not request any evaluation")
    except StopIteration as stop:
        best_local_idx, _rows = stop.value
    assert best_local_idx == max(
        range(len(legal_moves)), key=legal_log_priors.__getitem__
    )
