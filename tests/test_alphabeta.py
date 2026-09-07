"""Deterministic CPU oracles for the evaluation-only search architecture."""
import hashlib
import math

import chess
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval.alphabeta import AlphaBetaConfig, search_stepwise, select_value_search
from imba_chess.eval.search import PositionEval


class Evaluator:
    def __init__(self, value=None, ordering='tied'):
        self.seen = []
        self.value = value or (lambda path, board: (int.from_bytes(hashlib.sha256('/'.join(path).encode()).digest()[:2]) % 201 - 100) / 100)
        self.ordering = ordering

    def extend(self, handle, uci, move_vocab_id=None):
        return (handle or ()) + (uci,)

    def evaluate(self, batch):
        rows = []
        for path, board in batch:
            assert path not in self.seen, 'duplicate neural evaluation'
            assert len(path) == 1 or path[:-1] in self.seen, 'child decoded before parent'
            self.seen.append(path)
            moves = list(board.generate_moves())
            ucis = [cozy_bridge.cozy_move_to_uci(board, m) for m in moves]
            priors = [0.0 if self.ordering == 'tied' else (-i if self.ordering == 'good' else i) for i in range(len(moves))]
            rows.append(PositionEval(self.value(path, board), moves, ucis, priors,
                                     [m.promotion is not None or cozy_bridge.is_capture_cozy(board, m) or cozy_bridge.gives_check(board, m) for m in moves],
                                     list(range(len(moves)))))
        return rows


FEN = '8/7k/7p/8/8/P7/K7/8 w - - 0 1'


def run(evaluator=None, board=None, **config):
    board = board or chess.Board(FEN)
    evaluator = evaluator or Evaluator()
    moves = list(board.legal_moves)
    report = select_value_search(evaluator=evaluator, root_handle=None, board=board,
                                 legal_moves=moves, legal_log_priors=[0.0]*len(moves),
                                 config=AlphaBetaConfig(**config))
    twin = board.copy()
    for uci in report.pv:
        move = chess.Move.from_uci(uci)
        assert move in twin.legal_moves
        twin.push(move)
    return report, evaluator


def minimax(board, depth, value, path=()):
    outcome = board.outcome(claim_draw=True)
    if outcome:
        return -(10000-len(path)) if outcome.winner is not None else 0.0
    if not depth:
        return value(path, cozy_bridge.board_to_cozy(board))
    return max(-minimax(child(board, move), depth-1, value, path+(move.uci(),)) for move in board.legal_moves)


def child(board, move):
    board = board.copy()
    board.push(move)
    return board


@pytest.mark.parametrize('depth', range(1,5))
@pytest.mark.parametrize('ordering', ['good','poor','tied'])
def test_exhaustive_minimax(depth, ordering):
    evaluator = Evaluator(ordering=ordering)
    expected = minimax(chess.Board(FEN), depth, evaluator.value)
    report, _ = run(evaluator, budget=100000, max_depth=depth)
    assert report.score == expected
    assert report.completed_depth == depth
    assert report.stats['new_neural_evaluations'] == len(evaluator.seen)


def test_every_budget_boundary_and_completed_publication():
    full, ev = run(max_depth=3, budget=100000)
    completed = {d: run(max_depth=d, budget=100000)[0] for d in range(1,4)}
    for budget in range(len(ev.seen)+1):
        report, evaluator = run(max_depth=3, budget=budget)
        assert len(evaluator.seen) <= budget
        if report.completed_depth:
            reference = completed[report.completed_depth]
            assert (report.score, report.pv) == (reference.score, reference.pv)
        else:
            assert report.score is None and report.pv == []
            assert report.stop_reason == 'no_completed_iteration_fallback'
    fixed, _ = run(max_depth=3, budget=1, iterative_deepening=False)
    assert fixed.score is None and fixed.completed_depth == 0


def test_stepwise_request_sequence():
    report, ev = run(max_depth=3, budget=180)
    twin = Evaluator()
    board = chess.Board(FEN)
    moves = list(board.legal_moves)
    gen = search_stepwise(extend=twin.extend, root_handle=None, board=board,
                         legal_moves=moves, legal_log_priors=[0.0]*len(moves),
                         config=AlphaBetaConfig(max_depth=3, budget=180))
    try:
        request = next(gen)
        while True:
            assert len(request.batch) == 1
            request = gen.send(twin.evaluate(request.batch))
    except StopIteration as stop:
        other = stop.value
    assert twin.seen == ev.seen
    assert (other.score, other.pv) == (report.score, report.pv)


@pytest.mark.parametrize('fen', ['R5k1/5ppp/8/8/8/8/8/7K b - - 0 1',
    '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', '7k/8/6K1/8/8/8/8/8 w - - 0 1'])
def test_terminal(fen):
    report, ev = run(board=chess.Board(fen))
    assert report.chosen_index is None and not ev.seen
    assert report.stop_reason == 'terminal_position'


def test_immediate_mate_without_budget():
    report, ev = run(board=chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1'), budget=0)
    assert report.score == 9999 and not ev.seen
    assert report.completed_depth == 1


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, 1.1])
def test_invalid_value(value):
    with pytest.raises(ValueError, match='Invalid neural value'):
        run(Evaluator(lambda *_: value))


def test_missing_root_coverage():
    board = chess.Board()
    with pytest.raises(ValueError, match='coverage'):
        select_value_search(evaluator=Evaluator(), root_handle=None, board=board,
                            legal_moves=list(board.legal_moves)[:-1], legal_log_priors=[0]*19,
                            config=AlphaBetaConfig())


def test_missing_child_coverage():
    class Broken(Evaluator):
        def evaluate(self, batch):
            ev = super().evaluate(batch)[0]
            return [ev._replace(legal_ucis=ev.legal_ucis[:-1])]
    with pytest.raises(ValueError, match='coverage'):
        run(Broken())


@pytest.mark.parametrize('fen', [
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1',
    '7k/P7/8/8/8/8/8/K7 w - - 0 1',
    '7k/8/8/8/8/8/r7/K7 w - - 0 1',
])
def test_special_moves_and_checked_static_horizon(fen):
    board = chess.Board(fen)
    evaluator = Evaluator()
    report, _ = run(evaluator, board=board, max_depth=2, budget=100000)
    assert report.score == minimax(board, 2, evaluator.value)


def test_draw_claim_and_repetition():
    board = chess.Board()
    for move in ['g1f3','g8f6','f3g1','f6g8']*2:
        board.push_uci(move)
    report, ev = run(board=board)
    assert report.score == 0 and not ev.seen
    report, ev = run(board=chess.Board('8/7k/7p/8/8/P7/K7/8 w - - 99 60'))
    assert report.score == 0 and not ev.seen


@pytest.mark.parametrize('depth', range(1,5))
@pytest.mark.parametrize('ordering', ['good','poor','tied'])
def test_pvs_agrees_with_alphabeta(depth, ordering):
    ab, _ = run(Evaluator(ordering=ordering), max_depth=depth, budget=100000)
    pvs, _ = run(Evaluator(ordering=ordering), max_depth=depth, budget=100000, policy='value_search_pvs')
    assert (ab.score, ab.completed_depth) == (pvs.score, pvs.completed_depth)
    assert pvs.stats['pvs_scout_calls'] > 0


@pytest.mark.parametrize('value', [0.0, -0.5, math.nextafter(0.0, math.inf), math.nextafter(-0.5, math.inf)])
def test_float_scouts(value):
    # Two adjacent representable scores must not be rounded to an integer window.
    fn = lambda path, board: value if path[-1] == 'a2a1' else math.nextafter(value, -math.inf)
    ab, _ = run(Evaluator(fn), max_depth=3, budget=100000)
    pvs, _ = run(Evaluator(fn), max_depth=3, budget=100000, policy='value_search_pvs')
    assert ab.score == pvs.score


def test_pvs_research_reuses_evaluations():
    report, evaluator = run(max_depth=4, budget=100000, policy='value_search_pvs')
    assert report.stats['pvs_full_window_researches'] > 0
    assert report.stats['raw_eval_cache_hits'] > 0
    assert report.stats['new_neural_evaluations'] == len(evaluator.seen)


def test_pvs_budget_boundaries():
    full, ev = run(max_depth=3, budget=100000, policy='value_search_pvs')
    completed = {d: run(max_depth=d, budget=100000, policy='value_search_pvs')[0] for d in range(1,4)}
    for budget in range(len(ev.seen)+1):
        report, evaluator = run(max_depth=3, budget=budget, policy='value_search_pvs')
        assert len(evaluator.seen) <= budget
        if report.completed_depth:
            assert report.score == completed[report.completed_depth].score
        else:
            assert report.score is None and not report.pv
