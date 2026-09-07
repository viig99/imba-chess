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
            assert len(path) <= 1 or path[:-1] in self.seen, 'child decoded before parent'
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


@pytest.mark.parametrize('policy', ['value_search_alphabeta','value_search_pvs'])
@pytest.mark.parametrize('depth', range(1,5))
def test_context_cache_equivalence(policy, depth):
    ref, _ = run(max_depth=depth, budget=100000, policy=policy)
    cached, _ = run(max_depth=depth, budget=100000, policy=policy, score_cache='context')
    assert cached.score == ref.score
    assert cached.completed_depth == depth


def test_cache_bounds_depth_profile_and_eviction():
    from imba_chess.eval.alphabeta import _ScoreCache, _Entry
    cache = _ScoreCache(capacity=2)
    key = (1, 2, ('value_search_pvs',))
    entry = _Entry(.5, 'lower', 0, ('a2a3',))
    cache.put(key, entry)
    assert cache.get((1, 3, key[2])) is None
    assert cache.get((2, 2, key[2])) is None
    assert cache.get((1, 2, ('other',))) is None
    cache.put('second', entry)
    assert cache.get(key) == entry
    cache.put('third', entry)
    assert cache.get('second') is None


def test_direct_fail_low_high_and_exact_cache():
    from imba_chess.eval.alphabeta import _Search
    from imba_chess.eval.search import _drive
    import math
    for alpha, beta, expected in [(-2, 2, 'exact'), (-2, -1.5, 'lower'), (1.5, 2, 'upper')]:
        evaluator = Evaluator()
        ctx = _Search(evaluator.extend, AlphaBetaConfig(score_cache='context'))
        board = chess.Board(FEN)
        root = ctx.node(cozy_bridge.board_to_cozy(board), [], (), 0, None)
        # evaluate root outside counted child rows for this internal-window test
        root.evaluation = evaluator.evaluate([((), root.board)])[0]
        score, pv, selective = _drive(ctx.visit(root, 2, alpha, beta), evaluator)
        entry = ctx.cache.get((root.identity, 2, ctx.profile))
        assert entry.bound == expected
        before = len(evaluator.seen)
        assert _drive(ctx.visit(root, 2, alpha, beta), evaluator) == (score, pv, selective)
        assert ctx.stats['score_cache_cutoffs'] > 0
        assert len(evaluator.seen) == before


def test_same_board_different_paths_never_share_scores():
    from imba_chess.eval.alphabeta import _Search
    from imba_chess.eval.search import _drive
    evaluator = Evaluator()
    ctx = _Search(evaluator.extend, AlphaBetaConfig(score_cache='context'))
    board = cozy_bridge.board_to_cozy(chess.Board(FEN))
    a = ctx.node(board, [], ('path-a',), 1, None)
    b = ctx.node(board, [], ('path-b',), 1, None)
    av = _drive(ctx.visit(a, 1, -math.inf, math.inf), evaluator)
    bv = _drive(ctx.visit(b, 1, -math.inf, math.inf), evaluator)
    assert a.identity != b.identity
    assert ctx.stats['board_hash_repeats'] > 0
    assert ctx.stats['score_cache_hits'] == 0
    assert av != bv


def test_interrupted_root_not_cached():
    from imba_chess.eval.alphabeta import _Search, _BudgetExhausted
    from imba_chess.eval.search import _drive
    evaluator = Evaluator()
    ctx = _Search(evaluator.extend, AlphaBetaConfig(budget=8, score_cache='context'))
    root = ctx.node(cozy_bridge.board_to_cozy(chess.Board(FEN)), [], (), 0, None)
    root.evaluation = evaluator.evaluate([((), root.board)])[0]
    with pytest.raises(_BudgetExhausted):
        _drive(ctx.visit(root, 4, -math.inf, math.inf), evaluator)
    assert ctx.cache.get((root.identity, 4, ctx.profile)) is None


def test_lmr_eligibility_exclusions():
    from imba_chess.eval.alphabeta import _Search
    ctx = _Search(Evaluator().extend, AlphaBetaConfig(policy='value_search_pvs', lmr=True))
    evaluator = Evaluator()
    node = ctx.node(cozy_bridge.board_to_cozy(chess.Board()), [], ('root',), 1, None)
    node.evaluation = evaluator.evaluate([(node.handle, node.board)])[0]
    i = node.evaluation.legal_ucis.index('a2a3')
    assert ctx.lmr_eligible(node, 3, 3, i, False)
    assert not ctx.lmr_eligible(node, 2, 3, i, False)
    assert not ctx.lmr_eligible(node, 3, 2, i, False)
    assert not ctx.lmr_eligible(node, 3, 3, i, True)
    node.ply = 0
    assert not ctx.lmr_eligible(node, 3, 3, i, False)
    for fen in ['7k/P7/8/8/8/8/8/K7 w - - 0 1',
                '7k/8/8/8/8/8/r7/K7 w - - 0 1',
                '7k/8/8/8/8/8/pR6/K7 w - - 0 1']:
        node = ctx.node(cozy_bridge.board_to_cozy(chess.Board(fen)), [], (fen,), 1, None)
        node.evaluation = evaluator.evaluate([(node.handle, node.board)])[0]
        for i, move in enumerate(node.evaluation.legal_moves):
            if node.board.checkers() or move.promotion or cozy_bridge.is_capture_cozy(node.board, move) or cozy_bridge.gives_check(node.board, move):
                assert not ctx.lmr_eligible(node, 3, 3, i, False)


def test_lmr_selective_entries_never_cut_off():
    from imba_chess.eval.alphabeta import _Search, _Entry
    from imba_chess.eval.search import _drive
    ev = Evaluator(lambda *_: 0.0)
    ctx = _Search(ev.extend, AlphaBetaConfig(policy='value_search_pvs', lmr=True, score_cache='context'))
    node = ctx.node(cozy_bridge.board_to_cozy(chess.Board(FEN)), [], ('root',), 1, None)
    result = _drive(ctx.visit(node, 3, 0.1, math.nextafter(.1, math.inf), False), ev)
    assert result[2] and ctx.stats['lmr_attempts'] > 0
    key = (node.identity, 3, ctx.profile+(False,))
    assert ctx.cache.get(key).bound == 'selective'
    # Even a fabricated huge selective score may only guide move ordering.
    ctx.cache.put(key, _Entry(9999, 'selective', 0, ()))
    repeated = _drive(ctx.visit(node, 3, .1, math.nextafter(.1, math.inf), False), ev)
    assert repeated[0] == result[0] and repeated[2]


def test_lmr_mandatory_verification():
    report, _ = run(max_depth=5, budget=100000, policy='value_search_pvs', lmr=True)
    assert report.stats['lmr_attempts'] > 0
    assert report.stats['lmr_full_depth_verifications'] > 0
    assert report.stats['selective_results'] > 0


@pytest.mark.parametrize('cache', ['off', 'context'])
def test_lmr_budget_and_legal_completed_pv(cache):
    report, ev = run(max_depth=5, budget=2048, policy='value_search_pvs', lmr=True, score_cache=cache)
    assert len(ev.seen) <= 2048
    assert report.completed_depth >= 1
    assert report.pv


def test_lmr_requires_pvs():
    with pytest.raises(ValueError, match='LMR requires PVS'):
        AlphaBetaConfig(lmr=True)


@pytest.mark.parametrize('policy', ['value_search_alphabeta','value_search_pvs'])
def test_shallow_favorite_refuted_only_after_completed_iteration(policy):
    def value(path, board):
        favorite = path[0] == 'a2a1'
        return (-.9 if favorite else -.5) if len(path) == 1 else (-.8 if favorite else .5)
    shallow, ev = run(Evaluator(value), max_depth=1, budget=10000, policy=policy)
    assert shallow.pv[0] == 'a2a1'
    interrupted, _ = run(Evaluator(value), max_depth=2, budget=len(ev.seen), policy=policy)
    assert interrupted.completed_depth == 1 and interrupted.pv == shallow.pv
    deep, _ = run(Evaluator(value), max_depth=2, budget=10000, policy=policy)
    assert deep.completed_depth == 2 and deep.pv[0] != 'a2a1'


def test_mate_scores_prefer_shorter_wins_and_later_losses():
    from imba_chess.eval.alphabeta import _Search
    ctx = _Search(Evaluator().extend, AlphaBetaConfig())
    board = cozy_bridge.board_to_cozy(chess.Board(FEN))
    short = ctx.node(board, [], None, 2, -1).terminal
    long = ctx.node(board, [], None, 4, -1).terminal
    assert -short > -long > 1
    assert long > short and long < -1
