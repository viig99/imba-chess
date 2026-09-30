"""Halving-style Gumbel options: Python statistics parity and tactical behavior."""
import hashlib
import math
import random

import chess
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval import gumbel_search
from imba_chess.eval.gumbel_search import GumbelConfig, select_gumbel, _PyNodeStats, _python_backup
from imba_chess.eval.search import PositionEval
from tests.test_gumbel_search import FakeEvaluator


def _unit(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class VariedEvaluator(FakeEvaluator):
    """Deterministic per-position values and priors, independent of batch order."""

    def evaluate(self, batch):
        out = []
        for ev, (handle, _) in zip(super().evaluate(batch), batch):
            key = handle or ()
            logits = [4 * _unit(key, uci) - 2 for uci in ev.legal_ucis]
            top = max(logits)
            log_z = top + math.log(math.fsum(math.exp(x - top) for x in logits))
            out.append(ev._replace(value_stm=1.8 * _unit(key, "v") - 0.9,
                                   legal_log_priors=[x - log_z for x in logits]))
        return out


FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
]


def _search(fen, config, python):
    board = chess.Board(fen)
    old = gumbel_search._uses_python_stats
    if python:
        gumbel_search._uses_python_stats = lambda c: True
    try:
        return select_gumbel(evaluator=VariedEvaluator(), board=board,
                             rng=random.Random(7), config=config)
    finally:
        gumbel_search._uses_python_stats = old


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("sims,top_m", [(16, 4), (64, 16), (200, 16)])
def test_python_statistics_match_native(fen, sims, top_m):
    config = GumbelConfig(simulations=sims, top_m=top_m, max_depth=8)
    native = _search(fen, config, python=False)
    python = _search(fen, config, python=True)
    assert python.move_uci == native.move_uci
    assert python.visits == native.visits
    assert python.qvalues == pytest.approx(native.qvalues, abs=1e-12)
    assert python.policy == pytest.approx(native.policy, abs=1e-12)
    assert python.neural_evaluations == native.neural_evaluations
    assert python.search_wdl == native.search_wdl


def test_options_off_keep_native_statistics():
    assert not gumbel_search._uses_python_stats(GumbelConfig())
    for kwargs in (dict(root_forcing=True), dict(forcing_floor=True), dict(minimax_weight=0.5)):
        assert gumbel_search._uses_python_stats(GumbelConfig(**kwargs))
    with pytest.raises(ValueError, match="minimax_weight"):
        GumbelConfig(minimax_weight=1.5)
    with pytest.raises(ValueError, match="booleans"):
        GumbelConfig(forcing_floor=1)


def test_root_forcing_adds_candidates():
    # White can capture on e5 (Nxe5); top_m=1 alone would search only one move.
    fen = FENS[1]
    board = chess.Board(fen)
    projected = cozy_bridge.project_legal_moves(cozy_bridge.board_to_cozy(board), FakeEvaluator().vocab)
    forcing = {uci for uci, flag in zip(projected[2], projected[3]) if flag}
    assert forcing
    plain = _search(fen, GumbelConfig(simulations=32, top_m=1, max_depth=4), python=False)
    tactical = _search(fen, GumbelConfig(simulations=32, top_m=1, max_depth=4, root_forcing=True), python=True)
    assert sum(1 for n in plain.visits if n) == 1
    visited = {u for u, n in zip(projected[2], tactical.visits) if n}
    assert forcing <= visited
    assert sum(tactical.visits) == 32


def test_forcing_floor_prefers_unvisited_forcing_by_prior():
    stats = _PyNodeStats(0.0, [-1.0, -3.0, -2.0, -0.5], [0.3, 0.1, 0.2, 0.4], 0.0)
    config = GumbelConfig()
    assert stats.interior(config, forced=[1, 2]) == 2
    assert stats.interior(config) == stats.interior(config, forced=())


class _Tree:
    def __init__(self, stats, value=None):
        self.stats, self.value, self.children = stats, value, {}


def test_minimax_backup_counts_refutation_fully():
    # Root edge 0 leads to an opponent node with one quiet reply (-0.2) and one
    # refutation (+0.9 for the opponent). Mean Q averages them; minimax does not.
    config_w = 1.0
    root = _Tree(_PyNodeStats(0.0, [0.0, 0.0], [0.5, 0.5], config_w))
    reply = _Tree(_PyNodeStats(0.0, [0.0, 0.0], [0.5, 0.5], config_w))
    quiet = _Tree(_PyNodeStats(0.2, [0.0], [1.0], config_w))       # root player to move: +0.2
    refute = _Tree(_PyNodeStats(-0.9, [0.0], [1.0], config_w))     # side to move (us) is losing
    root.children[0], reply.children[0], reply.children[1] = reply, quiet, refute
    _python_backup([(root, 0), (reply, 0)], quiet.stats.value)
    _python_backup([(root, 0), (reply, 1)], refute.stats.value)
    # Opponent's best reply is the refutation: its edge value is +0.9 for them.
    assert reply.stats.qvalues()[1] == pytest.approx(0.9)
    assert reply.stats.side_to_move_value() == pytest.approx(0.9)
    assert root.stats.qvalues()[0] == pytest.approx(-0.9)       # minimax: refuted
    assert root.stats.means[0] == pytest.approx((0.2 - 0.9) / 2)  # mean dilutes the refutation


@pytest.mark.parametrize("kwargs", [dict(root_forcing=True), dict(forcing_floor=True),
                                    dict(minimax_weight=0.5), dict(root_forcing=True, forcing_floor=True,
                                                                  minimax_weight=1.0)])
def test_options_complete_exact_budgets(kwargs):
    for fen in FENS:
        result = _search(fen, GumbelConfig(simulations=64, top_m=8, max_depth=8, **kwargs), python=True)
        assert sum(result.visits) == 64
        assert sum(result.policy) == pytest.approx(1.0)
        assert all(math.isfinite(q) for q in result.qvalues)
