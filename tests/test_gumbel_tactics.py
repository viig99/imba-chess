"""Halving-style Gumbel options in the native search statistics."""
import hashlib
import math
import random

import chess
import imba_chess_native as cc
import pytest

from imba_chess.eval import cozy_bridge
from imba_chess.eval import gumbel_search
from imba_chess.eval.gumbel_search import GumbelConfig, select_gumbel
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
OPTIONS = [dict(), dict(root_forcing=True), dict(forcing_floor=True), dict(minimax_weight=0.5),
           dict(minimax_weight=1.0), dict(forcing_prior=True), dict(own_width=3),
           dict(root_forcing=True, forcing_floor=True, minimax_weight=1.0, forcing_prior=True, own_width=3)]


def _search(fen, config):
    return select_gumbel(evaluator=VariedEvaluator(), board=chess.Board(fen),
                         rng=random.Random(7), config=config)


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("options", OPTIONS)
def test_tactical_options_complete_exact_budgets(fen, options):
    result = _search(fen, GumbelConfig(simulations=64, top_m=8, max_depth=8, **options))
    assert sum(result.visits) == 64
    assert sum(result.policy) == pytest.approx(1.0)
    assert all(math.isfinite(q) and abs(q) <= 1 for q in result.qvalues)


def test_default_options_keep_plain_search():
    for fen in FENS:
        plain = _search(fen, GumbelConfig(simulations=64, top_m=8, max_depth=8))
        explicit = _search(fen, GumbelConfig(simulations=64, top_m=8, max_depth=8, root_forcing=False,
                                             forcing_floor=False, minimax_weight=0.0))
        assert plain == explicit


def test_config_validation():
    with pytest.raises(ValueError, match="minimax_weight"):
        GumbelConfig(minimax_weight=1.5)
    with pytest.raises(ValueError, match="booleans"):
        GumbelConfig(forcing_floor=1)
    with pytest.raises(ValueError, match="booleans"):
        GumbelConfig(forcing_prior=1)
    with pytest.raises(ValueError, match="own_width"):
        GumbelConfig(own_width=-1)
    stats = cc.NodeStats(0.0, [0.0, -1.0], [0.6, 0.4])
    with pytest.raises(ValueError, match="forcing floor requires set_forcing"):
        stats.interior(50.0, 0.1, 1e-8, True)
    with pytest.raises(ValueError, match="candidate mask"):
        stats.set_candidates([False, False])
    with pytest.raises(ValueError, match="minimax_weight"):
        cc.NodeStats(0.0, [0.0], [1.0], 2.0)


def test_root_forcing_adds_candidates():
    fen = FENS[1]  # White can capture on e5 (Nxe5)
    board = chess.Board(fen)
    projected = cozy_bridge.project_legal_moves(cozy_bridge.board_to_cozy(board), FakeEvaluator().vocab)
    forcing = {uci for uci, flag in zip(projected[2], projected[3]) if flag}
    assert forcing
    plain = _search(fen, GumbelConfig(simulations=32, top_m=1, max_depth=4))
    tactical = _search(fen, GumbelConfig(simulations=32, top_m=1, max_depth=4, root_forcing=True))
    assert sum(1 for n in plain.visits if n) == 1
    assert forcing <= {u for u, n in zip(projected[2], tactical.visits) if n}


def test_forcing_floor_prefers_unvisited_forcing_by_prior():
    stats = cc.NodeStats(0.0, [-1.0, -3.0, -2.0, -0.5], [0.3, 0.1, 0.2, 0.4])
    stats.set_forcing([False, True, True, False])
    assert stats.interior(50.0, 0.1, 1e-8, True) == 2
    assert stats.interior(50.0, 0.1, 1e-8) == stats.interior(50.0, 0.1, 1e-8, False)


def test_minimax_backup_counts_refutation_fully():
    # Root edge 0 -> opponent node with a quiet reply (+0.2 for us) and a
    # refutation (-0.9 for us). Mean averages them; negamax takes the refutation.
    root = cc.NodeStats(0.0, [0.0, 0.0], [0.5, 0.5], 1.0)
    reply = cc.NodeStats(0.0, [0.0, 0.0], [0.5, 0.5], 1.0)
    cc.gumbel_backup([(root, 0), (reply, 0)], 0.2)
    cc.gumbel_backup([(root, 0), (reply, 1)], -0.9)
    visits, sums, q = root.snapshot()
    assert visits[0] == 2 and sums[0] / 2 == pytest.approx((0.2 - 0.9) / 2)
    assert q[0] == pytest.approx(-0.9)
    assert reply.snapshot()[2] == pytest.approx([-0.2, 0.9])
    plain = cc.NodeStats(0.0, [0.0, 0.0], [0.5, 0.5])
    plain_reply = cc.NodeStats(0.0, [0.0, 0.0], [0.5, 0.5])
    cc.gumbel_backup([(plain, 0), (plain_reply, 0)], 0.2)
    cc.gumbel_backup([(plain, 0), (plain_reply, 1)], -0.9)
    assert plain.snapshot()[2][0] == pytest.approx(-0.35)


def _node(role, config):
    board = chess.Board(FENS[1])
    evaluation = VariedEvaluator().evaluate([(("x",), cozy_bridge.board_to_cozy(board))])[0]
    node = gumbel_search._Node(cozy_bridge.board_to_cozy(board), [], None)
    node.initialize(evaluation, config, role)
    return node, evaluation


def test_forcing_prior_raises_only_opponent_forcing_replies():
    config = GumbelConfig(forcing_prior=True)
    opponent, evaluation = _node("opponent", config)
    best = max(evaluation.legal_log_priors)
    assert any(evaluation.legal_forcing)
    for raw, adjusted, forcing in zip(evaluation.legal_log_priors, opponent.priors, evaluation.legal_forcing):
        assert adjusted == (max(raw, best) if forcing else raw)
    for role in ("own", "root"):
        assert _node(role, config)[0].priors == list(evaluation.legal_log_priors)


def test_own_width_restricts_our_interior_nodes():
    def spread(node, visits=12):
        # Back each visit up at the node's own value, so only the visit deficit
        # moves selection; uncapped nodes then fan out across many moves.
        seen = set()
        for _ in range(visits):
            action = node.stats.interior(50.0, 0.1, 1e-8)
            seen.add(action)
            cc.gumbel_backup([(node.stats, action)], -node.value)
        return seen

    own, _ = _node("own", GumbelConfig(own_width=3))
    top3 = set(sorted(range(len(own.priors)), key=lambda i: (-own.priors[i], i))[:3])
    assert spread(own) <= top3
    assert len(spread(_node("opponent", GumbelConfig(own_width=3))[0])) > 3  # opponent uncapped
    assert len(spread(_node("own", GumbelConfig())[0])) > 3  # no cap by default
