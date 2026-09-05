"""Independent coverage/quiescence controls, including a pre-change baseline."""

import hashlib
import itertools
import json

import chess
import pytest

from imba_chess.eval import cozy_bridge, search
from tests.test_search import _MaterialEvaluator
from tests.test_search_stepwise import _RecordingEvaluator


# Captured from the original algorithm before adding either switch. Includes
# chosen index, every legacy arm field, and the ordered evaluator request FENs.
_LEGACY = [
    (chess.STARTING_FEN, [
        "7f29fdb3cdd6303aa627f2e7aeb2d0c478eee82701ef2ebf0623592b9c1a58f2",
        "ad56282a59f0f92e3ec2505242a4b91907bbd50fae8699594e6d33afd3dea3da",
        "13630fa9e3907142f755eab881bdd9890f0a8fdb2d8f65d52787e9299efb0ff2",
    ]),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", [
        "96c424aa8d10f43519a399331871cc4e5dc88485785527b695f1fb34e79b7a5a",
        "0f3f982ef080a88444373ae977f38dc0fd7d29c96dafbe77439d8d494e128a97",
        "b95f7adcdee5331933fc5e55ad4a3d16017b4b42d7933b6280bb42bbf34d4de2",
    ]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [
        "6696fb8d0a0c898f713f329aa4f1fadfe854bf9d9d7102b6800b1411d92b2f34",
        "e00f054acdbefd245a9f0a3c828a154ca3476d75b3f6e40fce01994725140d71",
        "000f58577166d39680486464da76e2d299e75107aeeace1e87dbb957208e82fa",
    ]),
]


@pytest.mark.parametrize("fen,hashes", _LEGACY)
@pytest.mark.parametrize("budget_index,budget", enumerate((0, 8, 64)))
def test_disabled_search_matches_prechange_trace(fen, hashes, budget_index, budget):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    evaluator = _RecordingEvaluator(_MaterialEvaluator())
    chosen, rows = search.select_value_search_halving(
        evaluator=evaluator, root_handle=None, board=board, legal_moves=moves,
        legal_log_priors=[-1 - .01 * i for i in range(len(moves))],
        config=search.HalvingConfig(budget=budget, top_m=8, max_depth=3),
    )
    legacy_rows = [{key: value for key, value in row.items() if key != "search_stats"} for row in rows]
    encoded = json.dumps([(chosen, legacy_rows), evaluator.calls], sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == hashes[budget_index]


def _expand(fen, config, *, depth=0, opponent=False):
    board = chess.Board(fen)
    assert board.is_valid()
    cozy = cozy_bridge.board_to_cozy(board)
    node = search._TreeNode(cozy, search._root_hash_seed(board), (), depth, -0.7)
    evaluator = _MaterialEvaluator()
    position = evaluator.evaluate([((), cozy)])[0]
    node.value_stm = position.value_stm
    arm = search._Arm(0, next(iter(board.legal_moves)), -0.7, node, None)
    search._push_children(
        arm, node, position, evaluator.extend, config, itertools.count(),
        not board.turn if opponent else board.turn,
    )
    return node, arm, evaluator


def test_coverage_adds_our_low_prior_capture_and_protects_priority():
    fen = "7k/8/8/3p4/4P3/8/8/K7 w - - 0 1"
    legacy, _, _ = _expand(fen, search.HalvingConfig(expand_top=1))
    covered, arm, evaluator = _expand(fen, search.HalvingConfig(expand_top=1, tactical_coverage=True))
    assert "e4d5" not in [n.handle[-1] for n in legacy.children]
    capture = next(n for n in covered.children if n.handle[-1] == "e4d5")
    assert capture.coverage_added
    assert capture.path_log_prior == covered.path_log_prior
    capture.value_stm = evaluator.evaluate([(capture.handle, capture.cozy_board)])[0].value_stm
    assert search._backed_stm(covered) > covered.value_stm
    stats = search._arm_search_stats(arm, search.HalvingConfig())
    assert stats["coverage_added_generated"] == stats["coverage_added_evaluated"] == 1


@pytest.mark.parametrize("opponent", [False, True])
@pytest.mark.parametrize("coverage,q,depth", [(True, 0, 0), (False, 2, 1), (True, 2, 1)])
def test_all_quiet_check_evasions_are_selected(opponent, coverage, q, depth):
    fen = "7k/8/8/8/8/8/r7/4K3 w - - 0 1"
    # Put the checking rook on e2; most king evasions are quiet.
    board = chess.Board(fen)
    board.remove_piece_at(chess.A2)
    board.set_piece_at(chess.E2, chess.Piece(chess.ROOK, chess.BLACK))
    assert board.is_check()
    node, _, _ = _expand(board.fen(), search.HalvingConfig(
        max_depth=1, expand_top=1, refutation_top_r=1,
        tactical_coverage=coverage, quiescence_plies=q,
    ), depth=depth, opponent=opponent)
    assert len(node.children) == board.legal_moves.count()
    assert all(n.check_evasion for n in node.children)
    assert all(n.path_log_prior == node.path_log_prior for n in node.children)
    assert not node.stand_pat


def test_coverage_includes_quiet_root_evasions_even_with_zero_budget():
    board = chess.Board("7k/8/8/8/8/8/4r3/4K3 w - - 0 1")
    moves = list(board.legal_moves)
    results = []
    for coverage in (False, True):
        _, rows = search.select_value_search_halving(
            evaluator=_MaterialEvaluator(), root_handle=None, board=board,
            legal_moves=moves, legal_log_priors=[-float(i) for i in range(len(moves))],
            config=search.HalvingConfig(budget=0, top_m=1, tactical_coverage=coverage),
        )
        results.append({r["move_uci"] for r in rows})
    assert len(results[0]) < len(moves)
    assert results[1] == {move.uci() for move in moves}


@pytest.mark.parametrize("coverage", [False, True])
def test_quiescence_finishes_exchange_and_changes_root_choice(coverage):
    # ...d5 exd5 is evaluated as losing a pawn at D=1; ...exd5 restores it.
    board = chess.Board("7k/3p4/4p3/8/4P3/8/8/K7 b - - 0 1")
    moves = [chess.Move.from_uci(u) for u in ("d7d5", "d7d6")]
    chosen_moves = []
    for q in (0, 2):
        evaluator = _MaterialEvaluator()
        chosen, rows = search.select_value_search_halving(
            evaluator=evaluator, root_handle=None, board=board, legal_moves=moves,
            legal_log_priors=[-.1, -.2],
            config=search.HalvingConfig(
                budget=128, top_m=2, rounds=1, refutation_top_r=1, expand_top=1,
                max_depth=1, tactical_coverage=coverage, quiescence_plies=q,
            ),
        )
        chosen_moves.append(moves[chosen].uci())
        assert rows[0]["backed_value"] == pytest.approx(.1 if q else 0)
        stats = search.summarize_search_rows(rows)
        assert (stats["quiescence_evals"] > 0) == bool(q)
        assert stats["evals_spent"] == evaluator.positions_evaluated <= 128
    assert chosen_moves == ["d7d6", "d7d5"]


@pytest.mark.parametrize("child_value,expected", [(0.4, 0.3), (-0.8, 0.8)])
def test_stand_pat_can_decline_bad_capture(child_value, expected):
    node, _, _ = _expand("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1",
                         search.HalvingConfig(max_depth=1, quiescence_plies=2), depth=1)
    node.value_stm = .3
    assert node.stand_pat
    assert len(node.children) == 1
    node.children[0].value_stm = child_value
    assert search._backed_stm(node) == pytest.approx(expected)


def test_check_cannot_stand_pat_and_unsearched_evasions_are_reported():
    config = search.HalvingConfig(max_depth=1, quiescence_plies=2)
    node, arm, _ = _expand("7k/8/8/8/8/8/4r3/4K3 w - - 0 1", config, depth=1)
    node.value_stm = .8
    # Keep a quiet evasion; the rook-capture evasion would be a terminal draw.
    node.children = [child for child in node.children if child.terminal_value_stm is None]
    assert search._backed_stm(node) == .8  # bounded-search fallback before any evasion is scored
    assert search._arm_search_stats(arm, config)["unresolved_in_check_other"] >= 1
    for child in node.children:
        child.value_stm = .6
    assert search._backed_stm(node) == -.6


@pytest.mark.parametrize("fen", [
    "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",  # en passant
    "7k/P7/8/8/8/8/8/K7 w - - 0 1",  # four promotion choices
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",  # castling and quiet checks
])
def test_quiescence_selects_exact_capture_and_promotion_set(fen, monkeypatch):
    board = chess.Board(fen)
    expected = {m.uci() for m in board.legal_moves if board.is_capture(m) or m.promotion}
    selected = []
    original = search.cc.push_and_classify

    def record(cozy, move, *args):
        selected.append(cozy_bridge.cozy_move_to_uci(cozy, move))
        return original(cozy, move, *args)

    monkeypatch.setattr(search.cc, "push_and_classify", record)
    _expand(fen, search.HalvingConfig(max_depth=1, quiescence_plies=2), depth=1)
    assert set(selected) == expected
    assert len(selected) == len(expected)


@pytest.mark.parametrize("coverage,q", [(False, 0), (True, 0), (False, 2), (True, 2)])
@pytest.mark.parametrize("budget", [0, 1, 9, 128])
def test_all_combinations_share_hard_budget_and_depth_limit(coverage, q, budget):
    board = chess.Board(_LEGACY[2][0])
    moves = list(board.legal_moves)
    evaluator = _MaterialEvaluator()
    _, rows = search.select_value_search_halving(
        evaluator=evaluator, root_handle=None, board=board, legal_moves=moves,
        legal_log_priors=[-.1 * i for i in range(len(moves))],
        config=search.HalvingConfig(budget=budget, top_m=4, max_depth=1,
                                   tactical_coverage=coverage, quiescence_plies=q),
    )
    stats = search.summarize_search_rows(rows)
    assert sum(r["evals_spent"] for r in rows) == evaluator.positions_evaluated <= budget
    assert stats["evals_spent"] == evaluator.positions_evaluated
    assert stats["max_depth"] <= 1 + q
    assert stats["max_quiescence_depth"] <= q
    if not coverage:
        assert stats["coverage_added_generated"] == 0
    if not q:
        assert stats["quiescence_evals"] == 0


def test_hard_quiescence_cap_in_check_and_terminal_backup():
    config = search.HalvingConfig(max_depth=1, quiescence_plies=2)
    node, arm, _ = _expand("7k/8/8/8/8/8/4r3/4K3 w - - 0 1", config, depth=3)
    assert not node.children
    assert not node.stand_pat
    assert search._arm_search_stats(arm, config)["unresolved_in_check_depth"] == 1
    node.terminal_value_stm = -1.0
    node.stand_pat = True  # terminal always overrides any stale metadata
    assert search._backed_stm(node) == -1.0


def test_terminal_root_evasion_is_generated_without_a_neural_evaluation():
    arm = search._Arm(0, chess.Move.from_uci("e1e2"), -.1, None, 0.0,
                      root_coverage_added=True, root_check_evasion=True)
    stats = search._arm_search_stats(arm, search.HalvingConfig(tactical_coverage=True))
    assert stats["coverage_added_generated"] == stats["check_evasions_generated"] == 1
    assert stats["evals_spent"] == stats["coverage_added_evaluated"] == 0


def test_quiescence_depth_must_be_nonnegative():
    with pytest.raises(ValueError, match="quiescence_plies"):
        search.HalvingConfig(quiescence_plies=-1)
