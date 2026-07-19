from __future__ import annotations

import math
import random

import chess

from imba_chess.eval import cozy_bridge
from imba_chess.eval.search import (
    HalvingConfig,
    PositionEval,
    _auto_rounds,
    _gumbel_top_k_order,
    select_greedy,
    select_value_rerank,
    select_value_search_d2,
    select_value_search_halving,
    terminal_value_for_color,
)


def _cozy_legal_moves_sorted(cozy_board):
    """Cozy legal moves + aligned ucis, UCI-sorted -- the canonical order
    real evaluators (position_evaluator._project_legal_logits_cozy) return.
    """
    moves = list(cozy_board.generate_moves())
    ucis = [cozy_bridge.cozy_move_to_uci(cozy_board, move) for move in moves]
    order = sorted(range(len(moves)), key=lambda i: ucis[i])
    return [moves[i] for i in order], [ucis[i] for i in order]


def test_select_greedy_returns_argmax_index_first_on_ties():
    assert select_greedy([-2.0, -0.5, -1.0]) == 1
    assert select_greedy([-1.0, -1.0]) == 0


def test_terminal_value_for_color():
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/7K b - - 0 1")  # black is mated
    assert terminal_value_for_color(mated, color=chess.WHITE) == 1.0
    assert terminal_value_for_color(mated, color=chess.BLACK) == -1.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # black stalemated
    assert terminal_value_for_color(stalemate, color=chess.WHITE) == 0.0
    ongoing = chess.Board()
    assert terminal_value_for_color(ongoing, color=chess.WHITE) is None


def test_auto_rounds():
    assert _auto_rounds(16) == 4
    assert _auto_rounds(2) == 1
    assert _auto_rounds(3) == 2
    assert _auto_rounds(1) == 1
    assert _auto_rounds(17) == 5


def test_halving_config_defaults_match_spec():
    config = HalvingConfig()
    assert config.budget == 256
    assert config.top_m == 16
    assert config.rounds == 0
    assert config.refutation_top_r == 2
    assert config.expand_top == 3
    assert config.max_depth == 4
    assert config.lam == 0.05
    assert config.gumbel_root_sampling is False


def test_gumbel_top_k_order_is_a_valid_permutation():
    priors = [-0.1, -2.0, -5.0, -0.5, -3.0]
    order = _gumbel_top_k_order(priors, rng=random.Random(0))
    assert sorted(order) == list(range(len(priors)))


def test_gumbel_top_k_order_is_reproducible_given_same_rng_state():
    priors = [-0.1, -2.0, -5.0, -0.5, -3.0]
    first = _gumbel_top_k_order(priors, rng=random.Random(123))
    second = _gumbel_top_k_order(priors, rng=random.Random(123))
    assert first == second


def test_gumbel_top_k_order_can_surface_low_prior_index_unlike_deterministic_cut():
    # Mirrors the paper's Example 1 (Danihelka et al., ICLR 2022, Sec 3.2):
    # a low-prior action can still be worth searching, and a deterministic
    # top-k-by-prior cut can never place it in the top-k regardless of seed.
    # Gumbel-Top-k must, for at least some seeds, place a low-prior index
    # ahead of a higher-prior one (softmax([-0.1,-0.3,-3.0])[2] ~= 0.03, so
    # 500 trials gives an expected ~15 occurrences -- not a coin-flip test).
    priors = [-0.1, -0.3, -3.0]  # index 2 is low prior but not vanishingly so
    surfaced = any(
        _gumbel_top_k_order(priors, rng=random.Random(seed))[0] == 2 for seed in range(500)
    )
    assert surfaced


def test_gumbel_root_sampling_disabled_matches_deterministic_order():
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    legal_log_priors = [-0.5, -0.6]
    evaluator = _ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6})
    config = HalvingConfig(budget=8, top_m=2, rounds=2, lam=0.05, gumbel_root_sampling=False)

    chosen, _ = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
        rng=random.Random(0),
    )
    assert legal_moves[chosen].uci() == "e2e4"


_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_stm(board: chess.Board) -> float:
    """Material balance from the side-to-move's POV, scaled to roughly [-1, 1]."""
    total = 0
    for piece in board.piece_map().values():
        sign = 1 if piece.color == board.turn else -1
        total += sign * _PIECE_VALUES[piece.piece_type]
    return max(-1.0, min(1.0, total / 10.0))


class _ArmValueEvaluator:
    """Value depends only on which root move started the line (handle[0]).

    value_stm flips sign with ply parity so negamax backup returns exactly
    the arm's root-POV value at any depth. Priors are uniform.
    """

    def __init__(self, arm_values_root_pov: dict[str, float]) -> None:
        self.arm_values = arm_values_root_pov
        self.eval_calls = 0
        self.positions_evaluated = 0

    def extend(self, handle, move_uci):
        return (handle or ()) + (move_uci,)

    def evaluate(self, batch):
        self.eval_calls += 1
        self.positions_evaluated += len(batch)
        results = []
        for handle, cozy_board in batch:
            value_root_pov = self.arm_values[handle[0]]
            stm_is_root_side = len(handle) % 2 == 0
            value_stm = value_root_pov if stm_is_root_side else -value_root_pov
            moves, ucis = _cozy_legal_moves_sorted(cozy_board)
            log_prior = math.log(1.0 / len(moves)) if moves else 0.0
            results.append(PositionEval(value_stm, moves, ucis, [log_prior] * len(moves)))
        return results


class _MaterialEvaluator:
    """Material-count value; priors rank captures/checks/promotions LAST."""

    def __init__(self) -> None:
        self.eval_calls = 0
        self.positions_evaluated = 0

    def extend(self, handle, move_uci):
        return (handle or ()) + (move_uci,)

    def evaluate(self, batch):
        self.eval_calls += 1
        self.positions_evaluated += len(batch)
        results = []
        for handle, cozy_board in batch:
            moves, ucis = _cozy_legal_moves_sorted(cozy_board)
            board = chess.Board(cozy_board.fen())
            priors = [
                -5.0
                if (
                    m.promotion is not None
                    or board.is_capture(chess.Move.from_uci(u))
                    or board.gives_check(chess.Move.from_uci(u))
                )
                else -0.1
                for m, u in zip(moves, ucis)
            ]
            results.append(PositionEval(_material_stm(board), moves, ucis, priors))
        return results


def test_halving_eliminates_low_value_arm_and_spends_exact_budget():
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    legal_log_priors = [-0.5, -0.6]
    evaluator = _ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6})
    config = HalvingConfig(budget=8, top_m=2, rounds=2, lam=0.05)

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )

    assert legal_moves[chosen].uci() == "e2e4"
    assert evaluator.positions_evaluated == 8  # exact budget
    by_move = {row["move_uci"]: row for row in rows}
    assert by_move["d2d4"]["eliminated_round"] == 0
    assert by_move["e2e4"]["eliminated_round"] is None
    # Round-2 budget flowed to the survivor.
    assert by_move["e2e4"]["evals_spent"] > by_move["d2d4"]["evals_spent"]
    assert by_move["e2e4"]["backed_value"] > by_move["d2d4"]["backed_value"]


def test_rounds_one_is_pure_beam_no_elimination():
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    evaluator = _ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6})
    config = HalvingConfig(budget=8, top_m=2, rounds=1, lam=0.05)

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=[-0.5, -0.6],
        config=config,
    )

    assert legal_moves[chosen].uci() == "e2e4"
    assert all(row["eliminated_round"] is None for row in rows)
    assert all(row["evals_spent"] > 0 for row in rows)
    assert evaluator.positions_evaluated <= 8


def test_mate_in_one_short_circuits_with_zero_evals():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    legal_moves = [chess.Move.from_uci("a1b1"), chess.Move.from_uci("a1a8")]
    legal_log_priors = [-0.1, -3.0]  # mate is LOW prior
    evaluator = _ArmValueEvaluator({"a1b1": 0.0, "a1a8": 0.0})
    config = HalvingConfig(budget=16, top_m=2, rounds=2)

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )

    assert legal_moves[chosen].uci() == "a1a8"
    assert evaluator.eval_calls == 0
    assert len(rows) == 1 and rows[0]["search_score"] == 1.0


def test_mate_in_one_short_circuits_with_zero_evals_via_rerank():
    # Pins _expand_root_candidates_stepwise's color_is_stm=False site (the
    # root-candidate terminal check, shared by select_value_rerank and
    # select_value_search_d2 via `yield from`): a low-prior mate must still
    # be recognized and short-circuit before any evaluate() batch is sent.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    legal_moves = [chess.Move.from_uci("a1b1"), chess.Move.from_uci("a1a8")]
    legal_log_priors = [-0.1, -3.0]  # mate is LOW prior
    evaluator = _ArmValueEvaluator({"a1b1": 0.0, "a1a8": 0.0})

    chosen, rows = select_value_rerank(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        top_k=2,
        lam=0.05,
    )

    assert legal_moves[chosen].uci() == "a1a8"
    assert evaluator.eval_calls == 0
    assert len(rows) == 1 and rows[0]["rerank_score"] == 1.0


def test_mate_in_one_short_circuits_with_zero_evals_via_d2():
    # Same fixture/site as the rerank version above, through the other
    # wrapper of _expand_root_candidates_stepwise.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    legal_moves = [chess.Move.from_uci("a1b1"), chess.Move.from_uci("a1a8")]
    legal_log_priors = [-0.1, -3.0]  # mate is LOW prior
    evaluator = _ArmValueEvaluator({"a1b1": 0.0, "a1a8": 0.0})

    chosen, rows = select_value_search_d2(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        top_k=2,
        lam=0.05,
    )

    assert legal_moves[chosen].uci() == "a1a8"
    assert evaluator.eval_calls == 0
    assert len(rows) == 1 and rows[0]["search_score"] == 1.0


def test_refutation_floor_catches_low_prior_forcing_refutation():
    # White Qd2, black Nb4. Qd3?? hangs the queen to Nxd3 — a capture the
    # priors rank last. Qh6 is safe. Only the forcing-reply floor finds Nxd3.
    board = chess.Board("k7/8/8/8/1n6/8/3Q4/K7 w - - 0 1")
    legal_moves = [chess.Move.from_uci("d2h6"), chess.Move.from_uci("d2d3")]
    legal_log_priors = [-0.7, -0.7]
    evaluator = _MaterialEvaluator()
    config = HalvingConfig(
        budget=8, top_m=2, rounds=1, refutation_top_r=1, expand_top=2, max_depth=2
    )

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )

    assert legal_moves[chosen].uci() == "d2h6"
    by_move = {row["move_uci"]: row for row in rows}
    assert by_move["d2d3"]["backed_value"] < by_move["d2h6"]["backed_value"]
    assert evaluator.positions_evaluated <= 8


def test_fools_mate_loss_attribution_via_d2():
    # Pins _d2_stepwise's color_is_stm=True site (the d2-reply terminal
    # check): after 1.f3 e5, 2.g4?? allows 2...Qh4# -- the reply's mate must
    # be attributed to g2g4 as a root-POV loss (worst_reply_value == -1.0),
    # not missed or mis-signed. Qh4+ is a forcing (check) reply, so the
    # low-prior-ranks-checks-last _MaterialEvaluator still finds it via the
    # forcing-reply floor. g2g3 is the safe alternative (blocks the h4-e1
    # diagonal, no check at all), so the search must not walk into the trap.
    board = chess.Board()
    board.push_uci("f2f3")
    board.push_uci("e7e5")
    legal_moves = [chess.Move.from_uci("g2g4"), chess.Move.from_uci("g2g3")]
    legal_log_priors = [-0.7, -0.7]
    evaluator = _MaterialEvaluator()

    chosen, rows = select_value_search_d2(
        evaluator=evaluator,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        top_k=3,
        lam=0.05,
    )

    by_move = {row["move_uci"]: row for row in rows}
    assert by_move["g2g4"]["worst_reply_value"] == -1.0
    assert by_move["g2g4"]["best_reply_uci"] == "d8h4"
    assert legal_moves[chosen].uci() == "g2g3"
