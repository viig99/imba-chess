from __future__ import annotations

import chess

from imba_chess.eval.search import (
    HalvingConfig,
    _auto_rounds,
    select_greedy,
    terminal_value_for_color,
)


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
