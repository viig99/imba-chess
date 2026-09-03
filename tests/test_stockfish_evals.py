from __future__ import annotations

import math

import pytest

from imba_chess.data.stockfish_evals import (
    CP_CEILING,
    LICHESS_WINPERCENT_K,
    eval_from_comment,
    winpercent_wdl,
)


def _lichess_win_fraction(cp: float) -> float:
    # scalachess WinPercent.fromCentiPawns: 50 + 50 * (2 / (1 + exp(-k cp)) - 1)
    return (50 + 50 * (2 / (1 + math.exp(-LICHESS_WINPERCENT_K * cp)) - 1)) / 100


@pytest.mark.parametrize("cp", [-1500.0, -400.0, -100.0, -1.0, 0.0, 37.0, 250.0, 999.0, 2000.0])
def test_winpercent_matches_lichess_definition(cp):
    p_loss, p_draw, p_win = winpercent_wdl(cp, None)
    clamped = max(-CP_CEILING, min(CP_CEILING, cp))
    assert p_win == pytest.approx(_lichess_win_fraction(clamped))
    assert p_draw == 0.0
    assert p_loss == pytest.approx(1.0 - p_win)


def test_winpercent_is_symmetric_and_bounded():
    l_neg, _, w_neg = winpercent_wdl(-150.0, None)
    l_pos, _, w_pos = winpercent_wdl(150.0, None)
    assert w_neg == pytest.approx(l_pos)
    assert l_neg == pytest.approx(w_pos)
    assert winpercent_wdl(0.0, None)[2] == pytest.approx(0.5)
    # Clamped at +-CP_CEILING: beyond it the target stops moving.
    assert winpercent_wdl(CP_CEILING, None) == winpercent_wdl(CP_CEILING * 5, None)


@pytest.mark.parametrize("mate", [1, 7, 40])
def test_mate_maps_to_ceiling_regardless_of_distance(mate):
    assert winpercent_wdl(None, mate) == winpercent_wdl(CP_CEILING, None)
    assert winpercent_wdl(None, -mate) == winpercent_wdl(-CP_CEILING, None)


def test_winpercent_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Exactly one"):
        winpercent_wdl(None, None)
    with pytest.raises(ValueError, match="Exactly one"):
        winpercent_wdl(10.0, 3)
    with pytest.raises(ValueError, match="non-zero"):
        winpercent_wdl(None, 0)
    with pytest.raises(ValueError, match="finite"):
        winpercent_wdl(float("nan"), None)


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("[%eval 0.14]", (14.0, None)),
        ("[%eval #3]", (None, 3)),
        ("[%eval #-2]", (None, -2)),
        ("[%clk 0:01:00]", (None, None)),
        ("[%eval nope]", (None, None)),
        ("ordinary comment", (None, None)),
    ],
)
def test_eval_from_comment(comment, expected):
    actual = eval_from_comment(comment)
    if expected[0] is not None:
        assert actual[0] == pytest.approx(expected[0])
        assert actual[1] is None
    else:
        assert actual == expected
