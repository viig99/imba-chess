from __future__ import annotations

import math


def eval_from_comment(comment: str) -> tuple[float | None, int | None]:
    """Return a Lichess ``[%eval]`` as White-POV centipawns or mate count."""
    start = comment.find("[%eval ")
    if start < 0:
        return None, None
    end = comment.find("]", start)
    if end < 0:
        return None, None
    token = comment[start + 7 : end].strip()
    if token.startswith("#"):
        try:
            return None, int(token[1:])
        except ValueError:
            return None, None
    try:
        return float(token) * 100.0, None
    except ValueError:
        return None, None


# Lichess's site-wide "winning chances from a Stockfish score" function:
# scalachess core/src/main/scala/eval.scala (WinPercent), consumed by lila's
# modules/analyse (AccuracyPercent) and modules/insight. Verified 2026-09-03.
# Fixed and population-free; see docs/VALUE_TARGET_WINPERCENT_HANDOFF.md for
# why it replaced the corpus-fitted calibration and how it compares.
LICHESS_WINPERCENT_K = 0.00368208
# scalachess Cp.CEILING. Any mate maps to +-CEILING (Cp.ceilingWithSignum).
CP_CEILING = 1000.0


def winpercent_wdl(
    cp_stm: float | None, mate_stm: int | None
) -> tuple[float, float, float]:
    """Side-to-move (p_loss, p_draw, p_win) from a Lichess eval.

    The function is a symmetric logistic with no draw term, so p_draw is
    always 0 and p_win - p_loss = 2 * p_win - 1. The three-way shape is kept
    only because the value head, search and checkpoints already speak WDL.
    """
    if (cp_stm is None) == (mate_stm is None):
        raise ValueError("Exactly one of cp_stm and mate_stm must be set")
    if mate_stm is not None:
        if mate_stm == 0:
            raise ValueError("mate_stm must be non-zero")
        cp = CP_CEILING if mate_stm > 0 else -CP_CEILING
    else:
        cp = float(cp_stm)
        if not math.isfinite(cp):
            raise ValueError("cp_stm must be finite")
        cp = max(-CP_CEILING, min(CP_CEILING, cp))
    p_win = 1.0 / (1.0 + math.exp(-LICHESS_WINPERCENT_K * cp))
    return (1.0 - p_win, 0.0, p_win)
