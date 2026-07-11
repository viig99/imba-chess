from __future__ import annotations


def compute_blended_value_target(
    *,
    root_wdl_unsearched: tuple[float, float, float],
    backed_value: float,
    real_outcome_stm: int,
    beta: float,
) -> list[float]:
    """blend(real_outcome, searched_value; beta) from spec Part 3.

    root_wdl_unsearched is (p_loss0, p_draw0, p_win0) from the trunk's own
    un-searched value head at the root position (a frozen snapshot recorded
    at rollout-generation time). backed_value is the searched, side-to-move
    POV scalar in [-1, 1] for the best arm. real_outcome_stm is the game's
    actual outcome from this token's side-to-move POV, in {-1, 0, 1}.

    Returns [p_loss, p_draw, p_win], summing to 1. beta=0 reproduces the
    one-hot real-outcome vector exactly; beta=1 is the pure searched
    estimate (draw mass equal to p_draw0, except in the rare case where an
    extreme backed_value combined with a high p_draw0 would otherwise drive
    p_win or p_loss negative -- there both are clamped to 0 and the vector
    is renormalized, which changes the draw share).
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if real_outcome_stm not in (-1, 0, 1):
        raise ValueError(f"real_outcome_stm must be in {{-1, 0, 1}}, got {real_outcome_stm}")

    _, p_draw0, _ = root_wdl_unsearched
    p_win_raw = (1.0 - p_draw0 + backed_value) / 2.0
    p_loss_raw = (1.0 - p_draw0 - backed_value) / 2.0
    p_win = max(0.0, p_win_raw)
    p_loss = max(0.0, p_loss_raw)
    p_draw = p_draw0
    total = p_win + p_loss + p_draw
    if total <= 0.0:
        raise ValueError(
            "root_wdl_unsearched and backed_value produced a degenerate "
            "(all-zero) searched value vector"
        )
    searched_vec = [p_loss / total, p_draw / total, p_win / total]

    real_outcome_vec = [0.0, 0.0, 0.0]
    real_outcome_vec[real_outcome_stm + 1] = 1.0

    return [
        (1.0 - beta) * real_value + beta * searched_value
        for real_value, searched_value in zip(real_outcome_vec, searched_vec)
    ]
