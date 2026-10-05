"""KLENT targets (Ota et al. 2026, github.com/KazukiOhta/klent main.py).

improved policy: pi'(a) ∝ exp((beta * logit(a) + q(a)) / (alpha + beta)) over legal moves
lambda-return:   G_t = r_t + gamma * ((1 - lam) * V_{t+1} + lam * G_{t+1}), gamma = -1
                 G_t = r_t at the terminal step
Rewards and values are from the perspective of the side to move at t.
"""

import math

import numpy as np
import torch


def lambda_from_tau(tau):
    return math.exp(-1.0 / tau)


def improved_policy(logits, q, legal_mask, *, alpha, beta):
    """[B, W] legal-gathered policy logits and Q -> [B, W] pi' (0 where masked)."""
    scores = (beta * logits.float() + q.float()) / (alpha + beta)
    return torch.softmax(scores.masked_fill(~legal_mask, -torch.inf), -1)


def lambda_returns(final_reward, values, lam):
    """Returns for one finished game of T moves.

    final_reward: reward of the LAST move for its mover (+1 mate, 0 draw).
    values[t]: bootstrap value of the position before move t, mover-at-t POV.
    Intermediate rewards are zero in chess, so r_t = 0 for t < T - 1.
    """
    values = np.asarray(values, dtype=np.float64)
    returns = np.empty(len(values), dtype=np.float32)
    carry = float(final_reward)
    returns[-1] = carry
    for t in range(len(values) - 2, -1, -1):
        carry = -((1.0 - lam) * values[t + 1] + lam * carry)
        returns[t] = carry
    return returns
