"""Position-only WDL value network distilled from Stockfish evaluations.

Consumes a single board state (no game history, no clocks) and predicts
win/draw/loss from the side-to-move POV. The body reuses the big model's
BoardSquareEncoder; scalar state features are broadcast-added to the 64
square tokens so side-to-move/castling can interact with square content.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import chess
import torch
import torch.nn as nn

from .hstu_model import BoardSquareEncoder

# Stockfish 17 win_rate_model (src/uci.cpp): polynomial coefficients for the
# logistic's midpoint (a) and slope (b) as functions of normalized material.
_SF17_AS = (-37.45051876, 121.19101539, -132.78783573, 420.70576692)
_SF17_BS = (90.26261072, -137.26549898, 71.10130540, 51.35259597)


def board_material_count(board: chess.Board) -> int:
    """Stockfish's material count over both sides: P + 3N + 3B + 5R + 9Q."""
    return (
        len(board.pieces(chess.PAWN, chess.WHITE))
        + len(board.pieces(chess.PAWN, chess.BLACK))
        + 3 * (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)))
        + 3 * (len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
        + 5 * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
        + 9 * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
    )


def _win_rate(v: float, a: float, b: float) -> float:
    return 1.0 / (1.0 + math.exp((a - v) / b))


def cp_to_wdl(cp: int, material: int) -> tuple[float, float, float]:
    """(p_loss, p_draw, p_win) from a UCI centipawn eval, given board material.

    Uses Stockfish 17's win_rate_model polynomial. Lichess cp values follow
    SF's normalized-cp convention (+100 cp == 50% win probability), so cp is
    mapped back to internal units via v = cp * a(material) / 100 before the
    logistic; p_loss is the same model evaluated at -v (symmetry), and draw
    mass is the remainder.
    """
    m = min(max(material, 17), 78) / 58.0
    a = ((_SF17_AS[0] * m + _SF17_AS[1]) * m + _SF17_AS[2]) * m + _SF17_AS[3]
    b = ((_SF17_BS[0] * m + _SF17_BS[1]) * m + _SF17_BS[2]) * m + _SF17_BS[3]
    v = cp * a / 100.0
    v = min(max(v, -4000.0), 4000.0)
    p_win = _win_rate(v, a, b)
    p_loss = _win_rate(-v, a, b)
    p_draw = max(0.0, 1.0 - p_win - p_loss)
    return (p_loss, p_draw, p_win)


@dataclass(frozen=True)
class ValueNetConfig:
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 6


class ValueNet(nn.Module):
    def __init__(self, config: ValueNetConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        self.piece_square_embedding = nn.Embedding(13 * 64, dim)
        self.turn_embedding = nn.Embedding(2, dim)
        self.castle_embedding = nn.Embedding(16, dim)
        self.ep_embedding = nn.Embedding(9, dim)
        self.encoder = BoardSquareEncoder(
            dim=dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            out_dim=dim,
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 3),
        )
        self.register_buffer(
            "square_ids", torch.arange(64, dtype=torch.long), persistent=False
        )

    def _clamp_ids(self, ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        return ids.clamp(min=0, max=num_embeddings - 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.piece_square_embedding.weight.device
        piece_ids = batch["piece_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        turn_id = self._clamp_ids(
            batch["turn_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.turn_embedding.num_embeddings,
        )
        castle_id = self._clamp_ids(
            batch["castle_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.castle_embedding.num_embeddings,
        )
        ep_file_id = self._clamp_ids(
            batch["ep_file_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.ep_embedding.num_embeddings,
        )

        pair_ids = piece_ids * 64 + self.square_ids
        squares = self.piece_square_embedding(pair_ids)  # [B, 64, dim]
        features = (
            self.turn_embedding(turn_id)
            + self.castle_embedding(castle_id)
            + self.ep_embedding(ep_file_id)
        )  # [B, dim]
        squares = squares + features.unsqueeze(1)
        pooled = self.encoder(squares)  # [B, dim]
        return self.head(pooled)  # [B, 3] WDL logits (0=loss, 1=draw, 2=win)
