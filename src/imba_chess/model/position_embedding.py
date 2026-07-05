"""Within-sequence positional embedding preprocessor for jagged batches."""

import math

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """Scales content embeddings and adds learnable positional embeddings.

    Combines: ``content * sqrt(D) + pos_embed(positions) -> dropout``.

    Positions are derived at runtime from jagged offsets: sequence *i*
    spans ``[offsets[i], offsets[i+1])``, so positions within each
    sequence restart from 0.

    Args:
        max_seq_len: Maximum sequence length (positions clipped to this).
        embedding_dim: Output embedding dimension (D).
        dropout_rate: Dropout applied after combining.
    """

    def __init__(
        self, max_seq_len: int, embedding_dim: int, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.trunc_normal_(self.embedding.weight, std=math.sqrt(1.0 / embedding_dim))

    def forward(self, content: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: [S, D] float — sum of token feature embeddings.
            offsets: [num_sequences + 1] int64 jagged offsets.

        Returns:
            [S, D] float — scaled content + positional embeddings, with dropout.
        """
        S = content.shape[0]
        positions = torch.arange(S, device=offsets.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        sequence_starts = torch.repeat_interleave(offsets[:-1], lengths)
        positions = torch.clamp(positions - sequence_starts, max=self.max_seq_len - 1)

        x = content * (self._embedding_dim**0.5) + self.embedding(positions)
        return self.dropout(x)

    def at_positions(
        self, content: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Same combine as forward() but with caller-supplied absolute positions.

        Used by the decode path, where positions are prefix_len + suffix depth
        rather than derived from jagged offsets.
        """
        positions = torch.clamp(positions, max=self.max_seq_len - 1)
        x = content * (self._embedding_dim**0.5) + self.embedding(positions)
        return self.dropout(x)
