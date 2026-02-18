Yep, that result makes sense, and it tells us something important: **you’re no longer dominated by “zeroing 64 bytes” or dict vs tuple**. You’re dominated by **`board.piece_map()`** (it allocates/iterates a dict) + Python overhead around it.

If you want a real speed jump, the best move is: **don’t rebuild the 64-square array from the board every time**. In PGN replay (and RL rollouts) you already know the move, and a chess move only changes a handful of squares. So update the token buffer **incrementally**.

That typically gets you from ~14–16 µs/ply down to **~2–5 µs/ply** in pure Python (sometimes better), because you replace “scan all pieces” with “touch 2–4 squares”.

Below is a solid incremental encoder.

---

## Incremental board token update (fast path)

### Contract

* `buf` is a `bytearray(64)` containing your piece IDs for the **current** board.
* You apply a move `move` (a `chess.Move`) to both:

  * update `buf` to represent the **next** board state
  * and then you `board.push(move)` (or push first, but then you need pre-move info for special cases)

### Handles

* normal moves + captures
* promotions
* castling
* en passant captures

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple

import chess

_PIECE_TO_ID = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}

# Promotion piece type -> symbol (depends on mover color)
_PROMO_SYMBOL = {
    chess.QUEEN: ("Q", "q"),
    chess.ROOK: ("R", "r"),
    chess.BISHOP: ("B", "b"),
    chess.KNIGHT: ("N", "n"),
}

@dataclass(frozen=True)
class BoardTokenConfig:
    en_passant: Literal["legal", "fen", "xfen"] = "legal"
    halfmove_max: int = 100
    halfmove_bucket_size: int = 2
    fullmove_max: int = 200
    fullmove_bucket_size: int = 2

def _bucket(x: int, max_x: int, bucket_size: int) -> int:
    if x < 0: x = 0
    elif x > max_x: x = max_x
    return x // bucket_size

def _castle_id_fast(board: chess.Board) -> int:
    cr = board.castling_rights
    m = 0
    if cr & chess.BB_H1: m |= 1
    if cr & chess.BB_A1: m |= 2
    if cr & chess.BB_H8: m |= 4
    if cr & chess.BB_A8: m |= 8
    return m

def _ep_file_id_fast(board: chess.Board, mode: str) -> int:
    ep = board.ep_square
    if ep is None:
        return 0
    if mode == "legal":
        if not board.has_legal_en_passant():
            return 0
    elif mode == "xfen":
        if not board.has_pseudo_legal_en_passant():
            return 0
    elif mode == "fen":
        pass
    else:
        raise ValueError(mode)
    return chess.square_file(ep) + 1

def init_piece_buf_from_board(board: chess.Board) -> bytearray:
    """One-time initialization from piece_map()."""
    buf = bytearray(64)
    for sq, piece in board.piece_map().items():
        buf[sq] = _PIECE_TO_ID[piece.symbol()]
    return buf

def apply_move_to_buf(buf: bytearray, board: chess.Board, move: chess.Move) -> None:
    """
    Update buf IN PLACE to reflect board state after `move`,
    assuming buf currently matches `board` BEFORE pushing `move`.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    mover = board.piece_at(from_sq)
    if mover is None:
        raise ValueError("No piece on from_square; buf/board out of sync?")

    mover_sym = mover.symbol()
    mover_id = _PIECE_TO_ID[mover_sym]

    # Default: clear from, set to mover
    buf[from_sq] = 0

    # Handle captures (including en passant)
    if board.is_en_passant(move):
        # Captured pawn is behind the destination square.
        # If white captures ep, pawn is on rank 5 -> captured square is to_sq - 8
        # If black captures ep, captured square is to_sq + 8
        cap_sq = to_sq - 8 if board.turn == chess.WHITE else to_sq + 8
        buf[cap_sq] = 0
    else:
        # Normal capture: destination square overwritten anyway, so no special action needed.
        # (If you wanted, you could clear buf[to_sq] first, but overwriting is enough.)
        pass

    # Handle castling: king moves two squares and rook moves accordingly
    if board.is_castling(move):
        # King destination identifies side
        if to_sq == chess.G1:      # white king side
            rook_from, rook_to = chess.H1, chess.F1
        elif to_sq == chess.C1:    # white queen side
            rook_from, rook_to = chess.A1, chess.D1
        elif to_sq == chess.G8:    # black king side
            rook_from, rook_to = chess.H8, chess.F8
        elif to_sq == chess.C8:    # black queen side
            rook_from, rook_to = chess.A8, chess.D8
        else:
            raise ValueError("Unexpected castling destination")

        # Move king
        buf[to_sq] = mover_id

        # Move rook
        rook_piece = board.piece_at(rook_from)
        if rook_piece is None:
            raise ValueError("No rook on rook_from; buf/board out of sync?")
        buf[rook_from] = 0
        buf[rook_to] = _PIECE_TO_ID[rook_piece.symbol()]
        return

    # Handle promotion
    if move.promotion is not None:
        w_sym, b_sym = _PROMO_SYMBOL[move.promotion]
        promo_sym = w_sym if board.turn == chess.WHITE else b_sym
        buf[to_sq] = _PIECE_TO_ID[promo_sym]
        return

    # Normal move (including captures)
    buf[to_sq] = mover_id

def extract_state_fields(board: chess.Board, cfg: BoardTokenConfig) -> Tuple[int, int, int, int, int]:
    """
    Extract non-piece fields AFTER the board has the correct state.
    Returns: (turn_id, castle_id, ep_file_id, halfmove_bucket_id, fullmove_bucket_id)
    """
    turn_id = 0 if board.turn == chess.WHITE else 1
    castle_id = _castle_id_fast(board)
    ep_file_id = _ep_file_id_fast(board, cfg.en_passant)
    hm = _bucket(board.halfmove_clock, cfg.halfmove_max, cfg.halfmove_bucket_size)
    fm = _bucket(board.fullmove_number, cfg.fullmove_max, cfg.fullmove_bucket_size)
    return turn_id, castle_id, ep_file_id, hm, fm
```

### How to use during PGN replay

```python
board = chess.Board()
buf = init_piece_buf_from_board(board)
cfg = BoardTokenConfig(en_passant="legal")

for move in game.mainline_moves():
    # buf + board currently represent s_t
    # ... feed (buf, extract_state_fields(board, cfg), etc.) as input ...

    # Update buf to s_{t+1} using only local changes
    apply_move_to_buf(buf, board, move)

    # Now advance board (must match the same move)
    board.push(move)

    # buf should now match board; extract globals for s_{t+1} if needed
```

---

## Why this wins

Your current approach does “rebuild from scratch” each ply. Incremental does “touch only changed squares”:

* normal move: 2 squares
* capture: still 2 squares
* en passant: 3 squares
* castling: 4 squares
* promotion: 2 squares

That’s where the real speedup lives.

---

## One sanity check you should run once

Occasionally assert sync:

```python
def buf_matches_board(buf: bytearray, board: chess.Board) -> bool:
    for sq, piece in board.piece_map().items():
        if buf[sq] != _PIECE_TO_ID[piece.symbol()]:
            return False
    # also ensure empties (optional expensive):
    # for i in range(64): ...
    return True
```

Run it on a small sample periodically while developing, then disable.

---

If you want, paste your benchmark harness and I’ll show exactly how to time **incremental per ply** (which is the benchmark that matters in your PGN streaming loop).
