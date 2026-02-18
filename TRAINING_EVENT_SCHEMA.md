# Training Event Schema (Next-Move Pretraining)

## Goal
Train a sequential model to predict `move_t` from game prefix up to ply `t`.

## Core Principle
- Keep board state structured (categorical IDs), not text-tokenized FEN/SAN.
- Use UCI move IDs as prediction targets.
- Keep winner/loser as metadata only (not policy input features).

## Sequence Definition
For a game with `N` plies:

1. Add `BOS` step.
2. For each ply `t in [1..N]`, build event `E_t` from state **before** `move_t`.
3. Target at step `t` is `move_t`.

Loss is masked off for `BOS` and padding.

## Special IDs
- `PAD_ID = 0`
- `BOS_ID = 1` (sequence token, not a move)
- `START_MOVE_ID = 0` in move vocab (used as `prev_move_id` for ply 1)
- Optional: `UNK_MOVE_ID` if move vocab is not fully closed

## Per-Step Event Fields
Each event `E_t` contains:

- `piece_ids`: shape `[64]`, values `0..12`
  - `0` empty, `1..6` white pieces, `7..12` black pieces
- `turn_id`: scalar, `0` white / `1` black
- `castle_id`: scalar `0..15`
- `ep_file_id`: scalar `0..8`
- `halfmove_bucket_id`: scalar `>=0`
- `fullmove_bucket_id`: scalar `>=0`
- `prev_move_id`: scalar move-vocab ID
  - `START_MOVE_ID` for ply 1
  - otherwise ID of `move_{t-1}`
- Optional later:
  - `time_taken_bucket_id`
  - `active_elo_bucket_id`
  - `opponent_elo_bucket_id`
  - `time_control_bucket_id`

Target:
- `target_move_id`: ID of `move_t` (UCI vocab)

## BOS Step
Represent BOS as a separate sequence position with:
- `seq_token_id = BOS_ID`
- all event fields set to neutral defaults (`0`)
- `target_move_id = PAD_ID`
- `loss_mask = 0`

Alternative (valid): no separate BOS, and only use `START_MOVE_ID` at `t=1`.  
Recommended for now: keep BOS explicit for cleaner transformer semantics.

## Batch Tensor Shapes
For batch size `B`, max length `T` (includes BOS):

- `seq_token_id`: `[B, T]` (BOS/PAD marker)
- `piece_ids`: `[B, T, 64]`
- `turn_id`: `[B, T]`
- `castle_id`: `[B, T]`
- `ep_file_id`: `[B, T]`
- `halfmove_bucket_id`: `[B, T]`
- `fullmove_bucket_id`: `[B, T]`
- `prev_move_id`: `[B, T]`
- `target_move_id`: `[B, T]`
- `attention_mask`: `[B, T]` (1 real, 0 pad)
- `loss_mask`: `[B, T]` (0 on BOS/pad, 1 on trainable plies)

## Mapping from Current Dataset
From each `play` in current dataset:
- `play.state.piece_ids` -> `piece_ids`
- `play.state.turn_id` -> `turn_id`
- `play.state.castle_id` -> `castle_id`
- `play.state.ep_file_id` -> `ep_file_id`
- `play.state.halfmove_bucket_id` -> `halfmove_bucket_id`
- `play.state.fullmove_bucket_id` -> `fullmove_bucket_id`
- `play.move_uci` -> `target_move_id`

## Move Vocab Strategy
Recommended first pass:
1. Build UCI vocab from training corpus once.
2. Reserve IDs for `PAD/START` (+ optional `UNK`).
3. Serialize vocab map for reproducibility.

Later optimization:
- Use fixed "all possible chess moves" vocab to avoid OOV handling.

## Minimal Implementation Plan (Next)
1. `move_vocab.py`: fit/save/load/encode/decode UCI IDs.
2. `event_builder.py`: convert `GameRecord` -> step tensors + targets.
3. `collator.py`: pad/stack + masks.
4. Smoke test with 2-3 games and shape assertions.

