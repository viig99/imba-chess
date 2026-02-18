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
4. BOS target is set to `IGNORE_INDEX` (currently `-100`).

## Special IDs
- `EVENT_ID = 0` (regular event token)
- `BOS_ID = 1` (sequence token)
- `START_MOVE_ID = 0` in move vocab (used as `prev_move_id` for ply 1)
- `IGNORE_INDEX = -100` for BOS target
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
- `target_move_id = IGNORE_INDEX`

Alternative (valid): no separate BOS, and only use `START_MOVE_ID` at `t=1`.  
Recommended for now: keep BOS explicit for cleaner transformer semantics.

## Jagged Batch Tensor Shapes
For a packed batch with total tokens `N = sum(seq_lens)`:

- `num_games`: scalar
- `total_tokens`: scalar
- `seq_lens`: `[num_games]`
- `seq_offsets`: `[num_games + 1]` prefix sums
- `seq_token_id`: `[N]`
- `piece_ids`: `[N, 64]`
- `turn_id`: `[N]`
- `castle_id`: `[N]`
- `ep_file_id`: `[N]`
- `halfmove_bucket_id`: `[N]`
- `fullmove_bucket_id`: `[N]`
- `prev_move_id`: `[N]`
- `target_move_id`: `[N]` (BOS positions are `IGNORE_INDEX`)

No `attention_mask` is needed for 1D jagged/flex attention if `seq_lens` or `seq_offsets` are consumed directly.

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
1. Use a fixed static UCI vocab (all from->to plus promotions).
2. Reserve IDs for `PAD/START` (+ optional `UNK`).
3. Save/load from `artifacts/move_vocab_static_uci.json`.

## Minimal Implementation Plan (Next)
1. `move_vocab.py`: static vocab build/save/load and auto load-or-create.
2. `event_builder.py`: convert `GameRecord` -> per-game event sequence.
3. `dataloader.py`: pack by `max_tokens_per_batch` and emit jagged tensors.
4. Smoke test with streamed games and printed tensor shapes.
