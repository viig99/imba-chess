# Value Head Plan for `imba-chess` (Repo-Specific)

This replaces the earlier draft with a plan aligned to how this repo actually works.

## Objective

Improve engine strength from the current baseline (`<5%` score vs Stockfish ~1400, per your note) by:

1. Learning a value head from Lichess game outcomes.
2. Using that value at inference time (not just as an auxiliary training loss).

## Critical Repo Facts (Must Respect)

1. Token states are encoded **before** the move target, not after.
   - `TRAINING_EVENT_SCHEMA.md:15`
   - `src/imba_chess/data/lichess_dataset.py:360`
2. `turn_id` is already side-to-move (`0` white, `1` black).
   - `src/imba_chess/data/event_builder.py:43`
3. Training sequences are `BOS + one token per ply`; there is no extra explicit terminal token.
   - `src/imba_chess/data/event_builder.py:24`
4. Current model/eval is policy-only in usage.
   - Policy head only: `src/imba_chess/model/hstu_model.py:133`
   - Inference move selection uses policy logits only: `scripts/eval_vs_stockfish.py:406`

Implication: adding value loss alone may help representations a bit, but large playing-strength gains usually require using value during move selection.

## What Was Good in the Previous Draft

1. WDL classification head (`loss/draw/win`) is the right starting target.
2. Side-to-move perspective is correct.
3. Late-position weighting is reasonable.
4. Start without TD(0) regularization.

## What Needed Correction

1. No need for a new per-token `side_to_move_id`; use existing `turn_id`.
2. "Exclude terminal token" logic should be adjusted because current dataset has no explicit terminal token.
3. Value plan must include inference integration; otherwise Stockfish results may barely move.
4. Value-only-from-outcome is noisy; we should debias and evaluate calibration, not only CE.

## Implementation Plan

### Phase 0: Evaluation Protocol First (Before Any Model Changes)

1. Keep a fixed deterministic benchmark profile for model strength:
   - `model_move_policy = greedy`
   - `opening_random_plies = 0`
   - fixed seed
2. Keep your current sampled profile as a secondary "style/diversity" profile.
3. Always compare value-head experiments to the same baseline protocol.

Reason: current sampled decoding can hide true strength differences.

### Phase 1: Data Schema Changes

#### 1. Add per-game result target

Add `game_result_white` in `EventSequence` and `JaggedBatch`.
- values: `+1` if `result == "1-0"`, `0` if draw, `-1` if `result == "0-1"`.

Files:
- `src/imba_chess/data/types.py`
- `src/imba_chess/data/event_builder.py`
- `src/imba_chess/data/collate.py`
- related tests in `tests/test_event_builder.py`, `tests/test_collate.py`

#### 2. Optional but recommended for value quality

Add per-game ELO context for analysis/debias:
- `white_elo`, `black_elo`, `elo_diff` (or just `elo_diff`).

This helps debug whether value is learning "position strength" vs "player strength priors".

### Phase 2: Model + Loss

#### 1. Add value head

In `HSTUChessModel.__init__`:

```python
self.value_head = nn.Linear(d, 3)  # [loss, draw, win] from side-to-move POV
```

#### 2. Forward output additions

Always output:
- `policy_logits`
- `value_logits`

Keep backward compatibility by also retaining `logits` key for policy if needed.

#### 3. Build value targets from existing fields

Use:
- `game_result_white` (per game)
- `turn_id` (per token, side-to-move)
- `seq_offsets` (token -> game mapping)

Target construction:
- Expand per-game result to per-token using `repeat_interleave`.
- Flip sign on black-to-move tokens.
- Map `{-1,0,+1}` -> `{0,1,2}`.

#### 4. Value weighting and masking

Start simple:
- Mask out BOS via `target_move_id != ignore_index`.
- No terminal masking needed for current schema.
- Progress weighting by ply index with `alpha` in `[1.0, 2.0]`.

#### 5. Combined loss

```text
total_loss = policy_loss + lambda_value * value_loss
```

Recommended start:
- `lambda_value = 0.15` (not `0.5` initially)
- `value_progress_alpha = 1.5`

Rationale: protect policy quality while value head starts learning.

#### 6. Config additions

Add to `ModelConfig` and `config/imba_chess.toml`:
- `enable_value_head = true`
- `value_loss_weight = 0.15`
- `value_weight_alpha = 1.5`
- `value_label_smoothing = 0.0` (optional)
- `value_use_class_weights = false` (optional)

### Phase 3: Training Schedule

1. Warm start (optional but recommended):
   - freeze backbone for 1k-3k steps
   - train only heads (`prediction_head`, `value_head`)
2. Joint training:
   - unfreeze all
   - keep `value_loss_weight` low at first
3. Monitor policy regressions:
   - if `top1/top10` drops sharply, reduce `value_loss_weight`.

### Phase 4: Value Metrics (Offline)

Add to eval:
1. `value_ce`
2. `value_acc` (overall)
3. `value_acc_late` (last 25% plies)
4. calibration buckets by predicted `V = p(win) - p(loss)`
5. value metrics by ply phase (opening/mid/end)

Goal: ensure value is actually meaningful, not just fitting global class priors.

### Phase 5: Use Value at Inference (Required for Strength Gain)

Add a new decoding mode in `scripts/eval_vs_stockfish.py`:
- `model_move_policy = value_rerank`

Algorithm (one-ply rerank):
1. Get policy logits on current state.
2. Take top-K legal policy candidates (e.g. `K=8`).
3. For each candidate move:
   - apply move on board
   - evaluate value on resulting position
   - convert value logits to scalar `V = p(win) - p(loss)` (from side-to-move of resulting node)
4. Score candidate:

```text
score(move) = logpi(move) - lambda_rerank * V(next_state)
```

`-V` because after we move, side-to-move is opponent in the next state.

Start with:
- `K = 8`
- `lambda_rerank = 0.35`

This is the fastest path to making value head matter in actual play.

## Ablation Matrix (Must Run)

1. Policy-only checkpoint + greedy decode (baseline).
2. Policy+value training, but greedy policy-only decode.
3. Policy+value training + value-rerank decode.
4. (Optional) policy+value + sampled decode.

Compare all on the same Stockfish ladder setup and seed policy.

## Practical Risks and Mitigations

1. Value collapses to draw prior.
   - Increase late weighting, use class weights, and check late-phase slices.
2. Policy quality regresses.
   - Lower `value_loss_weight`; keep checkpoint selection on policy metrics too.
3. Value learns player-strength bias.
   - Analyze by ELO-diff slices; optionally train value on near-equal-ELO games first.
4. Inference latency increases with reranking.
   - Keep K small; batch candidate evals later if needed.

## File-Level Checklist

1. Data schema:
   - `src/imba_chess/data/types.py`
   - `src/imba_chess/data/event_builder.py`
   - `src/imba_chess/data/collate.py`
2. Model/config:
   - `src/imba_chess/model/hstu_model.py`
   - `src/imba_chess/config.py`
   - `config/imba_chess.toml`
3. Training/eval metrics:
   - `scripts/train.py`
   - `src/imba_chess/eval/ignite_evaluator.py`
   - `src/imba_chess/eval/metrics.py`
4. Engine eval integration:
   - `scripts/eval_vs_stockfish.py`
5. Tests:
   - `tests/test_event_builder.py`
   - `tests/test_collate.py`
   - `tests/test_hstu_model.py`
   - plus eval metric tests if new metrics are added

## Immediate Next Step

Implement Phase 1 and Phase 2 first (data fields + value head/loss), then wire Phase 5 (`value_rerank`) before judging impact on Stockfish.
