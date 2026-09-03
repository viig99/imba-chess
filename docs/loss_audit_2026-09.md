> **Superseded the same day.** The value target and every knob discussed
> below were replaced by a single fixed function and the surrounding code
> deleted; see `docs/VALUE_TARGET_WINPERCENT_HANDOFF.md`. This file is kept
> as the record of what was wrong with the outcome-label / calibration
> design and why.

# Loss audit, 2026-09-03

Audit of the training objectives used by `config/imba_chess_sf_finetune_low_lr.toml`
(the Lichess Stockfish-eval value finetune), the issues found, and what was changed.

## Objectives (all in `HSTUChessModel.forward`, `src/imba_chess/model/hstu_model.py`)

| Loss | Form | Finetune weight |
|---|---|---|
| Policy | CE over the full UCI vocab, mover-Elo weighted tokens | 1.0 |
| Value | 3-class WDL CE, side-to-move POV. Tokens with a Lichess `[%eval]` get a soft target from the cp-to-WDL calibration blended with the real outcome (`beta`); others get the hard game result | 1.0 |
| Moves-left | Huber on `log1p(plies remaining)` | 0.05 |
| Policy KL | CE to a detached self-tilted target over searched arms | 0.0 (off) |

Sign and index conventions were traced end to end and are consistent (White-POV
eval flipped to STM, eval after move k attached to the state before move k+1,
`[loss, draw, win]` ordering shared by hard and soft targets, `beta=1` round-trips
the calibration vector exactly).

## Issues and fixes

### 1. Progress weighting starved early-game engine targets  (fixed)

Value tokens were weighted `progress^value_weight_alpha` (0.9 in this config). That
discount exists because the *game outcome* is a noisy label early on. With `beta=1`
the label is a full-depth engine eval whose quality does not depend on progress, so
the weighting simply removed most of the value gradient from the opening and early
middlegame (ply 10 of 80 got ~15% of the final ply's weight). The Elo scale was also
applied to these tokens, though mover strength says nothing about an engine label.

**Fix.** Three new `[model]` keys, defaults preserving the old behaviour:

- `value_soft_target_weight_alpha` (None = inherit `value_weight_alpha`; `0.0` = flat)
- `value_soft_target_elo_weighting` (bool)
- `value_hard_target_weight` (see 2)

The finetune config sets alpha `0.0` and Elo weighting `false`.

### 2. Mixed label semantics inside one game  (fixed)

`require_stockfish_eval` only checks that a game contains *some* `[%eval]`. The first
ply and the ply before the final position never carry one by construction, and many
Lichess games are partially annotated. Those tokens silently fell back to the hard
outcome label, so the value head was trained toward engine WDL on some plies and the
realised result on their neighbours.

**Fix.** `value_hard_target_weight = 0.0` in the finetune config masks outcome-labelled
tokens whenever the batch carries soft targets. Masking was chosen over a
full-coverage filter because full coverage is impossible (see above) and masking keeps
partially annotated games.

### 3. Label smoothing over the whole move vocabulary  (config change)

`label_smoothing = 0.05` spreads 5% of target mass over ~1900 UCI classes, almost all
illegal in any given position, and puts a floor under the reported policy loss that
the un-smoothed validation CE does not have. It also distorts the softmax that
`value_rerank` and search read. Set to `0.0` for the finetune. A legal-move-restricted
smoothing would be the principled replacement if smoothing is wanted again.

### 4. Calibration smeared the exact-0.00 point mass  (fixed, calibration refit)

7% of the corpus's evals are exactly 0.00 (447k of 6.38M): repetition, fortress,
tablebase draws. Their draw rate (0.275) is ~3x that of positions at +-5 cp. Quantile
binning put them in one bin whose value was then linearly interpolated into the
neighbours, lending e.g. -3 cp a draw probability it does not have.

**Fix.** `scripts/calibrate_evals_to_wdl.py` now holds `cp == 0` out as its own
`cp_zero` bucket and fits the continuous bins on the rest. `CpToWdlCalibration`
returns that bucket for an exact zero and interpolates everything else; old JSONs
without the key still load unchanged. Refit output: `artifacts/corpus/cp_to_wdl_600k_zero.json`
(the config now points at it; the old file is kept for comparison). No dataset needs
regenerating, targets are computed on the fly from the JSON.

Not addressed: the mapping ignores ply and material (an equal opening and an equal
rook ending get the same WDL), and mate is a single bucket per sign regardless of
distance.

### 5. Over-length games were truncated, not rejected  (fixed)

Games longer than `max_seq_len` were cut; the prefix then carried the full game's
result as its value label, told the moves-left head zero plies remained at the cut,
and reached `progress = 1.0` mid-game. `LichessDataset._extract_plays` and the
offline extractor `scripts/extract_lichess_evals.py` now both reject such games,
matching the existing treatment of parse errors.

### 6. Validation never measured the value loss  (fixed)

The training evaluator only tracked next-move CE and hit-rate, so a finetune whose
whole point is the value head had no held-out signal for it. `create_next_move_evaluator`
now takes `track_value_loss`; when the model has a value head the train script enables
it on all four evaluators, which run the loss path and report `value_loss` (pooled by
`value_weight_sum`, so it is the true weighted mean over the epoch, not a mean of batch
means; the exported weight sum is unclamped, so a batch with no evaluated
tokens has weight 0 and does not move the metric). Val/test datasets now parse `[%eval]` comments too when a calibration path is
set, so the held-out value loss is against the same targets the head trains on.
Un-annotated val games are not filtered; with `value_hard_target_weight = 0` they
contribute policy metrics only.

### 7. Different normalisers for policy and value loss  (no change)

Policy is an Elo-weighted mean over valid tokens; value was a mean over its own
progress-and-Elo weights, so the relative gradient scale drifted with each batch's
progress distribution. After 1 and 2 the finetune's value loss is a plain mean over
evaluated tokens and the drift is gone. Left as is.

## Files touched

- `src/imba_chess/model/hstu_model.py`, `src/imba_chess/config.py`: soft-target weighting knobs, `value_weight_sum` output
- `src/imba_chess/data/lichess_dataset.py`, `scripts/extract_lichess_evals.py`: reject over-length games
- `src/imba_chess/data/stockfish_evals.py`, `scripts/calibrate_evals_to_wdl.py`: `cp_zero` bucket
- `src/imba_chess/eval/metrics.py`, `src/imba_chess/eval/ignite_evaluator.py`, `scripts/train.py`: held-out `value_loss`
- `config/imba_chess_sf_finetune_low_lr.toml`: smoothing off, soft-target weighting, new calibration path
- `artifacts/corpus/cp_to_wdl_600k_zero.json`: refit calibration
- Tests updated/added in `tests/test_hstu_model.py`, `test_lichess_dataset.py`, `test_stockfish_evals.py`, `test_eval_metrics.py`, `test_lichess_eval_extraction.py`, `test_train_lr_override.py`

Runs from before this change are not loss-comparable with runs after it: the value
loss is now a mean over a different token set with different weights, and the policy
loss no longer includes the smoothing floor.
