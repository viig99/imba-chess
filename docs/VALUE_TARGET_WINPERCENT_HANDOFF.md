# Fixed-function WDL value target: handoff

Written 2026-09-03 for a fresh session. Self-contained; assumes no memory of the
conversation that produced it. Companion to `docs/loss_audit_2026-09.md`, which
records the state of the current losses and the fixes applied the same day.

This supersedes an earlier draft (`SCALAR_VALUE_HEAD_HANDOFF.md`, deleted) that
proposed a separate scalar value head. That idea is kept in §9 as a rejected
alternative, with the reason.

**Status 2026-09-03: adopted.** The ckpt27 fine-tune scored 0.6640 over 750
games against the ckpt23 protocol's 0.6107 anchor (§6). The preserved model is
`artifacts/checkpoints/best_winpercent_checkpoint_27_hr10=0.9613_sf2200=0.6640.pt`.
The calibration arm A was stopped before the change and is not kept; there is
one value target in the tree. Subsequent training uses the full game stream,
with value loss masked to evaluated plies, rather than filtering whole games.

---

## 1. Decision and scope

**Recipe.** Keep the three-logit WDL value head and all of its inference,
search and rollout plumbing. Replace the corpus-fitted cp→WDL calibration with
Lichess's own fixed win-percent function. Train the value loss **only** on
plies that carry a Lichess `[%eval]`, in every run, fine-tune or from scratch.
Everything that existed to cope with outcome-label noise goes: the outcome
blend and `beta`, progress weighting, Elo weighting on value, the hard-outcome
fallback, the calibration fitting pipeline.

**Why this and not more.** Under engine labels the WDL target is a
deterministic function of a scalar, so any head trained on it encodes that
scalar and nothing else. Swapping head kinds cannot add information and would
cost three inference conversion sites, a checkpoint-conversion path and about
78 test references. Keeping the head makes the change a data-side swap with
zero model or search code churn.

**What it unifies.** Once the value loss is annotation-gated there is no
separate "learn WDL from human outcomes, then fine-tune on engine evals"
stage. The from-scratch config and the fine-tune config are the same recipe
with a different initialisation. That is the two-stage complexity being
removed.

**Scope.** The fixed function is the only value target in the tree; the
calibration pipeline, blend, weighting knobs, policy-KL and rollout-training
hook are deleted (§8). The ckpt27 evaluation passed the adoption gate (§6);
the same full-stream recipe is now the default for continuation and the next
from-scratch run.

---

## 2. Where things stand today

### 2.1 Losses in the tree (after the change)

All in `HSTUChessModel.forward`, `src/imba_chess/model/hstu_model.py`:

| Loss | Form | Fine-tune weight |
|---|---|---|
| Policy | CE over the full UCI vocab, mover-Elo weighted, no label smoothing | 1.0 |
| Value | 3-logit WDL CE, STM POV, against `winpercent_wdl(eval)`; plain mean over tokens with an eval | 1.0 |
| Moves-left | Huber on `log1p(plies remaining)` | 0.05 |

### 2.2 The calibration fine-tune (arm A, stopped)

`artifacts/stockfish_finetune/ckpt_stockfish_low_lr_beta1_8k/` holds the
partial run with the corpus calibration target (started 2026-09-03 06:59,
stopped by hand the same morning). It is not a comparison arm; its config no
longer loads. Delete the directory when the disk is needed.

### 2.3 The pretrained trunk

`artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt` ("ckpt23").
Eval anchor from memory notes: 0.6107 vs Stockfish 2200 over 750 games with
`value_search_halving` at budget 2048 (SE 0.0147). White scores ~76 Elo better
than Black at that setting: always pair openings with colour reversal.

### 2.4 Data facts

- `LichessDataset._extract_plays` (`src/imba_chess/data/lichess_dataset.py`)
  emits `eval_cp_stm` / `eval_mate_stm` per play when `parse_stockfish_evals`
  is on: White-POV eval flipped to side to move, the comment after move k
  attached to the state before move k+1. The first ply and the final position
  never carry an eval.
- Prior-session figures (corpus parquets were removed from `artifacts/corpus/`
  on 2026-09-03, so not re-verifiable from disk): ~15% of qualifying games are
  annotated; within an annotated game ~98.7% of usable plies carry an eval.
  Training keeps the full stream: every game supervises policy and moves-left,
  while only evaluated plies supervise value.
- Games longer than `max_seq_len` are rejected, not truncated.
- **The centipawn scale is not uniform across the corpus.** Lichess evals are
  produced by whatever Stockfish version fishnet ran at analysis time. Since
  Stockfish 15.1 (Dec 2022) the reported cp is normalised so that +100 means
  roughly a 50% win probability in engine self-play; before that +100 meant
  roughly a pawn. Games from 2021 to 2022 sit on the old scale. Every fixed
  function, and the corpus calibration too, averages over this. Lichess
  itself applies one function uniformly, so this recipe is no worse than
  Lichess's own accuracy numbers.

### 2.5 How search consumes the value head

`p_win - p_loss` after a softmax, in `position_evaluator.py`
(`_value_scalar_from_logits`, `_batched_value_scalars`) and
`actor_server.py` (`_batched_value_stm`). Nothing uses the draw column on its
own. None of this changes.

---

## 3. The label: Lichess WinPercent

### 3.1 What Lichess actually uses (verified from source, 2026-09-03)

The function on the accuracy blog page is not blog-only. It is defined once in
scalachess and consumed by Lichess's backend:

- **Definition**: `scalachess/core/src/main/scala/eval.scala`,
  `WinPercent.fromCentiPawns(cp) = 50 + 50 * winningChances(cp.ceiled)` with
  `winningChances(cp) = 2 / (1 + exp(-0.00368208 * cp)) - 1`, `Cp.CEILING =
  1000`. Mate: `WinPercent.fromMate` maps any mate to `±1000 cp` via
  `Cp.ceilingWithSignum`, i.e. win percent 97.5 / 2.5 regardless of distance.
- **Backend consumers** (`lichess-org/lila`):
  `modules/analyse/src/main/AccuracyPercent.scala` (the server-side move and
  game accuracy shown on every analysed game: `103.17 * exp(-0.0435 *
  winDiff) - 3.17 + 1`, windowed volatility weighting, harmonic mean);
  `modules/insight/src/main/PovToEntry.scala` (the `winPercent` and
  `accuracyPercent` insight metrics); the tutor module builds on insight.
- **Client copy**: `ui/lib/src/ceval/winningChances.ts`, same constant and
  clamp, used for puzzle-similarity checks; its mate rule `(21 - min(10,
  |mate|)) * 100` is also clamped to ±1000 in practice.

So the constant is Lichess's site-wide definition of "winning chances from a
Stockfish score", server and client.

### 3.2 The target vector

```
p = 1 / (1 + exp(-0.00368208 * clamp(cp_stm, -1000, 1000)))
mate_stm > 0 → cp_stm = +1000 ;  mate_stm < 0 → cp_stm = -1000
target = [p_loss, p_draw, p_win] = [1 - p, 0, p]
```

**Be explicit about what this is.** The function is symmetric and has no draw
term, so the draw entry is always 0. A WDL head trained only on this target
learns a two-class distribution; its draw logit is driven down and
`p_win - p_loss = 2p - 1`. Functionally that is a scalar head expressed as a
softmax. The three-logit parametrisation is kept because every consumer,
checkpoint and test already speaks it, not because the draw column carries
information. It does not, and it cannot from a Lichess eval (§9).

### 3.3 Why this function, numerically

Side-to-move view. "E" is expected score, "WL" is win minus loss, the quantity
search consumes. Corpus columns are the refit calibration; Stockfish columns
are the engine's own `win_rate_model` (§3.4) at full material.

| cp | corpus E | corpus WL | corpus draw | Lichess E | Lichess WL | Stockfish WL | Stockfish draw |
|---|---|---|---|---|---|---|---|
| -400 | 0.213 | -0.574 | 0.062 | 0.187 | -0.627 | -1.000 | 0.000 |
| -100 | 0.380 | -0.240 | 0.078 | 0.409 | -0.182 | -0.500 | 0.500 |
| -20 | 0.459 | -0.082 | 0.091 | 0.482 | -0.037 | -0.015 | 0.980 |
| 0 | 0.489 | -0.022 | 0.275 | 0.500 | 0.000 | 0.000 | 0.987 |
| +50 | 0.547 | 0.094 | 0.086 | 0.546 | 0.092 | 0.075 | 0.924 |
| +100 | 0.588 | 0.176 | 0.077 | 0.591 | 0.182 | 0.500 | 0.500 |
| +200 | 0.647 | 0.293 | 0.075 | 0.676 | 0.352 | 0.993 | 0.007 |
| +800 | 0.873 | 0.746 | 0.041 | 0.950 | 0.900 | 1.000 | 0.000 |
| mate | 0.932 | 0.866 | 0.049 | 0.975 | 0.951 | 1.000 | 0.000 |

- **Lichess WL tracks the corpus WL within ~0.03 across ±200 cp**, the band
  where move choice actually changes. It is on the same scale as the head's
  current output, so `value_rerank_lambda`, the halving thresholds and every
  `[-1, 1]` fixture stay valid.
- **It is more optimistic in the tails** (+800: 0.90 vs 0.75) because it does
  not model human non-conversion. That is the intended semantic shift: the
  head now estimates engine advantage, not human expected score.
- **It has no exact-0.00 bucket.** 7% of evals are a plain 0.00 and the
  corpus fit gave them 27.5% draw. The fixed function calls them equal like
  any other 0. This is the one thing the fit knew that the function does not.
  It is the price of deleting the pipeline and it is the same for every arm.

### 3.4 Alternatives considered

**Stockfish's own `win_rate_model`** (`src/uci.cpp`, master 2026):
`w = 1000 / (1 + exp((a - v) / b))`, `l` with `-v`, `d = 1000 - w - l`, with
`a`, `b` cubic in clamped material (`as = {-142.72, 372.35, -340.71, 415.23}`,
`bs = {5.94, 15.61, -30.58, 69.64}`, `m = clamp(P+3N+3B+5R+9Q, 17, 78) / 58`),
and reported cp defined by `to_cp = 100 * v / a`. It is fixed, published and
material-aware, and it is the only candidate with a real draw term. It is
rejected because it models **engine self-play at long time control**: at full
material an equal position is 98.7% draw and +200 is 99.3% won (table above).
That saturates WL by ±150 cp and tells the head that the whole band humans
actually play in is a draw. It is also only meaningful for normalised-scale
evals (Stockfish ≥ 15.1), which excludes the 2021 to 2022 part of the corpus.

**The corpus calibration** (arm A) is the thing being replaced: it needs an
extraction script, a fitting script, a parquet, a JSON with 48 bins plus a
zero bucket, and it still has no ply or material dependence.

**A separate scalar head**: §9.

---

## 4. The recipe

```
policy loss   = CE(human move), mover-Elo weighted, label_smoothing = 0
value loss    = CE(WDL logits, [1-p, 0, p]) on tokens with an eval, plain mean
moves-left    = unchanged (0.05)
total         = policy + value_loss_weight * value + 0.05 * moves-left
```

No beta, no blend, no progress weighting, no Elo weighting on value, no
hard-outcome fallback, no policy KL. Tokens without an eval have value weight
0. The policy loss stays on the full stream; only the value loss is gated.
In a from-scratch or continuation run about 15% of games carry value targets.
The value loss is normalized over evaluated tokens in each batch, and batches
contain many games, so this does not simply dilute its configured weight to
15%. The ckpt27 held-out alignment result supports keeping
`value_loss_weight = 1.0` (§6).

---

## 5. Implementation plan

### 5.1 What was implemented (2026-09-03)

- `src/imba_chess/data/stockfish_evals.py`: `winpercent_wdl(cp_stm, mate_stm)`
  with the constants `LICHESS_WINPERCENT_K` and `CP_CEILING`; the only
  cp→target function in the tree.
- `EventBuilder` (`event_builder.py`) takes only the vocab and always emits
  `value_target` `[S, 3]` and `has_value_target` `[S]`; `collate.py` and
  `types.py` carry them as required keys. No optional key structure remains.
- `HSTUChessModel.forward`: one value branch, soft CE against
  `value_target` over `has_value_target & valid_mask`, plain mean, exporting
  `value_loss` and the unclamped `value_weight_sum`. The moves-left head is
  unchanged; the policy loss is unchanged.
- `scripts/train.py`: `parse_stockfish_evals` follows
  `model.enable_value_head` on every split; the train step logs
  `value_tokens` / `value_coverage`.
- `config.py`: `ModelConfig` keeps `enable_value_head`, `value_loss_weight`,
  `moves_left_loss_weight`; the `[expert_iteration]` section is gone and an
  unknown top-level section now fails loudly.
- `config/imba_chess_sf_finetune_low_lr.toml` is the full-stream continuation
  recipe config
  (`checkpoint_dir = artifacts/stockfish_finetune/ckpt_winpercent_low_lr_8k`).
- Tests: `test_stockfish_evals.py` checks the function against the scalachess
  definition, symmetry, clamp and mate rule; `test_event_builder.py` checks
  the per-ply POV flip and masking; `test_hstu_model.py` checks the masked
  mean, BOS masking, zero weight sum and gradient locality.

### 5.2 Launch

```
.venv/bin/python scripts/train.py \
  --config config/imba_chess_sf_finetune_low_lr.toml \
  --resume artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt \
  --lr-override 5e-5
```

### 5.3 Next from-scratch config

`config/imba_chess_v4.toml`: the v3 config with `enable_value_head = true` and
`label_smoothing = 0`. Nothing else: all games train policy and moves-left,
evaluated plies train value, and the recipe has no other value-side keys.

### 5.4 Held-out measurement

The training evaluator already reports `value_loss` pooled by
`value_weight_sum` (`WeightedValueLoss`, `src/imba_chess/eval/metrics.py`)
and val/test parse evals whenever the value head is enabled, so the
fine-tune's held-out `value_loss` is against the same targets it trains on.
Running ckpt23 through `--eval-only` with this config gives the base's value
loss on the same targets for a like-for-like curve. Losses from runs before
this change (outcome or calibration targets) are not comparable.

---

## 6. Experiment result

The WinPercent fine-tune from ckpt23 used constant lr 5e-5 and seed 42. The
calibration arm was stopped before the change and was not evaluated.

The decisive comparison was ckpt27 versus ckpt23. Ckpt23's 750-game anchor was
0.6107 (SE 0.0147, §2.3). Ckpt27 completed the same protocol at 0.6640:
416 wins, 164 draws, 170 losses; 375 games per colour; legal coverage 1.0.
Its empirical SE is about 0.0150. The +0.0533 score-rate improvement is about
2.54 independent-sample standard errors of the difference and roughly +40
protocol Elo. The fixed WinPercent recipe therefore cleared the predeclared
~0.04 adoption threshold.

The result JSON is
`artifacts/eval/winpercent_ckpt27_sf2200_750.json`; checksums and preserved
training evidence are recorded in
`artifacts/eval/winpercent_ckpt27_manifest.txt`.

### 6.1 Held-out value alignment

The full 100,000-game validation split contained 1,214,415 evaluated positions.
For checkpoint 27:

- cross-entropy: 0.607045
- target entropy (irreducible even at perfect prediction): 0.581452
- excess KL: 0.025593
- scalar MAE / RMSE: 0.123561 / 0.202862
- scalar Pearson correlation: 0.888742
- nonzero-target sign accuracy: 0.875169
- mean predicted draw mass: 0.00000094

Only 4.2% of the reported cross-entropy is avoidable KL; the remainder is the
soft target's own entropy. Predictions are somewhat compressed toward zero in
the ±200-to-400 cp bands and tails, but the aggregate bias is small
(mean predicted scalar 0.03393 versus target 0.03106). This result does not
support increasing `value_loss_weight` above 1.0. Full bucket measurements are
in `artifacts/eval/winpercent_ckpt27_value_alignment_val100k.json`.


---

## 7. Move-accuracy weighting of the policy loss (assessed, deferred)

Lichess's `AccuracyPercent` gives every annotated human move a quality score
from the win-percent drop between consecutive plies (`103.17 * exp(-0.0435 *
drop) - 3.17`, plus a +1 bonus for analysis imperfection). It is a natural
per-token weight for the policy cross-entropy and, with the same WinPercent
function now in the tree, costs about five small edits: keep the final
trailing eval in `_extract_plays` so the last move can be scored; emit
`policy_token_weight = clamp(accuracy / 100, 0.1, 1)` from the event builder;
pass it through collate; multiply into `policy_token_weights` next to the Elo
scale; a config flag defaulting off.

Expected effect is small: only the ~15% annotated share can be weighted; at
2000+ most moves score near 100 so the weighting materially touches under 2%
of policy tokens; it filters blunders but never shows the better move; and
Elo weighting already does a crude version. Run it as one arm **after** the
value-target decision, never during it. Pre-check: measure how much total
policy weight it removes on a val pass; under a few percent, skip the play
test.

---

## 8. Cleanup (done 2026-09-03)

Deleted in the same change as the new target:

- `src/imba_chess/data/value_target_blend.py`; `CpToWdlCalibration` in
  `stockfish_evals.py`; `scripts/calibrate_evals_to_wdl.py`;
  `scripts/extract_lichess_evals.py`; all calibration JSONs.
- `[expert_iteration]` keys `beta`, `stockfish_eval_calibration_path`,
  `policy_kl_weight`, `policy_kl_sigma`; the policy-KL branch in `forward`;
  `src/imba_chess/data/policy_target_kl.py`.
- `[model]` keys `value_weight_alpha`, `value_label_smoothing`,
  `value_soft_target_weight_alpha`, `value_soft_target_elo_weighting`,
  `value_hard_target_weight`; the hard-outcome CE path and the progress /
  Elo weighting of value tokens in `forward`; `game_result_white` stays only
  if something else needs it.
- The whole-game `require_stockfish_eval` fast-filter and its config flag;
  value masking alone now determines where the head trains.
- Rename `value_target_soft` / `has_rollout_value_target` to
  `value_target` / `has_value_target` across `event_builder.py`,
  `collate.py`, `types.py`, `hstu_model.py`.
- `scripts/eval_value_loss.py` (superseded by the evaluator's `value_loss`);
  the calibration mode of `scripts/value_head_vs_stockfish.py`.
- The rollout-training hook: `rollout_path`, `load_rollout_lookup` in
  `train.py`, `assert_rollout_checkpoint_consistency`, the rollout branch of
  `EventBuilder`. `scripts/generate_search_rollouts.py` and `rollout_store.py`
  stay as generation / analysis tooling; if search-backed value targets are
  ever wanted again, a rollout row's backed scalar `q` maps to
  `[ (1-q)/2, 0, (1+q)/2 ]`, consistent with §3.2.
- `config/imba_chess_exit_*.toml` (Phase 1a/1b experiment configs; they
  referenced the removed keys and would no longer load), the fitted
  calibration JSONs, `scripts/phase1b_paired_run.sh`,
  `scripts/value_distill_run.sh`, `scripts/value_distill_eval.sh`,
  `scripts/eval_value_by_progress.py`.
- Tests for all of the above: `tests/test_value_target_blend.py`,
  `test_policy_target_kl.py`, `test_eval_policy_kl_loss.py`,
  `test_inline_vs_parquet_equivalence.py`,
  `test_expert_iteration_end_to_end.py`, `test_lichess_eval_extraction.py`.

The loss file then contains policy CE, one masked soft-target CE on the WDL
logits, and the moves-left Huber (its own ablation is a later question).

---

## 9. Rejected: a separate scalar head

The earlier draft proposed replacing the WDL head with a one-output head
regressing the same logistic-squashed score. Leela's 2019 move in the
opposite direction (lczero.org/blog/2020/04/wdl-head) was the prompt to
reconsider: their draw column is real information because it is learned from
self-play outcomes at engine strength. Ours, under any engine-eval target, is
a recoding of a scalar, so head kind changes loss geometry only, which the
eval ruler cannot resolve, while costing new head plumbing, three inference
conversion sites, a checkpoint conversion and ~78 test references. If draw
awareness is ever wanted back, the label source has to change, not the head:
a self-run Stockfish reporting its own WDL, self-play, or human outcomes.

---

## 10. Risks

- **Dead draw column.** By construction. Nothing consumes it today; anything
  future that does must bring its own draw label.
- **Mixed cp scale across engine versions** (§2.4). Shared by every option.
- **Annotated games are self-selected.** Policy stays on the full stream, so
  only the value head sees the subset.
- **Fresh-init from scratch** has value targets on ~15% of games; watch that
  the value curve moves at all in the first epoch, and raise
  `value_loss_weight` if it does not.
- **Eval ruler.** Single 200-game runs of the same checkpoint have spanned
  0.53 to 0.63. Read nothing from fewer than 750 paired games per arm.
