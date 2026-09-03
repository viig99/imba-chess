# Stockfish supervision: handoff

Written 2026-08-23 for a fresh session. Self-contained; assumes no memory of
the conversation that produced it.

This document is deliberately **factual**. It records the question, the data,
the code, and what has already been measured. It does not propose a design --
that is the new session's job, and the request was explicitly to start from a
clean slate and prioritise a simple one.

---

## 1. The question

Lichess ships Stockfish evaluations for a subset of games. We want to use them
as supervision. The open questions, in the asker's words:

1. During training, does the model **keep improving at picking human moves**?
2. **How do we correctly use the Stockfish score?**
3. Instead of "pick the human move, with cross-entropy", do we want something
   like "this move is scored X by Stockfish"?
4. **Does the value head loss need to change?**

And the umbrella question:

> `policy + value_loss_weight·value + moves_left` -- what exactly is the
> information from Stockfish, and what is the best way to incorporate it into
> this loss?

**Stated design preference**: fine-tune from the current checkpoint on the
Stockfish-annotated subset, resuming **at (or slightly below) the learning rate
training left off at**, over the whole annotated dataset. Keep the design
simple. §7 records prior LR results for completeness; the explicit instruction
is *not to over-read them*.

---

## 2. What the Stockfish information actually is

This is the single most important section for question 3, because it bounds
what is possible.

### Format

Evals live inline in the PGN `movetext`, in the comment **after** each move:

```
1. Nf3 { [%eval 0.14] [%clk 0:03:00] } 1... Nf6 { [%eval 0.18] [%clk 0:03:00] } ...
```

- Units are **pawns, from White's point of view**. `0.14` = +14 centipawns for
  White.
- Forced mates appear as `#N` (`#3`, `#-2`) and are **not** on the centipawn
  scale. They are 4.2--4.6% of eval rows.
- Provenance: Lichess's own server-side analysis (fishnet). The database does
  **not** record depth, node count, or engine version. Analysis is
  user-requested, which makes the annotated subset self-selected -- see §3.

### What you get, and what you do NOT get

**You get exactly one number per ply: the evaluation of the position reached
after the move that was actually played.**

You do **not** get evaluations of alternative moves. There is no
`(move, score)` table per position. This directly constrains question 3: you
cannot build a per-move policy target of the form "move X scores Y" from this
data, because only one move per position was ever scored.

Two signals are derivable:

| signal | definition | availability |
|---|---|---|
| **position value** `V(s_t)` | the eval of the position at ply `t` | every annotated ply |
| **played-move quality** `Δ_t = V(s_{t+1}) − V(s_t)` (STM-signed) | how much the played move gained or lost | every annotated ply pair |

`V` is a value-head target. `Δ` is a *scalar quality score for the human's
move* -- it says whether that move was good or a blunder, but says nothing
about which other move would have been better. Neither has been used for the
policy head yet.

### Two alignment facts (silent-corruption class)

Both are already implemented and tested in `scripts/extract_lichess_evals.py`;
they are recorded here because getting either wrong still trains, still reports
a plausible loss, and teaches the wrong thing.

1. **One-ply shift.** PGN puts a comment after the move it follows, so
   `1. Nf3 { [%eval 0.14] }` evaluates the position *after* Nf3.
   `LichessDataset._extract_plays` stores `plays[i]["state"]` as the position
   *before* move `i`. So the eval trailing ply `i` describes ply `i+1`, and is
   keyed `ply = i + 1` to match the `(game_id, ply)` convention that
   `rollout_store` and `EventBuilder` already use.
2. **Perspective.** Evals are White-POV; the value head is side-to-move. The
   position before ply `j` has White to move iff `j` is even.

Token indexing: token `0` is BOS; **token `i+1` holds the position before ply
`i`** and its policy target is ply `i`'s move.

### Centipawns → WDL

`scripts/calibrate_evals_to_wdl.py` fits `P(win/draw/loss | cp)` empirically
from this corpus (~477k positions carrying both an eval and a known result),
rather than borrowing Leela's curve. Measured, and monotone:

| cp (STM) | P(win) | P(draw) | P(loss) | E[score] |
|---|---|---|---|---|
| < −800 | 0.088 | 0.033 | 0.878 | 0.105 |
| −250…−175 | 0.291 | 0.070 | 0.639 | 0.326 |
| −10…10 | 0.377 | 0.224 | 0.399 | 0.489 |
| 125…175 | 0.575 | 0.076 | 0.349 | 0.613 |
| 500…800 | 0.800 | 0.049 | 0.151 | 0.824 |
| > 800 | 0.867 | 0.035 | 0.098 | 0.885 |
| mate for STM | 0.902 | -- | -- | -- |

**Two facts worth carrying forward**: at `|cp| > 800` these players convert
only **86.7%**, and **mate-scored positions convert 85.1%**. The game outcome
is a very noisy function of true position quality at 2000--2400 Elo. That noise
is what the current value target is made of.

### Engine-era question: settled, no restriction needed

Compared 2021 H1 vs 2024 H1 (spanning roughly SF13 → SF16):

```
P(win) at matched cp buckets differ by <= 0.03, most within 0.01
CE(era-calibrated eval || outcome):   2021 = 0.8391    2024 = 0.8395
```

A 0.0004 nat difference. The limiting factor is the humans, not the engine --
both eras' evals are far past what 2000--2400 players can exploit. **Use the
full window; a single global cp→WDL calibration is fine.** (Annotation rate
does rise over time: 12.9% in 2021 H1 → 15.4% in 2024 H1.)

---

## 3. The data: how much there is

### Annotation rate (measured)

| slice | games annotated |
|---|---|
| 50,000 train games | 13.3% |
| 600,000 train games | 14.8% |
| 40,000 val games (2025-07) | 15.4% |
| 2021 H1 (60k) | 12.9% |
| 2024 H1 (60k) | 15.4% |

**Within an annotated game the median is 100% of plies annotated** (only 0.1%
of annotated games miss more than one ply). It is all-or-nothing per game, not
a sample within games. So ~13% of games ⇒ **13.05% of every ply in the corpus**.

For comparison, the search-rollout artifacts cover **1.35%** of plies
(214,499 labels over a 218,232-game training stream, 9.7 labelled plies per
game). The Stockfish evals are ~10x that coverage, cost no GPU, and are an
*outside* teacher -- a search rollout is partly self-distillation, since the
search producing it is guided by the value head being trained.

### Total available

From parquet footers: **~92.5M raw games/month × 54 months (2021-01 → 2025-06)
≈ 5 billion raw games**. After the config filter (`min_avg_elo ≥ 2000`,
`min_time_control_sec ≥ 180`) -- pass rate not precisely measured, plausibly
1--5% -- that is 50--250M games, of which ~14% are annotated:
**roughly 7--35 million annotated games available.**

Measured throughput for building it:

| step | rate |
|---|---|
| materialize (post-filter rows) | ~960 rows/s, ~0.8 KB/game |
| extract evals | ~800 games/s |
| train | 1,500 steps in ~8 min (~53 games/step, ~4,050 tokens/step) |

So ~1M annotated games ≈ 2 h materialize (5.5 GB) + 20 min extract + ~1.7 h
train.

### Selection bias (must be controlled for)

Annotation is user-requested, and strongly correlates with rating and time
control:

| avg Elo | annotated | | time control | annotated |
|---|---|---|---|---|
| 2000--2199 | 9.6% | | 300+0 | 7.1% |
| 2200--2399 | 15.5% | | 180+0 | 11.2% |
| 2400--2599 | 39.8% | | 600+0 | 21.7% |
| 2600--2799 | 64.3% | | 600+5 | 37.3% |

**Any arm trained on the annotated subset must be compared against a control
trained on the same subset**, or "better targets" is confounded with "stronger
games". Game length is not biased (median 71 vs 73 plies).

### Artifacts already built

Under `artifacts/corpus/`:

| file | contents |
|---|---|
| `seed42_train_600k.parquet` | 600,000 train games (materialized, seed-pinned) |
| `train_600k_annotated.parquet` | 88,926 annotated games |
| `train_600k_evals.parquet` | 6,651,745 per-ply evals (cp/mate, STM-signed) |
| `train_600k_value_targets.parquet` | same, calibrated to WDL, **in rollout schema** |
| `val_2025_07.parquet` | 40,000 held-out val games |
| `val_evals.parquet`, `val_eval_targets.parquet` | 463,257 held-out eval targets |
| `cp_to_wdl_600k.json` | the fitted calibration |

Scripts: `materialize_corpus.py` (`--split`), `filter_annotated_corpus.py`,
`extract_lichess_evals.py`, `calibrate_evals_to_wdl.py`,
`value_head_vs_stockfish.py`.

---

## 4. Where the losses are implemented

All in `src/imba_chess/model/hstu_model.py`, inside `forward(..., return_loss=True)`.

```
total_loss = policy_loss
           + value_loss_weight   * value_loss          (line 565)
           + policy_kl_weight    * policy_kl_loss      (line 636)
           + moves_left_loss_weight * moves_left_loss  (line 652)
```

| term | lines | shape |
|---|---|---|
| `policy_loss` | 441--477 | `F.cross_entropy(policy_logits, target_move_id, label_smoothing=0.05)`, per token, Elo-weighted |
| `value_loss` | 492--565 | 3-class CE over W/D/L from STM POV, weighted by `progress^value_weight_alpha` |
| `policy_kl_loss` | 566--636 | arm-based, currently off (settled null) |
| `moves_left_loss` | 637--652 | auxiliary |

### Policy loss, in detail (question 1 and 3 live here)

```python
per_token_policy_loss = F.cross_entropy(
    policy_logits.float(), safe_targets,
    reduction="none", label_smoothing=self.config.label_smoothing)
```

`target_move_id` is the **human move actually played**. Tokens are weighted by:

```python
elo_norm  = clamp((played_by_elo - min_elo) / (max_elo - min_elo), 0, 1)
elo_scale = 1.0 + elo_loss_weight_strength * elo_norm**elo_loss_weight_alpha
```

so stronger players' moves already pull harder. There is currently **no
move-quality term** -- a 2600-rated player's blunder is weighted the same as
their best move. `Δ_t` from §2 is exactly the signal that would change that,
and it is unused today.

### Value loss, in detail (question 4 lives here)

The hard path derives its own target from the game result:

```python
z_token      = game_result_white[token_game_id]
y            = where(turn_id == 0, z_token, -z_token)   # STM POV
value_target = (y + 1).clamp(0, 2)                      # 0=loss 1=draw 2=win
value_weights = progress**value_weight_alpha * valid_mask * elo_scale
```

and there is already a **soft-target override**:

```python
if has_rollout_value_target is not None and value_target_soft is not None:
    per_token_soft_loss = -(soft_targets * log_softmax(value_logits)).sum(-1)
    per_token_value_loss = torch.where(rollout_mask, per_token_soft_loss,
                                       per_token_value_loss)
```

This is the injection point. `torch.where` (not `if mask.any()`) is
deliberate: the training config uses `torch.compile(fullgraph=True)`, which
cannot trace a Python branch on tensor contents.

Note the comment at lines 533--541: covered tokens deliberately keep the same
`progress^alpha` weight, because an earlier attempt to give them a constant
weight made held-out value loss **worse**.

### How targets get in

`EventBuilder` (`src/imba_chess/data/event_builder.py`) takes a
`rollout_lookup: dict[(game_id, ply), RolloutRow]` and emits, per token:

- `value_target_soft` `[T, 3]` -- from `compute_blended_value_target(...)`
- `has_rollout_value_target` `[T]`
- `policy_kl_arm_ids` / `_qhat` / `_mask` -- unused when arms are empty

`compute_blended_value_target` (`src/imba_chess/data/value_target_blend.py`):

```python
blend(real_outcome, searched_value; beta)     # beta=0 => one-hot outcome
```

It reads only `p_draw0` from `root_wdl_unsearched` and reconstructs win/loss
from `backed_value`, so setting `root_wdl_unsearched = (P(loss), P(draw),
P(win))` and `backed_value = P(win) − P(loss)` makes `beta=1` reproduce a
calibrated vector exactly. **`beta=0` has been verified to reproduce the
model's own hard target token-for-token (0 argmax mismatches over 122,215
covered tokens).**

`assert_rollout_checkpoint_consistency` (`rollout_store.py`) normally requires
targets to come from the checkpoint being resumed. Rows whose `checkpoint`
starts with `EXTERNAL_TEACHER_PREFIX` (`"external:"`) are exempt, because a
Stockfish eval was produced by no checkpoint at all. Mixed files still validate
their checkpoint-generated rows.

### Loading a corpus

`dataset.local_corpus_path` routes `LichessDataset.stream()` through a
materialized parquet. It **refuses to shard** (`num_shards > 1` raises), and
`_make_dataset` passes it for the train split only.

**`dataloader.num_workers = 0` is load-bearing whenever `rollout_path` is set.**
Sharding the stream silently zeroes every target and reports nothing --
this destroyed every Phase 1a/1b result before 2026-07-25.

---

## 5. `value_search_halving`, and why the loss shape matters to it

The search (`src/imba_chess/eval/search.py`) is what turns a better model into
Elo. Root arms are scored at `_score_arm` (line 623):

```python
arm.score = backed_root + lam * arm.root_log_prior
```

`backed_root` is a depth-`search_max_depth` minimax over the value head;
`root_log_prior` is the policy's log-probability for that move. Measured over
212,234 real search positions (15.8 arms each):

| | median |
|---|---|
| within-position spread of `backed_value` | 0.6925 |
| within-position spread of `lam·log_prior` | 0.3505 |
| **positions where the policy term changes the argmax** | **34.0%** |
| margin between best and runner-up | 0.092 (p10 0.015, p90 0.350) |

So value carries ~2x the weight, but **policy co-decides a third of moves**,
and **half of all decisions are made on a margin below 0.10**. Both heads
matter; neither dominates.

Two consequences for loss design:

- Search consumes **only** the scalar `p_win − p_loss`
  (`_batched_value_scalars`, `position_evaluator.py`). Draw mass enters only
  through the softmax normalizer.
- Because value enters additively against `lam·log_prior`, **a pure rescale of
  the value is exactly a change of `lam`** (measured: applying a temperature
  `T` is equivalent to `lam_eff = lam/α` with `α` the best global rescale;
  `T=1.757` ⇒ `lam ≈ 0.075`, `R² = 0.979`). Calibration changes to the value
  head are therefore partly redundant with a hyperparameter you already have.

---

## 6. Hyperparameters that may need tuning

### Loss weights (`[model]` in the config)

| param | current | note |
|---|---|---|
| `value_loss_weight` | 1.0 | scales the whole value term |
| `value_weight_alpha` | 0.9 | `progress^alpha` per-token weight |
| `value_label_smoothing` | 0.0 | applied to soft targets too |
| `label_smoothing` (policy) | 0.05 | |
| `elo_loss_weight_alpha` | 1.0 | shape of the Elo curve |
| `elo_loss_weight_strength` | 1.0 | `1 + strength·curve` |
| `elo_weight_min_elo` / `max_elo` | 2000 / 2800 | |
| `moves_left_loss_weight` | 0.05 | |
| `policy_kl_weight` | 0.1 in base config | **set to 0**: settled null, and leaving it on confounds |

### Expert-iteration (`[expert_iteration]`)

| param | current | note |
|---|---|---|
| `beta` | 0.0 | blend weight, outcome ↔ supplied value |
| `rollout_path` | -- | point at the calibrated targets parquet |
| `policy_kl_sigma` | 1.0 | unused when weight is 0 |

### Search (`[eval_vs_stockfish]`) -- affects Elo, not training

| param | current |
|---|---|
| `search_lam` | 0.05 (lower ⇒ **more** weight on value) |
| `search_budget` | 2048 |
| `search_max_depth` | 8 |
| `search_top_m` | 16 |
| `search_refutation_top_r` / `search_expand_top` | 2 / 3 |
| `value_rerank_lambda` / `value_rerank_top_k` | 0.05 / 16 |

`lam` was swept once at 15--45 games/arm, which is pure noise -- it is
effectively untuned. Note it was tuned (such as it was) against the *current*
value head; a better-calibrated value head may want a different `lam`.

### Training

| param | current |
|---|---|
| `max_lr` | 7e-4 (OneCycle) |
| lr at the end of the ckpt23 run | **6.2e-4** |
| `max_tokens_per_batch` | 40960 in the pretrain config, 4096 in the fine-tune config |
| `weight_decay` / `grad_clip_norm` | 0.01 / 1.0 |
| `dtype` / `compile_model` | bfloat16 / true |

---

## 7. Prior experiments (for reference; do not over-read)

The explicit instruction is to **start from a clean slate**. These are recorded
so they are not re-derived by accident, not as constraints.

### Learning rate

`artifacts/lr_probe_kl0/` contains arms at **1e-5, 5e-5, 2e-4, 6.2e-4**. The
conclusion recorded in `scripts/phase1b_paired_run.sh` was:

> `lr 5e-5 (the KL-off probe's cliff: >=2e-4 degrades, <=5e-5 improves)`

6.2e-4 is the LR the ckpt23 run ended at. Caveat in both directions: that probe
was a short fine-tune (~4k steps) at constant LR on a small corpus. A long run
over ~1M annotated games (~20k steps) with a decaying schedule is a different
regime, and the asker has chosen to re-test resuming at the original LR.

### Reference points on the eval ruler

All at SF2200, budget 2048, depth 8, `lam` 0.05, `opening_random_plies` 0,
Stockfish at 40,000 nodes, `concurrent_games` 6, seed 42.

| run | games | score | note |
|---|---|---|---|
| `checkpoint_23` anchor | 750 | **0.6107 ± 0.0147** | the reference; White 0.6627 / Black 0.5587 |
| `kl00` (4,200 plain steps, lr 5e-5) | 200 | 0.6675 | +0.057 vs anchor, z = 2.24 |
| value-distill **control** (1,500 steps, annotated corpus) | 750 | **0.6627 ± 0.0173** | +0.052 vs anchor, z = 2.29 |
| value-distill **distill** (identical, `beta=1`) | 335 (partial) | 0.6642 ± 0.0258 | vs control: **+0.0015, z = +0.05** |

**Plain continued training at 5e-5 has now produced ~+0.05 twice, on
independent runs.** It is the only intervention in this line of work that has
moved Elo more than once.

Colour matters more than most effects being measured (~76 Elo White/Black gap),
so openings must be paired with colour reversal. 750 games/arm gives SE ≈ 0.0147
(≈ ±11 Elo, 1σ); a two-arm comparison resolves ~±30 Elo at 95%.

### Value-distillation result (`artifacts/value_distill/`)

Reproduce with `scripts/value_distill_run.sh` (trains both arms) then
`scripts/value_distill_eval.sh` (750 games/arm vs SF2200, same protocol as the
ckpt23 anchor). Both wrap `scripts/train.py` / `scripts/eval_vs_stockfish.py`
over `config/imba_chess_exit_kl_probe_nw0.toml`.

Two arms from ckpt23, identical corpus (88,926 annotated games), identical
1,500 steps at lr 5e-5, differing **only** in the value target
(`beta` 0 vs 1). 79,580 games consumed by each, single pass, 97.4% ply coverage.

Held out on 46,416 val positions carrying an eval:

| | ckpt23 | control | distill |
|---|---|---|---|
| KL(sf ‖ model) | 0.0664 | 0.0719 | **0.0300** |
| Spearman r (value vs Stockfish) | 0.7971 | 0.8173 | **0.8322** |
| sign agreement | 81.2% | 82.7% | 83.5% |
| CE(model ‖ outcome) | 0.8135 | 0.8127 | **0.8367** |

The distillation **worked on its own terms** -- 58% of the gap to Stockfish
closed, on 6.08M tokens (0.06% of ckpt23's 9.65B). It did **not** convert to
Elo at 335 games (z = +0.05). The last 415 games were not run.

Note the direction of the trade: the distilled model is now *worse* at
predicting human game outcomes (0.8367) than the calibrated Stockfish target
itself (0.8261). Note also that the control's KL *worsened* versus ckpt23
(0.0664 → 0.0719): more outcome-fitting moves the value head further from
Stockfish.

### The value plateau (the original motivation)

Over the last 75% of the ckpt23 pretraining run:

```
train/policy_loss      2.3227 -> 2.1261   (-8.5%)
train/value_loss       0.7773 -> 0.7645   (-1.7%)
train/moves_left_loss  0.2868 -> 0.2820   (-1.7%)
```

The value head effectively stopped learning around step 40k of 235k. It is
0.296M params (1.05% of the model): `Linear(768→384) → act → Linear(384→3)`.

### Policy-KL (settled null)

Distilling the search's *move distribution* into the policy head was measured
over 5 × 200 games (2026-08-22): kl01 0.6225 vs base 0.6300 vs control 0.6675,
z = 1.26. No effect. `policy_kl_weight` should be 0 in any new run.

---

## 8. Measurement infrastructure that already exists

| tool | what it does | cost |
|---|---|---|
| `scripts/value_head_vs_stockfish.py` | held-out KL/Spearman/sign-agreement of the value head vs calibrated Stockfish, bucketed by decisiveness | ~4 min |
| `scripts/eval_value_loss.py` | held-out value loss on val/test | minutes |
| `scripts/label_gate_diff.py` | exact diff of two rollout parquets across all 18 label columns | seconds |
| `scripts/eval_vs_stockfish.py` | the Elo ruler | ~2.5 h per 750-game arm |
| `scripts/match_two_checkpoints.py` | head-to-head, no Stockfish, far more sample-efficient | **never been run** |
| `scripts/value_distill_run.sh` | trains the paired control/distill arms for the value-target experiment | ~1 h |
| `scripts/value_distill_eval.sh` | the 750-game/arm verdict run for those two arms | ~5 h |

`train.py` logs `train/policy_loss`, `train/value_loss`,
`train/policy_kl_loss`, `train/moves_left_loss`, `train/lr`, `train/games`,
`train/tokens` to TensorBoard under the checkpoint dir. In-run validation
(`val_fast`, `val_full`) tracks **policy metrics only** -- `loss_ce`, `ppl`,
`mrr`, `top1/3/5/10_acc` (`hr10` == `val_full/top10_acc`). **There is no
val-side value metric in the training loop**; `eval_value_loss.py` and
`value_head_vs_stockfish.py` fill that gap as separate scripts.

Answering question 1 ("does it keep improving at picking human moves?") needs
nothing new: `val_full/top1_acc` and `val_full/loss_ce` already measure it. For
reference, ckpt23 ended at `val_full/loss_ce` **1.6577** and `top1_acc`
**0.4925**, both still falling monotonically at the last epoch.

### Non-negotiables

- `dataloader.num_workers = 0` whenever `rollout_path` is set.
- Verify `train/value_loss` (and `train/policy_kl_loss`, if used) is non-zero
  and moving before trusting any run.
- Any arm trained on the annotated subset needs a control on the **same**
  subset (§3, selection bias).
- Do not run GPU work while the machine's owner is gaming.
- Never touch system suspend/power settings.
