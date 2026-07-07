# Search-Backed Value Distillation (Expert Iteration, Phase 1a) — Design

## Purpose

The trunk's value head is trained on a **constant-per-game** target: every
ply in a game gets the identical WDL class derived from `game_result_white`
(`hstu_model.py:416-449`), only reweighted by `progress^value_weight_alpha`
and the Elo scale. A position that was objectively winning at move 40 but
lost to a later blunder gets the same "loss" label as the actually-lost
position at move 15 — the target describes the game's ending, not the
position's own quality. This is a different, lower-resolution problem than
the one the standalone value net (`2026-07-05-value-net-distillation-design.md`)
was built to fix: that net supplies an independent *opinion* blended at
inference time, and its failure (blend's edge shrank from +0.115 to -0.035
as the trunk improved, see README §Results) diagnosed a *semantics*
mismatch (perfect-play values misrank small edges against a beatable
opponent) — it says nothing about whether the trunk's own head has room to
improve from better-*resolved* training labels, since it was never
retrained on anything but the constant per-game target.

`value_search_halving` already computes a much better per-position estimate
today, for free, as a side effect of picking a move: it backs up a value
over several plies of real lookahead with explicit forcing-move refutation
checking (this mechanism is why search already beats `greedy`/`value_rerank`
at inference using the *same* value head — 0.34→0.915 vs SF1400 — no
external oracle involved). This design distills that per-position estimate
back into the trunk's own value-head training, replacing/blending the
constant per-game label with a position-resolved one.

## Roadmap

This doc specs **Phase 1a** in full and documents Phase 1b/2 as scoped
future work sharing the same foundation:

| Phase | What changes | New hyperparameters | Status |
|---|---|---|---|
| **1a** (this doc) | Value target only: `blend(real_outcome, search_backed_value; β)` | `β` | Build now |
| 1b | + Policy target: `evals_spent`-normalized distribution over search arms, confidence-gated against the human move | `+ m` (margin) | Designed below, deferred pending 1a's result |
| 2 | Self-play game generation + GRPO-style policy-gradient update using full-trajectory outcomes | N/A (different algorithm family) | Out of scope, documented as future work |

Phase 1a isolates the one piece of this idea backed by direct diagnostic
evidence (label resolution) from the piece backed by analogy to
AlphaZero/ExIt (policy relabeling helps). Shipping 1a alone means any score
movement is attributable to a single change, not a confound between two
simultaneous ones — the same attribution discipline the e14 SF2200 round
flagged as skipped ("attribution between checkpoint/budget/depth was not
decomposed... judged not worth the eval time" — this time it's cheap enough
to not skip). Policy relabeling (1b) reuses the same rollout data and is a
pure addition, not a rework, if 1a's result motivates going further.

## Part 1: Rollout generation — `scripts/generate_search_rollouts.py`

New standalone script; generates data, trains nothing (mirrors
`scripts/train_value_net.py`'s role as a small, focused, non-Ignite
script).

1. Load the current best trunk checkpoint and build a `PositionEvaluator`
   the same way `eval_vs_stockfish.py` does today (same model, same board
   encoding — this is the mechanism that makes "search corrects the head's
   own blind spots" apply here with zero new inference code).
2. Iterate the **training split only** of `LichessDataset` (val/test stay
   untouched — rollouts must never leak into held-out eval metrics).
   Replay each game ply-by-ply with `chess.Board`; sample a bounded subset
   of plies per game (config knob, e.g. every Nth ply plus a random
   offset, to bound total search calls — exact sampling rate is a
   plan-time/compute-budget decision, not a design-time one).
3. At each sampled position, call `select_value_search_halving` with the
   same `HalvingConfig` used at inference (config-driven — budget/depth
   should match whatever the live eval protocol uses, so the label
   reflects the same lookahead strength the model will actually rely on).
   `select_value_search_halving` only scores the root's *children* (the
   arms), never the root position itself — so also make one extra
   unsearched `evaluator.evaluate` call at the root to capture its raw
   3-way WDL split, needed for Part 3's draw-mass-preserving formula.
4. Record one rollout row per sampled position (schema below), including
   the human's actual next move, the game's real final outcome (both
   already known — no extra computation), and the root's raw WDL split
   from step 3. This must be a **frozen snapshot** from the checkpoint
   that generated the rollout, not recomputed live during training — the
   training model's own weights change every step, so recomputing it live
   would make the value target a moving one instead of a fixed label.

Rollout generation is fully decoupled from which phase consumes it: 1a only
reads `backed_value` off the best arm, 1b additionally reads
`evals_spent`/`log_prior` across all arms. Recording the full top-M arm
data now means 1b never needs to regenerate rollouts.

## Part 2: Rollout data structure

Parquet under `artifacts/rollouts/`, one row per sampled position, fixed
top-M(=16) width (matching `HalvingConfig.top_m`, padded/masked when fewer
legal moves) — flat columns rather than nested structs, so it collates the
same way the existing per-token id arrays do:

```
game_id: str
ply: int                        # index into the game's token sequence
human_move_uci: str
human_move_backed_value: float | null   # null if human move fell outside
                                         # the searched top-M and wasn't
                                         # separately evaluated
real_outcome_stm: int            # {-1, 0, 1}, side-to-move POV at this ply
best_arm_move_uci: str
best_arm_backed_value: float     # [-1, 1], side-to-move POV — the value
                                  # target source for Phase 1a
root_wdl_unsearched: list[float, len=3]  # frozen snapshot, root position,
                                          # pre-search value head output
                                          # (Part 3's p_draw0 source)
arm_move_uci: list[str, len=16]         # padded with "" past legal count
arm_backed_value: list[float, len=16]   # padded with 0.0
arm_evals_spent: list[int, len=16]      # Phase 1b only; recorded now
arm_log_prior: list[float, len=16]      # Phase 1b only; recorded now
search_config: struct             # budget/depth/rounds, for provenance
checkpoint: str                   # which checkpoint generated this rollout
```

## Part 3: Value target construction (Phase 1a)

Two things need blending: the *scalar* backed value (`[-1, 1]`, no draw
information — same `p(win) − p(loss)` convention as the rest of the search)
and the *discrete* real outcome (today's exact one-hot class). Blending
scalars first and reconverting to a 3-vector would need an arbitrary
draw-mass assumption; instead, borrow the draw mass from the trunk's own
un-searched value-head output at the root position (which already has a
genuine 3-way split, before search touches it), and rescale the win/loss
split to match the searched value:

```
p_loss0, p_draw0, p_win0 = root_wdl_unsearched   # frozen snapshot, from
                                                  # the rollout row (Part 2)
p_win    = (1 - p_draw0 + backed_value) / 2
p_loss   = (1 - p_draw0 - backed_value) / 2
p_draw   = p_draw0
searched_vec = [p_loss, p_draw, p_win]

real_outcome_vec = one_hot(real_outcome_stm)   # exactly today's target
blended_vec = (1 - β) * real_outcome_vec + β * searched_vec
```

This preserves the model's own sense of "how drawish is this position"
(search doesn't change that) while updating the win/loss split from the
deeper lookahead. `β = 0` reproduces today's exact target (regression
safety); `β = 1` is the pure searched estimate.

## Part 4: Training integration

**No new trainer.** `scripts/train.py`'s Ignite loop, `LichessDataset`, and
`HSTUChessModel` are all reused:

- Dataset/collate gain an optional rollout lookup keyed by `(game_id,
  ply)`: when building a game's token sequence, attach a per-token
  `value_target_soft [seq_len, 3]` (zeros where absent) and a
  `has_rollout_value_target [seq_len]` mask — the same "one more optional
  per-token array" pattern `piece_ids`/`turn_id`/etc. already use through
  `packing.py`/`collate.py`. A game with zero sampled rollouts collates
  byte-identically to today.
- `hstu_model.py`'s value-loss branch (`hstu_model.py:439-449`) becomes
  per-token conditional: tokens with `has_rollout_value_target` use soft
  cross-entropy (`-(target * log_softmax(logits)).sum()`, the same
  function already used in `train_value_net.py`) against `blended_vec`;
  all other tokens keep the exact existing hard-class CE path. `β = 0`
  or an unset rollout file reproduces current behavior exactly, matching
  the "unset = unchanged" convention the value-net checkpoint config
  already established.
- New `[expert_iteration]` config section: `rollout_path` (optional,
  default unset), `beta`. Unset → today's training, byte-identical.

## Part 5: Phase 1b (designed now, deferred pending 1a's result)

Adds a policy target from the same rollouts, gated by a confidence margin
so only *confident* search disagreements override the human-move label:

- Target distribution: normalize `arm_evals_spent` across the searched
  arms (`π_i ∝ evals_spent_i`) — sequential halving already spends more
  budget on arms that survive rounds, so this is a direct analogue of
  AlphaZero's visit-count policy target, available with no new
  computation. Alternative considered: softmax over `arm_backed_value /
  τ` — kept as a fallback if the `evals_spent` distribution proves poorly
  shaped (too peaked/flat) in practice.
- Confidence gate: only replace the human-move one-hot with the search
  distribution when `best_arm_backed_value − human_move_backed_value >
  m`; below that margin, keep the human move as the target unchanged (a
  small disagreement is search noise, not a diagnosed blunder — and the
  human move has no ground-truth status the way the real game outcome
  does, so it isn't blended in with a floor probability the way the value
  target is).
- Loss: KL divergence between the model's policy softmax and the gated
  target, replacing the existing hard CE for gated tokens only.

## Tuning methodology

`β` (and later `m`) are *training* hyperparameters, not inference-time
knobs like `value_net_alpha` — evaluating a candidate requires retraining,
not just rerunning eval with a flag. Three-stage funnel to keep this
affordable:

1. **Label-level statistics, no training.** Once rollouts are generated
   once, sweep a small `β` grid (e.g. 3 values) purely as post-hoc
   arithmetic over already-computed `blended_vec`s: what fraction of
   sampled positions have their WDL argmax class flipped relative to the
   `β=0` baseline. Picks a sane starting grid before spending GPU time.
2. **Validation-loss proxy.** Short fine-tune resumes (not full retrains)
   from the current best checkpoint at the shortlisted `β` values; compare
   held-out value loss and policy hr@10 on the existing val split (same
   metrics `train.py` already tracks). Narrows to 1-2 finalists.
3. **Live eval confirmation.** Only the finalist(s) get the full 100-game
   eval-vs-Stockfish protocol at the established SF2200, budget 2048,
   depth 8 rung, compared against the current 0.595 baseline (checkpoint
   23, α=0) — the same discipline every other sweep in this repo (α, λ,
   rounds) has used.

## Testing

- `blended_vec` formula: sums to 1, `β=0` reproduces `real_outcome_vec`
  exactly, `β=1` reproduces `searched_vec` exactly, draw mass equals
  `p_draw0` regardless of `β`, monotone in `backed_value`.
- Rollout generation determinism: fixed seed + checkpoint + config
  reproduces identical rollout rows (search itself is already
  deterministic under fixed seed; this test covers the sampling/replay
  wrapper only).
- Data join: a game with a rollout at ply k gets `has_rollout_value_target
  = True` only at token k, soft target matches the row exactly; a game
  with no sampled rollouts collates byte-identical to a `rollout_path`-unset
  run (regression safety).
- Small-scale end-to-end training smoke test: a handful of steps against a
  tiny synthetic rollout file — loss finite, no NaNs, soft-CE path actually
  exercised (not silently falling back to hard CE).

## Acceptance protocol

Three-stage funnel above; final gate is the live SF2200/2048/d8 eval
beating the 0.595 baseline. Secondary/leading indicator: held-out value
loss on the existing val split, tracked during the validation-loss-proxy
stage before any live-eval compute is spent.

## Out of scope (v1 / Phase 1a)

- Policy relabeling (Phase 1b, designed above, not built here).
- Self-play generation and any RL/policy-gradient update (Phase 2).
- Iterative rounds (regenerating rollouts from an improved checkpoint and
  retraining again) — this doc covers one round; iteration is a follow-up
  once round 1 validates the approach.
- Rollout sampling from val/test splits.
- Any change to the standalone value net or its inference-time blend
  (`value_net_alpha`) — orthogonal, already-shipped infrastructure.
