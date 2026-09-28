# Self-play regression: handoff reconciliation and measured findings

Date: 2026-09-21. Repo HEAD `5b05214`. All results below are read-only analysis of
existing artifacts; no GPU work was run and the live supervised trainer (pid 1819976)
was not touched.

## 1. Handoff reconciliation

Verified as stated:

- Frozen checkpoint `artifacts/self-play-handoff-2026-09-21/last_checkpoint_51750.pt`
  SHA-256 `b688ccdc…b0cb469d` matches `checkpoint.json`.
- `validation.json` reproduces the handoff table exactly (MRR 0.645487, HR@1 47.3615%,
  HR@16 98.4875%, policy CE 1.670444, value CE 0.619123).
- Supervised training is live at constant lr 4.151649e-4, 1280 tokens/batch, 4 workers,
  compile disabled, resumed from `last_checkpoint_11000.pt`. Latest rotating checkpoint
  is `last_checkpoint_52500.pt`; fast-val at that point is top1 0.474054 / HR@16 0.984901,
  i.e. still flat against the frozen 51,750 snapshot.
- The September-20 audit scripts/docs remain untracked in `git status`.

Corrections to the handoff:

- **The `value0-frozen` experiment never ran.**
  `artifacts/self_play/remote5090-ckpt34-s200-q01-value0-frozen-surprise-2026-09-20/`
  contains only an empty `operation/` directory — no `state.json`, no metrics, no steps.
- **Its config would not have been a value-0 run.**
  `artifacts/eval/remote5090-q01-value0-frozen-2026-09-20/config.toml` carries
  `value_weight = 1.0`. The directory name does not describe the configuration.
- **Even with `value_weight = 0`, that experiment cannot answer its question.**
  `trainer.py:67-68` freezes only `model.value_head` parameters. The shared trunk still
  moves under policy gradients, so the *value function* keeps drifting. A frozen
  evaluator does not exist in the current code.

### Inventory: completed vs. not

| Experiment | Status |
|---|---|
| Frozen-model search vs greedy (ckpt34) | completed — `artifacts/eval/search-improvement-2026-09-20/REPORT.md` |
| Policy/value source swap (actor54 × ckpt34) | completed — `artifacts/eval/policy-regression-2026-09-20/` |
| Gumbel/mctx component audit | completed — `artifacts/gumbel_audit_2026-09-19/` |
| Search-scale engine screen | completed — `docs/SEARCH_SCALE_SCREEN_2026-09-19.md` |
| Stage-2 loss/gradient audit | completed — `artifacts/loss_audit_2026-09-19/` |
| Self-play runs at scale 1.0 / 0.1 / 0.1-detached / 0.03 | completed (4 runs) |
| Value-0 / frozen-evaluator control | **never started** |
| Historical learning-transfer scoring | blocked (remote replay unavailable) |
| Any measurement on the new flattened checkpoint | **not started** |

## 2. Self-play never improved, in any run, at any scale

`state.json` for all four runs reports `best = actor-000000.pt` — the starting
checkpoint — after 13, 53, 54 and 73 actor generations respectively.

Every completed promotion screen (50 opening pairs, colour-swapped, 100 completed
games, zero protocol failures) scored below 0.5 against that generation-0 actor:

| Run | screen 1 | screen 2 | screen 3 |
|---|---|---|---|
| scale 1.0 | 0.405 [0.335, 0.475] | 0.400 [0.330, 0.470] | 0.370 [0.300, 0.440] |
| scale 0.1, detached value | 0.465 [0.400, 0.530] | 0.435 [0.360, 0.515] | 0.390 [0.315, 0.465] |
| scale 0.03 | 0.400 [0.330, 0.470] | 0.440 [0.375, 0.505] | — |

8/8 screens below 0.5, mean 0.413, and declining within every run. The scale-1.0 run's
final screen crossed the `upper < 0.45` rollback threshold in `evaluation.py:decision`.
Individual screens are wide (±0.07); the *consistency and direction* carry the signal,
not any single screen.

## 3. What the value head actually learned during self-play

Per-decile training scalars (from each run's TensorBoard):

| metric | scale 1.0 | scale 0.1 detached | scale 0.03 |
|---|---|---|---|
| `conditional_wl_accuracy` first → last | 0.816 → 0.829 | 0.791 → 0.793 | 0.813 → 0.822 |
| `value_loss` first → last | 0.625 → 0.540 | 1.124 → 0.739 | 1.029 → 0.743 |
| `predicted_draw` first → last | 0.044 → 0.076 | 0.120 → 0.193 | 0.250 → 0.268 |
| `observed_draw` first → last | 0.072 → 0.092 | 0.184 → 0.216 | 0.274 → 0.284 |

**Value discrimination never improves.** `conditional_wl_accuracy` — win-vs-loss accuracy
on decisive positions — is flat to within 1-2 points in every run. Essentially all of the
`value_loss` decrease is `predicted_draw` converging onto the marginal `observed_draw`
rate, i.e. the head relearning one scalar base rate that stage 1 had never taught it
(stage-1 `winpercent_wdl` sets draw probability identically zero).

So stage 2 spends its entire budget recalibrating a base rate while contributing nothing
to the ordering that search depends on.

## 4. Why the policy is spared and the value head is not

Batch composition, measured:

| | scale 1.0 | scale 0.1 | scale 0.03 |
|---|---|---|---|
| context tokens / step | 953 | 939 | 937 |
| supervised positions / step | 687 | 720 | 723 |
| positions / game | 74.3 | 95.7 | 100.0 |
| **games / optimizer step** | **~12.7** | **~9.7** | **~9.3** |
| value targets per independent outcome | 54 : 1 | 74 : 1 | 78 : 1 |
| tracked `policy_weight_ess` | — | 574 | 536 |

`trainer.py:152-175` packs **whole games** into a 1024-token microbatch, because the HSTU
history model needs a position's full prefix. Consequently:

- The **policy** target is per-position and diverse: effective sample size ≈ 550.
- The **value** target is `outcome_wdl(game["outcome_white"], board.turn)`
  (`dataset.py:11-13, 84`) — the *final game result*, applied uniformly to every position
  from move 1 to mate, with no bootstrapping from search root values, no discounting and
  no progress weighting (`losses.py:27` is a plain mean). Within one game it carries zero
  positional information; it only alternates sign with side-to-move.

So each optimizer step updates the value function from roughly **ten independent labels**,
against ~720 for the policy — a ~60x effective-sample-size gap in the same batch. This is
the asymmetry the policy/value swap experiment measured as −0.5pp policy vs −15.5pp value.

## 5. Gradient clipping discards magnitude on every single step

| run | mean grad norm | % steps clipped at 1.0 | mean shrink factor |
|---|---:|---:|---:|
| scale 1.0 | 3.84 | **100.0%** | 0.312 |
| scale 0.1 detached (policy group) | 1.99 | **100.0%** | 0.528 |
| scale 0.1 detached (value head) | 3.12 | 86.2% | 0.489 |
| scale 0.03 | 5.65 | **100.0%** | 0.235 |

`grad_clip = 1.0` was inherited from stage 1, where a batch was 40,960 tokens. Stage-2
batches are ~950 tokens, so the gradient norm is always above the clip. Every stage-2
update is therefore a *unit-norm direction* times a constant lr of 1e-4: gradient
magnitude carries no information at all, and the direction is set by whichever ~10 games
landed in the batch. Nothing in the loop can down-weight a noisy step.

## 6. What the policy loss shows

`policy_target_kl` = KL(search target ‖ learner); `search_prior_kl` = KL(search target ‖
generating actor). Their difference is how much closer to the target the learner is than
the actor that produced it:

| run | gap, decile 1 → decile 10 |
|---|---|
| scale 1.0 | 0.188 → 0.206 |
| scale 0.1 detached | 0.079 → 0.081 |
| scale 0.03 | 0.001 → 0.005 |

The gap is reached within the first decile and is then a **constant** for the remaining
90% of every run. A chasing loop does hold a roughly constant KL, so this is not by itself
a defect — but note the ordering: scale 0.03 extracts almost nothing from search
(`search_prior_kl` 0.19, gap 0.01) and still regresses, while scale 1.0 has the most to
learn (`search_prior_kl` 1.9) and regresses *fastest*. More search signal produced worse
outcomes, which is the opposite of the distillation premise.

Also of note: at scale 1.0 the targets are near-deterministic (`policy_entropy` 0.18 nats)
while the model stays at `model_policy_entropy` 1.85 and does not sharpen over 6,101 steps.

## 7. Checks that came back clean

- **Value/position index alignment in `dataset.reconstruct` is correct.** `values` and
  `has_value_target` are length `seq_len` because `_SequenceHistory.seq_token_id` starts
  with a BOS token (`position_evaluator.py:39`); `supervised_indices` recorded before each
  `append_observed_position` lands exactly on the continuation positions. `collate.py:51-57`
  independently enforces the length invariant.
- **WDL perspective is correct.** `outcome_wdl` negates by side-to-move and emits
  [loss, draw, win] matching the value head's output order.
- **Administrative stops do not leak labels.** `collector.py:127-129` clears `targets` on
  any non-terminal exit.

## 8. Assessment

Supported by measurement: self-play degrades playing strength monotonically at every
search scale tried; the degradation lives in the value pathway; during self-play the value
head improves only its draw base rate and never its discrimination.

The mechanism most consistent with all of the above is that **stage 2 replaces a
high-information, per-position, Stockfish-distilled value function with a low-information
terminal-outcome regression, delivered at ~10 independent labels per step, under a clip
that discards gradient magnitude** — while the policy, whose targets are per-position and
diverse, is left roughly intact. Search consults the value at every node, so a value
function that is losing discrimination degrades play even when the policy is unchanged.

This is an objective/estimator problem, not a component bug: the mctx audit, the index
alignment and the label perspective all check out. AlphaZero runs this same pure-outcome
target successfully, but it samples individual positions from an enormous buffer (batch
ESS ≈ batch size) over ~44M games; these runs have ~20-28k games and batch ESS ≈ 10.

### What this does NOT establish

- It does not prove the value head could never improve given far more data.
- It does not rule out an additional target-quality problem at scale 1.0.
- Nothing here is measured on the new flattened checkpoint; all four runs start from ckpt34.

### The decisive next experiment

A **genuinely frozen evaluator**: run self-play with search using a complete, separate,
frozen value network while only the policy trains. The current code cannot do this —
`value_weight = 0` freezes the private head but the shared trunk keeps drifting under
policy gradients, so the value function still changes. This needs a small, isolated
addition (a second frozen model instance used for node evaluation), not a rewrite.

If policy-only training against a frozen evaluator holds or improves strength, the value
pathway is confirmed as the cause and the fix is an estimator change (position-level
sampling or gradient accumulation to raise value ESS, value-target mixing with search root
values, and a clip re-tuned to the stage-2 batch size). If it still regresses, the cause is
upstream in target quality and this diagnosis is wrong.

## 9. The frozen-evaluator control (implemented 2026-09-21)

### Implementation

`src/imba_chess/eval/composed_runtime.py` holds `ComposedRuntime`: two ordinary
search coroutines run in lockstep over identical composed predictions, taking legal
actions and priors from one complete network and the side-to-move value from another.
Each coroutine owns its own network and its own KV tree; only prediction outputs cross
the boundary. This is the composition the 2026-09-20 policy/value swap already used —
promoted out of `scripts/audit_policy_regression.py` so there is one implementation.
That script now subclasses it, pinning its recorded `options` block and its zero-noise
requirement, and its tests still pass.

The one capability added is noisy collection. Lockstep needs both sides on the same
search path, and the only nondeterminism in Gumbel search is the root noise vector
(`gumbel_search.py:187-189`), so the composition draws it once and passes it explicitly
to both sides. That consumes the caller's generator in exactly the pattern a single
ordinary runtime would, so per-game RNG streams stay reproducible.

`scripts/run_self_play.py` gains one opt-in flag, `--frozen-evaluator PATH`, which
refuses to run unless `learning.value_weight = 0`. Collection and both sides of the
screen then play with learner policy and frozen values. The evaluator's hash is written
into `state.json` and a resume that disagrees is rejected. Without the flag nothing
changes.

### Verification

On GPU, with the frozen update-51,750 checkpoint loaded twice:

- **Self-composition is bit-identical to plain search.** Every field of the returned
  `GumbelResult` — chosen move, policy target, visits, Q values, root WDL, priors —
  matches a plain single-network search under the same noise vector.
- **Cross-composition changes only what it should.** Composing the new checkpoint's
  policy with old ckpt34's value keeps the learner's root log-priors exactly, while the
  root value, the policy target and the chosen move all change.
- 14 unit tests cover composition, lockstep divergence detection, noise hoisting
  (including that it matches what a single runtime would draw and leaves the caller's
  generator in the same state), cache clearing and the preserved audit guard; 3 more
  cover the CLI contract. The full collect → train → screen → collect cycle was smoked
  end to end with the evaluator wired in, reporting `value_gradient_norm: 0.0`.
- Cost: 200 MiB per model; 3.1 GiB peak at 32 concurrent games.

### Incidental finding on the new checkpoint

The frozen update-51,750 checkpoint returns root WDL `(0.511, 0.000, 0.489)` — draw
probability exactly zero. The stage-1 `winpercent_wdl` artifact described in §3 is
present in the checkpoint the user wants to use going forward, not just in ckpt34. Any
stage-2 run from it will again spend its early budget relearning the draw base rate.

### Measured laptop throughput (RTX 3070 Ti, 200 simulations, 32 concurrent games)

| arm | positions/hour | iterations/hour | peak VRAM |
|---|---:|---:|---:|
| control (two networks) | 19,657 | 4.80 | 3.1 GiB |
| baseline (one network) | 50,439 | 12.31 | 1.7 GiB |

The control is 2.57x slower, which is the inherent cost of evaluating two networks at
every node. 64 concurrent games was worse (3.78 iterations/hour, 5.8 GiB), so 32 stands.

### The running experiment

Two arms from the same frozen update-51,750 checkpoint, identical in every setting
except the two variables under test, both non-streaming on
`artifacts/corpus/v4_self_play_seeds_4096.json` (3,638 train / 392 monitor prefixes),
200 simulations, scale 0.1, top-m 16, depth 32, lr 1e-4, 18 iterations, screens at
iterations 6/12/18 over 50 colour-swapped opening pairs, observe-only so every screen
measures against the unchanged generation-0 actor:

- **control** — `--frozen-evaluator`, `value_weight = 0`: policy trains, values fixed.
- **baseline** — ordinary joint training, `value_weight = 1`.

Prediction if §8 is right: the control holds near 0.5 or improves, because the
degrading component is pinned, while the baseline reproduces the decline. If the
control also declines, the cause is upstream in target quality and §8 is wrong.

These 50 monitor openings are the same set the earlier runs screened against. That is
deliberate here — the two arms must be matched — but it means a positive control result
needs confirmation on a fresh opening set before it is believed.

## 10. Results: frozen-evaluator control vs matched baseline — INCONCLUSIVE

Both arms ran 18 iterations from the frozen update-51,750 checkpoint, identical in every
setting except `value_weight` and `--frozen-evaluator`, on the same fixed seed manifest,
screening every 6 iterations over the same 50 colour-swapped opening pairs against an
unchanged generation-0 reference. All six screens completed 100/100 games with zero
protocol failures.

| iteration | control (frozen evaluator, policy-only) | baseline (ordinary joint training) |
|---|---|---|
| 5 | 0.590 [0.510, 0.670] | 0.465 [0.395, 0.540] |
| 11 | 0.550 [0.470, 0.630] | 0.550 [0.475, 0.630] |
| 17 | 0.505 [0.435, 0.575] | 0.495 [0.425, 0.565] |

### The effect is not established

Both arms screen against references with identical model weights, so they can be paired
by opening, removing opening-difficulty variance:

| iteration | control - baseline | paired 95% CI | sign-flip p |
|---|---:|---|---:|
| 5 | +0.125 | [+0.025, +0.225] | 0.012 |
| 11 | +0.000 | [-0.110, +0.110] | 0.524 |
| 17 | +0.010 | [-0.075, +0.095] | 0.451 |
| **pooled, 150 pairs** | **+0.045** | **[-0.013, +0.102]** | |

**The pooled interval includes zero.** The iteration-5 result does not replicate at 11 or
17 and must not be reported as a finding on its own.

This was foreseeable and was foreseen. `docs/`-adjacent prior work established that the
same checkpoint spans 0.5275-0.6300 across runs and that roughly 750 games per arm are
needed to resolve effects of this size; these screens are 100 games. The baseline's own
trajectory demonstrates the noise floor directly: 0.465 -> 0.550 -> 0.495 for a model
that only ever trains forward, an 0.085 swing that is pure measurement error. The
iteration-5 gap sits inside that band.

### Neither arm reproduced the historical regression

Control 0.590/0.550/0.505, baseline 0.465/0.550/0.495 — both end at parity, neither
collapses to the 0.370-0.390 the ckpt34 runs reached.

This cannot yet be attributed to the new checkpoint. **These runs are 18 iterations; the
historical declines were measured at iterations 35-65.** The historical *first* screens,
at iterations 17-22, were 0.400-0.465, which overlaps the 0.495/0.505 measured here at
iteration 17. "The decay does not happen on the new checkpoint" and "these runs stopped
before the decay" are not distinguishable from this data.

### Consequence for §8

The control's decay to parity, attributed above to saturation against a fixed evaluator
(`search_prior_kl` falling 0.950 -> 0.860, the learner-vs-actor KL gap peaking at 0.225
then decaying to 0.119, `policy_target_kl` and model entropy both bottoming and
reversing), is equally consistent with the baseline's identical wander to parity — that
is, with both arms being noise around 0.5. The training scalars are real and the turn in
them is real; the claim that the turn *caused* the screen decay is not supported, because
the screen decay is not resolvable from noise.

§8 therefore stands as an evidence-ranked hypothesis, not a demonstrated cause. What
survives unchanged is the measurement base in §2-§6: the historical runs never improved,
value discrimination never improved in any of them, the value target rests on ~10
independent labels per step against ~550 for the policy, and the new checkpoint is
draw-blind (`predicted_draw` 0.000 across all 450 control steps while the real draw rate
reached 20%).

### Head-to-head resolves it

Two separate reference matches carry both arms' sampling error. A head-to-head between
the arms' final actors makes one game one paired observation. Control actor-000018
composed with its frozen evaluator versus baseline actor-000018, 250 opening pairs /
500 games, zero Gumbel noise, via `scripts/match_self_play_arms.py` (which reuses the
production paired-opening protocol rather than reimplementing it). 500/500 completed.

**Score 0.543, 95% CI [0.508, 0.579] — excludes 0.5.**

| design | point estimate | 95% CI | width |
|---|---:|---|---:|
| pooled reference matches, 150 pairs | +0.045 | [-0.013, +0.102] | 0.115 |
| head-to-head, 250 pairs | **+0.043** | **[+0.008, +0.079]** | 0.071 |

The two designs agree on effect size to within 0.002. The reference-match design was not
wrong, only too noisy to resolve a ~4pp effect; pairing the arms directly nearly halved
the interval and cleared zero. This also retro-justifies the retraction above: the true
effect is ~+4.3pp, not the +12.5pp that one screen suggested.

### Decomposition: the advantage is entirely the value head

The head-to-head compares *systems*: control plays as learned policy + frozen original
value, baseline as learned policy + its own learned value. A second match putting **both**
arms' policies on the identical frozen evaluator isolates policy quality. 500/500
completed.

| comparison | score | 95% CI |
|---|---:|---|
| control system vs baseline system | 0.543 | [0.508, 0.579] — control wins |
| control **policy** vs baseline **policy**, same frozen evaluator | 0.481 | [0.448, 0.514] — no difference |

Both matches use the same candidate and the same 250 prefixes; only the opponent's value
source differs, so they pair by opening:

**Value-source effect: +0.0620, 95% CI [+0.0140, +0.1110], sign-flip p = 0.0072.**

Restoring the baseline's original value head makes it ~6.2pp stronger. The policies are
statistically indistinguishable (0.481, interval spanning 0.5, point estimate marginally
*below* it).

## 11. Conclusion: two independent failures, not one

**Confirmed by matched, resolved measurement:** 18 iterations of self-play degrade the
value function by ~6.2pp [+1.4, +11.1]. This independently reproduces the 2026-09-20
policy/value swap result (-15.5pp after 54 iterations) on the new flattened checkpoint,
under a cleaner design that pairs on openings and removes both reference matches'
variance.

**Refuted:** the §8 expectation that pinning the value would make the loop work. It does
not. With the degradation removed, the policy still does not improve — control policy vs
baseline policy is 0.481 [0.448, 0.514], and both arms finish at ~0.5 against their own
starting checkpoint.

So the loop fails in two independent ways:

1. **The value head degrades.** Measured, resolved, ~6.2pp over 18 iterations.
2. **The policy does not improve.** Measured, resolved as a null.

Freezing the value addresses (1) and produces a better playing *system*, but produces no
learning. (2) is a separate defect and is the one that actually blocks self-play from
working.

### Why the two components behave so differently — hypothesis, not measurement

Scale asymmetry fits every observation. The control ran 450 training steps of ~580
positions, roughly 261k position-exposures at lr 1e-4. The supervised checkpoint behind
it has seen orders of magnitude more (a single full validation pass covers 7.6M policy
targets). For the **policy**, the self-play target is a *refinement* of what the model
already does — `policy_loss` ~1.5, `policy_target_kl` moving only 0.786 -> 0.706 before
reversing — so 450 steps is far too small a perturbation to move a converged policy.

For the **value**, the stage-2 target is not a refinement but a *change of objective*:
stage-1 `winpercent_wdl` sets draw probability identically zero, while self-play produces
20-32% draws. `predicted_draw` sat at 0.000 for all 450 control steps. The head therefore
faces a large, immediate gradient toward a genuinely different function — it moves fast,
and it moves downhill relative to the Stockfish-distilled starting point.

This is consistent with §4's low-ESS argument but is **not the same claim**, and this
experiment does not separate them. Two distinct candidate causes for (1) remain:

- **Target-semantics shift** — the draw-blind starting head must relearn a new output
  distribution.
- **Estimator variance** — ~10 independent outcome labels back ~720 value targets per
  step (§4).

### Discriminating test for (1)

Fix draw-blindness first, independently of ESS: warm the value head on draw-aware targets
(or correct the stage-1 `winpercent_wdl` mapping) and rerun the matched pair. If the
degradation disappears, it is target semantics. If it persists, it is estimator variance,
and the fix is gradient accumulation to raise value ESS (§4) or bootstrapped/mixed value
targets (§9 discussion).

### Next step for (2)

(2) is now the binding problem and is untouched by anything tried here. It needs its own
experiment — most cheaply, a far longer policy-only run against a frozen evaluator to test
whether the policy moves at all given enough exposure, since search demonstrably produces
better moves than the raw policy (85-90% vs greedy at budgets 32/200) yet distilling them
has never yielded a stronger policy.

### Standing caveats

- 18 iterations; historical declines were measured at 35-65. Whether the regression
  reproduces at longer horizons on this checkpoint is untested.
- All openings come from the manifest earlier runs screened against. A fresh confirmation
  set is required.
- One run per arm, one seed. No replication.
