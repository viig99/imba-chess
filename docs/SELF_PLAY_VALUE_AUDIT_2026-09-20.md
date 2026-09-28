# Self-play value-objective audit

Inspected the scale-0.03 run through step 5,346 / 3,864,803 exposures. No running
training configuration or model was modified. The Stockfish evaluation continued.

## Findings

The supervised value target in `data/stockfish_evals.py::winpercent_wdl` is a
centipawn-derived logistic split between loss and win, with draw mass identically
zero. Self-play switches to actual terminal one-hot loss/draw/win targets for
every accepted continuation position. This changes both target semantics and the
source of supervision. It is not itself proof of an implementation bug: outcome
supervision is intentional, but the transition from this pretrained model is
substantial.

In the first 20 updates, mean observed draw fraction was .328, predicted draw
probability approximately .00001, and value CE 5.713. Thus the initial value-loss
shock is consistent with the previously suppressed draw output.

Raw metrics (not TensorBoard's downsampled reservoir):

| Metric | Steps 400–600 | Latest 200 updates |
|---|---:|---:|
| Unweighted policy CE | 1.29229 | 1.21871 |
| Search-target entropy | 1.12160 | 1.03039 |
| Learner-target KL | .17069 | .18832 |
| Search-actor KL | .18436 | .19101 |
| Model policy entropy | 1.35396 | 1.25824 |
| Value CE | .77129 | .75845 |
| WDL Brier | .44520 | .44692 |

The falling policy CE is explained by sharper targets; target fitting measured by
KL has not improved over this interval. Values are training-stream averages on
changing data, not a fixed held-out comparison.

The last 1,000 updates have value CE median .701, 10th–90th percentiles .468–1.058,
and range .139–1.916. Observed draw fraction ranges from 0 to 1. Value CE correlates
with observed draw fraction at .387 and conditional win/loss CE at .529. Whole-game
sampling within 1,024-token batches produces correlated outcome labels; draw mix
explains part, but not all, of the volatility. Do not infer a loss bug from a
single batch spike. Global pre-clipping gradient norms have a large outlier
(171.8); the configured global clip is 1.0. Raw norms are not Adam update norms.

## Label and gradient probe

Read nine recent completed remote replay games (three per terminal outcome,
sequence length below 512), without writing remote state. Verified every
supervised token's board encoding and side-to-move outcome independently by
replaying the moves: all 1,873 positions passed. Reconstruction additionally
validates legal targets and the actual terminal board. Thirty existing self-play
and surprise-loss tests passed. This is sampled verification, not proof every
possible data path is correct.

Built three outcome-mixed whole-game batches below 1,024 tokens and ran the actual
model and self-play loss on CPU, without optimizer updates. Compared ckpt34 with
actor52 on identical batches. Measured policy and value gradients separately on
shared trunk parameters, excluding private heads and the tied input/output move
embedding. Dropout is zero. These are current training replay samples, not held-out
quality measurements or representative random samples.

| Model | Value/policy trunk gradient norm ratios | Gradient cosines |
|---|---|---|
| ckpt34 | 5.36, 3.18, 3.47 | .238, .072, .124 |
| actor52 | 1.94, 3.38, 5.23 | .058, .058, .058 |

At equal loss coefficients, value gradients can dominate the shared-trunk raw
gradient norm. Cosines are positive, so this probe does **not** demonstrate direct
opposing gradients. Nor does it prove value learning causes strength loss;
optimizer preconditioning and parameter history matter. It makes a controlled
value-objective ablation more informative than another unstructured scale sweep.

## Proposed isolation experiment

From identical ckpt34 weights and optimizer initialization, use frozen replay,
identical batches/exposures, and fixed surprise settings. Compare the current
joint objective with a diagnostic policy-only arm, then an arm where value loss
updates only the private value head (detach its shared input). The private-head
arm still allows policy training to move the shared features, so it does not freeze
value predictions. Measure fixed held-out policy quality, value ranking/calibration,
and paired strength. A lower-value-weight arm can follow if evidence implicates
value updates. Do not change labels to artificial draws, discard terminal
supervision, or deploy a new objective solely from this probe.

Evidence and reproducible probe script:
`artifacts/loss_audit_2026-09-19/{q003_raw_summary.json,current_replay_sample.json,gradient_probe.py,gradient_probe.json}`.
