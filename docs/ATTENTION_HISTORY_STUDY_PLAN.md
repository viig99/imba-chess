# How much temporal history does HSTU use?

Proposed diagnostic study, not measured results. Keep GQA and all other architecture changes out of this first study, so changes are attributable to temporal context.

## Dataset and conventions

Start with 100 held-out games, sampled with a fixed seed. Sample about 20 decision positions per game, stratified by game progress. Treat games as statistical clusters, not thousands of independent positions. Separate positions with short available histories from positions long enough to test all windows; otherwise early positions artificially favor small windows. Add a targeted repetition/draw, tactical and transposition suite because 100 ordinary games can miss rare failures.

Use windows W = 4, 8, 16, 32, 64 and full. W counts attention keys including the current position: W=16 means current plus 15 earlier positions. Sixteen earlier positions plus current is W=17. Positions in the model correspond approximately to plies, not complete White+Black moves. Record checkpoint, data IDs, exact feature contract and precision.

## Stage A: inspect actual attention distributions

For every sampled query, layer and head, reconstruct the actual temporal attention probabilities from SiLU-transformed Q/K, head-width scaling, learned relative bias and the original causal/game mask. Run in eval mode. Validate that the diagnostic path reproduces ordinary logits. Do not mistake the raw QK score, relative-bias table or UVQK SiLU activations for the attention distribution.

Process one game at a time and reconstruct only sampled query rows; do not materialize attention over a packed batch of unrelated games. Stream summaries rather than keeping every matrix. This is a diagnostic path, not the throughput benchmark: fused attention normally avoids exposing a full attention matrix.

For A[l,h,t,j], record:

- Self mass A[l,h,t,t].
- Window mass M(W) = sum A[l,h,t,j] for max(0,t-W+1) <= j <= t.
- Old-history mass 1 - M(W).
- Conditional old-history mass among non-self entries, with near-zero historical mass reported as such rather than dividing by it.
- The smallest contiguous recent window containing 90%, 95% and 99% of the mass.
- Distance buckets: current, 1–3 plies back, 4–7, 8–15, 16–31, 32–63, 64+; record BOS separately.

Report each layer/head, game-phase strata, per-game aggregates and upper quantiles of old-history mass. A grand mean can hide one specialized head or a rare position that depends on distant information. Softmax weights are normally positive for all valid keys, so there is no intrinsic binary “activated” percentage. Any thresholded count must state its threshold.

Attention weights describe mixing coefficients, not final predictive importance. Value vectors, cancellation, HSTU's normalization/U gate, residuals and later layers all affect the result. A recent token can already encode older context. This is why maps should be paired with interventions, not used as an adoption criterion alone. See [Jain and Wallace](https://aclanthology.org/N19-1357/) and [Wiegreffe and Pinter](https://arxiv.org/abs/1908.04626/).

## Stage B: intervene on the current checkpoint

For each sampled position, compare the full-history prediction with each candidate window, keeping current-board features, absolute positions, relative distances among retained tokens, legal moves and precision fixed.

Primary intervention: impose the window in every temporal layer and recompute the full sequence forward. Reusing cached states computed under full attention leaks old information into the retained states and does not evaluate the proposed sliding-window model.

Separate diagnostic: re-encode a hard last-W raw-position segment. This removes more context than per-layer sliding attention. With eight layers and W=16, the latter has a theoretical receptive field of 1 + 8*15 = 121 positions, whereas hard truncation supplies only 16 raw positions. Handle positional IDs explicitly so position resetting is not an accidental second intervention.

A secondary layer-by-layer window intervention can identify which layers benefit from global access. Do not start by simultaneously changing heads, gate, positional encoding or square attention.

For every window, report:

- Legal-policy Jensen–Shannon divergence from full history, plus mean and tail values.
- Top-move change rate, stratified by the original policy margin; a near-tie flip is not equivalent to a confident disagreement.
- Absolute predicted-value/WDL change, and calibration/Brier changes against available held-out outcomes.
- Policy-target cross-entropy changes when suitable held-out policy targets exist.
- Tactical answer failures and engine-assessed regret of changed moves; disagreement with the full model alone is not proof of weaker chess.
- Repetition/draw failures, transposition/path dependence and worst-case examples.

Use paired differences and confidence intervals clustered by game. Avoid arbitrary universal acceptance thresholds; define tolerances against existing evaluation noise and intended strength/performance goals before selecting a winner.

## Stage C: validate a model trained for the proposed mask

Inference-only windowing is a distribution shift. Sensitivity of the current checkpoint does not prove that a window-trained model needs the same history. Fine-tune or train the most promising one or two windows with matched data/exposure and optimizer budgets, alongside a full-history control. Then perform paired search evaluation at equal neural budgets and equal wall-clock budgets. Only after this should GQA be tested separately and combined.

## Visual report

1. Layer × head heatmap of old-history mass for each W.
2. Cumulative recency curves, with self attention separated and uncertainty across games.
3. Window size versus policy divergence, value error and tactical failures.
4. Scatter plot of removed attention mass versus actual prediction change, exposing cases where attention weight is a poor importance proxy.
5. Clickable board/trajectory examples for the largest regressions, including original and windowed legal policies.

One hundred games can rank candidates and expose clear failures. It cannot establish that rare, strategically important long-history cases never occur, or establish final Elo equivalence.
