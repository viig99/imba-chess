# Gumbel search and policy-surprise audit — 2026-09-19

Audited repository baseline: `3f34119`. Reference: Google DeepMind mctx
`88f92056a420c2673bed282f5a0c00211f126e78`, which was also upstream HEAD at the time
of this audit. The repo already pinned that exact revision and included upstream
depth-one search fixtures. This audit extends that comparison to multi-depth
search, native statistics, and the new surprise-weighted training path.

## Findings and changes

1. **Fixed: numerical false positives in policy surprise.** Stored actor priors
   are FP32 log-softmax results, while Gumbel targets use Python double-precision
   softmax. Even when search does not change the policy, residual log-normalizer
   errors can look like KL divergence. With uniform policies over 1–60 legal
   actions, apparent KL reached `6.20e-8`, and game-relative weights ranged from
   `0.5855` to `1.6465`. `dataset._policy_surprise` now normalizes stored actor logs
   and target mass in double precision and treats divergences at or below `1e-12`
   as numerical zero. This is much smaller than the `1e-8` relative-weight
   regularizer. The reproduction now returns all weights equal to 1. True small
   positive KL (`2e-10`) is retained. No current learner prediction enters this
   calculation. Existing historical loss diagnostics keep their prior meanings.

2. **Fixed test coverage: several old KL fixtures used unnormalized priors.**
   For example, logs `[-2, -100]` are not a log-probability distribution. Those
   fixtures could not catch the normalization issue. They now use valid
   distributions, alongside offset-invariance and actual FP32-path regressions.

3. **Evaluation protocol difference, not a demonstrated search bug:** the paired
   evaluator, including matches against ckpt34, historically used Gumbel noise
   for both players. The paper's main perfect-information evaluation used zero
   Gumbel noise, and also reported a stochastic-evaluation ablation. Its training
   experiments additionally used early-game visit-count exploration; our
   collector always plays the sequential-halving recommendation. These are
   experiment-replication differences. They do not establish the cause of the
   observed training decline. [Paper, Appendix E and Figure 8](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf).

   `eval_self_play.py` now uses zero noise by default for both neural players,
   including the neural side of Stockfish matches, without a CLI flag. The
   separate `eval_vs_stockfish.py` already used zero noise. The result identity
   records the protocol, and resuming noisy progress as zero-noise is rejected;
   existing noisy result files require a new output path. Self-play training
   retains its existing exploration. An explicit internal API override remains
   available for controlled noisy-protocol comparisons.

4. **Compact legal-action search matches mctx; masked-action padding differs.**
   All direct comparisons using the same compact action set passed. However,
   mctx's Q transform includes unvisited *invalid* padded actions in its extrema.
   When every legal action has been visited and the mixed value lies outside
   their Q range, a padded mctx call normalizes differently from our legal-only
   implementation. This is a real representation-dependent difference, not a
   proven defect in legal-only search. It is deliberately left unchanged during
   the noise experiment.

   Example: raw value `-0.5`, two equally likely legal actions, visits `[1, 1]`,
   Q `[0.1, 0.1001]`, scale 1.0. Compact search gives target approximately
   `[0, 1]`; upstream with one masked invalid action gives `[0.4936, 0.5064]`.
   Recomputing targets on the 2,136-position local benchmark replay found 9
   changed targets (TV > `1e-6`), maximum TV `0.5653`, overall mean TV `0.000911`.
   Those data used scale 0.1. This is a final-target counterfactual on stored Q,
   **not** a full search rerun or an estimate of strength impact. Matching this
   artifact would require defining the invalid-action representation first;
   adding fictitious moves just to imitate it is not recommended without an
   isolated experiment.

## Search comparison

| Component | Local behavior | Reference comparison |
| --- | --- | --- |
| Root candidate scheduling | Visit-count sequential halving, exact budget including non-powers of two | Same schedule |
| Root ranking | Fixed noise + centered actor logit + transformed Q, constrained by eligible visit count | Same score |
| Final move | Highest score among most-visited actions | Same recommendation |
| Missing Q | Raw node value blended with prior-weighted visited Q | Same completed-Q estimator |
| Q transform | Node-local range normalization, epsilon `1e-8`, `(50 + max visits) * scale` | Same default transform; scale is configurable |
| Interior selection | Improved-policy probability minus `visits / (1 + total visits)` | Same deterministic selection |
| Policy target | Softmax of actor logits plus completed transformed Q, all legal actions | Same target, no direct Gumbel term |
| Backup | Side-to-move values, one sign reversal per edge | Equivalent to reward 0 and discount -1 |
| Revisited cutoffs | Reuse deterministic cached evaluation | mctx re-evaluates; same result for deterministic evaluation |
| Terminal leaves | Exact chess outcome, no inference or further expansion | Compared using absorbing terminal states |

Reference implementations:
[halving](https://github.com/google-deepmind/mctx/blob/88f92056a420c2673bed282f5a0c00211f126e78/mctx/_src/seq_halving.py),
[action selection](https://github.com/google-deepmind/mctx/blob/88f92056a420c2673bed282f5a0c00211f126e78/mctx/_src/action_selection.py),
[Q transforms](https://github.com/google-deepmind/mctx/blob/88f92056a420c2673bed282f5a0c00211f126e78/mctx/_src/qtransforms.py),
[policy output](https://github.com/google-deepmind/mctx/blob/88f92056a420c2673bed282f5a0c00211f126e78/mctx/_src/policies.py),
[tree search and backup](https://github.com/google-deepmind/mctx/blob/88f92056a420c2673bed282f5a0c00211f126e78/mctx/_src/search.py).

The direct multi-depth comparison executes upstream mctx with CPU JAX and fixed
noise. It covers 22 synthetic alternating-player trees, branching factors 2–7,
budgets 3–200, maximum depths 1–6, terminal absorption, repeated depth cutoffs,
candidate counts above/below the branching factor, and scales 0.1 and 1.0.
Chosen actions and root visit vectors match exactly; maximum target difference
was `2.22e-16`, and maximum root-Q difference `1.11e-16` with reference x64 enabled.
The local implementation uses the actual production Rust selectors and backup;
only environment transitions and neural values are synthetic. This complements,
but does not replace, the existing chess terminal/history/inference tests.

Checked-in fixtures and regeneration script:
`tests/fixtures/gumbel/multidepth_mctx.json` and
`tests/fixtures/gumbel/generate_multidepth_reference.py`. Ordinary tests do not
need JAX/mctx. The isolated reference environment used JAX 0.11.2 and Chex 0.1.92.
The original depth-one reference tests and native parity tests also pass.
A further direct upstream check compared 200 randomized completed-Q/root/interior
selection cases (maximum Q error `5.68e-14`) and all 32,768 schedules for candidate
counts 1–64 and budgets 1–512; all passed. Evidence and the scalar comparison
script are under `artifacts/gumbel_audit_2026-09-19/`.

## Surprise weighting semantics

The implemented method remains the requested **policy-loss-only adaptation**.
KataGo's documented method changes training sample frequency, and its implementation
also has reduced-search/value-surprise handling. Our method retains every value
target, uses the stored unnoised actor prior, excludes zero-base-weight targets
from normalization and policy loss, caps before normalization, and leaves missing
priors at multiplier 1. It is not an exact reproduction of KataGo's training
pipeline. The method is documented in KataGo's later methods notes (introduced
in its g170 run), rather than as a Gumbel/mctx component.
[KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#policy-surprise-weighting),
[KataGo training-row weights](https://github.com/lightvector/KataGo/blob/master/cpp/program/play.cpp).

The weighted-loss denominator and game normalization match the agreed design.
Tests cover zero weights, missing priors, padding, detached weights, equal/zero
surprise, base eligibility, unchanged WDL gradients, disabled-path loss/gradient
parity, and legacy replay/checkpoint behavior. The new normalization fixes only
surprise computation; it does not alter search targets, search budget, value loss,
replay sampling, or optimizer settings.

The available configuration of the original remote scale-1.0 run has weighting
disabled. This edge case therefore cannot explain that run's regression. The
newer scale-0.1/surprise run is a separate experiment; the local fix does not
retroactively change its already-generated checkpoints or a running process.

## Controlled evaluation follow-up

`scripts/compare_gumbel_eval_noise.py` freezes the retained remote actors 73, 53,
and 33, ckpt34, the original config and the original monitoring openings. It
reuses their completed noisy controls and reruns each actor against ckpt34 with
zero noise. Both players change protocol together. Search stays at 512
simulations, scale 1.0, top-m 16 and depth 32; there are 50 color-swapped opening
pairs (100 games) per checkpoint. Inputs and identities are checked, and the
summary reports the paired bootstrap interval of the score difference across
openings. This comparison measures the evaluation-noise effect, not the training
scale effect.

Campaign output: `artifacts/eval/gumbel-noise-audit-2026-09-19/`.
`status.json` reports progress; `summary.json` contains completed comparisons.
Individual historical controls remain in `actor-*/noisy.json`, and new results
in `actor-*/zero/results.json`. No remote training was restarted or modified.

Completed checkpoint 73 comparison: historical noisy score **32.0%** (paired
95% interval 25–39%); zero-noise score **31.5%** (24–39%), with 29 wins,
5 draws and 66 losses. Zero minus noisy is **−0.5 percentage points**, paired
95% interval **−8.5 to +8.0 points**. Thus its regression against ckpt34 survives
removing evaluation noise. This does not establish that training scale 1.0
caused the regression or that scale 0.1 is stronger. Checkpoint 53 also remains weaker: noisy **38.0%** (30.5–45.5%),
zero-noise **37.0%** (28.5–45.5%). Its change is −1.0 percentage point
(paired 95% interval −9.0 to +7.0). Checkpoint 33 could not be rerun: the saved candidate checkpoint disappeared
from both the frozen campaign and original evaluation directories before its
evaluation started. Completed results for 53 and 73 remain available.

Validation after making zero noise the evaluation default: **2,299 tests passed**,
19 extended tests deselected; three existing warnings.
