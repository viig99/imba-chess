# Fixed-position scale screen, 2026-09-19

Purpose: screen how much search changes ckpt34's policy and whether those changes
improve an independent engine's assessment. This is not a training experiment or
an estimate of match win rate.

## Protocol

- Frozen ckpt34 SHA256: `5844b09fdde268f5fd2aba363603c43e9c1020d776c2d5294a17c2f912962826`.
- Original monitoring prefix manifest, preserving full move history. First 20
  positions for the pilot; next 30 distinct positions for validation. These are
  human-prefix positions, not a representative sample of all future self-play.
- 200 simulations, top-m 16, depth 32, scales 0 / .003 / .01 / .03 / .1 / .3 / 1.
- Two matched Gumbel training-noise seeds per position and scale. This screens
  the targets generated during training, not zero-noise match performance.
- Stockfish 18, full strength, one thread, 64 MiB hash, 100,000 nodes **per legal
  move**, root restricted to that move, clearing hash before each analysis.
  All legal moves receive scores; no probability mass is dropped.
- Scores and WDL are from the original side to move. Expected score is
  `P(win) + 0.5 P(draw)`, using Stockfish's own WDL model. This is an engine
  proxy, not a calibrated probability for either neural player.
- Target gain is `sum_a (target(a) - actor_prior(a)) * engine_expected_score(a)`.
  Centipawn gain is the analogous expectation on positions where every legal
  move has a finite centipawn score. Mates remain explicit and are never assigned
  invented centipawn values. CP summaries exclude 9 of the 50 positions.
- Average the two noise replicates within each position, then average positions
  equally. Intervals bootstrap positions (2,000 replicates), not individual
  searches. Intervals do not account for Stockfish error or multiple comparisons.
- No surprise loss weighting is applied to this diagnostic. It evaluates the
  search target before per-game training weights and optimizer updates.

## Independent 30-position validation

Gains below are percentage points of engine expected score, not chess match
win-rate gains. KL is target versus stored actor prior, in nats.

| Scale | KL | Target gain | Position-bootstrap 95% interval |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| .003 | .0021 | +.27 | +.07 to +.53 |
| .01 | .0211 | +.83 | +.24 to +1.59 |
| .03 | .1472 | +2.07 | +.49 to +4.19 |
| .1 | .6331 | +2.34 | +.59 to +4.70 |
| .3 | 1.1528 | +1.46 | -.83 to +4.20 |
| 1 | 1.5914 | -.05 | -4.15 to +3.53 |

Paired comparison of .1 minus .03: +.27 points, 95% interval -.55 to +1.09
(10,000 position bootstrap replicates). Thus this sample does not establish
that .1 improves target quality beyond .03, despite much greater policy change.
.03 minus .01: +1.24 points, interval +.25 to +2.51. These are exploratory
comparisons, not multiplicity-adjusted claims of an optimum.

The worst position-averaged target gain was -.30 points at .01, -.64 at .03,
-3.08 at .1, and -49.40 at 1. This flags a potential downside of aggressive
search-value corrections, but does not establish the cause of the earlier
online run's regression.

Interpretation: .01 is a credible conservative candidate for a short training
comparison; .03 is another promising candidate with a larger measured target
gain. The data do not establish a universally optimal scale or guarantee either
will improve a trained checkpoint. Selected-move performance under training
noise is distinct from target quality and is retained in the raw results.

## Reproduction and artifacts

Run `scripts/audit_search_scales.py` with the frozen checkpoint, evaluation config,
monitor seed manifest and an output directory. Defaults reproduce the 20-position
pilot; `--offset 20 --positions 30` runs validation. Results are restartable with
identity checks and cached Stockfish scores. Two tests cover distributional
scoring/mate handling and bootstrap clustering.

Artifacts: `artifacts/eval/scale-screen-2026-09-19/{pilot,validation}/results.json`,
per-run summaries, and `combined-summary.json`. The validation phase took 110
seconds for Stockfish and 64 seconds for the neural sweep (excluding runtime
loading). Pilot neural searches took 40 seconds. There were 700 neural searches
across both samples; no model was trained and no remote run was changed.

`depth_check.py` in that artifact directory re-scores the first ten validation
positions at 400,000 nodes per move, preserving the same neural targets. Its
results are recorded in `depth-check.json`.

## Stronger-reference stability check

Re-scoring the first ten validation positions at four times the Stockfish budget
preserved the signal. These are the same ten positions in both columns:

| Scale | Target gain at 100k nodes/move | At 400k nodes/move |
|---|---:|---:|
| .003 | +.411 points | +.415 points |
| .01 | +1.191 | +1.202 |
| .03 | +3.199 | +3.220 |
| .1 | +3.088 | +3.158 |
| .3 | +.415 | +.473 |
| 1 | -3.067 | -3.039 |

This is a reference-budget sensitivity check, not additional independent sample
size. It supports the initial small-gain finding without proving perfect engine
labels. In particular, .01 and .03 remain sensible candidates for a controlled
short training experiment; no remote settings were changed by this audit.
