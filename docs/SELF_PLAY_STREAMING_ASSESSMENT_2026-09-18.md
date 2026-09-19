# Streaming self-play: Stockfish results and search-budget options

The first streaming night has not demonstrated a playing-strength improvement.
It successfully generated diverse training data, but that operational success is
separate from strength. No training settings were changed during this assessment.

## Completed Stockfish evaluation

Actor147 is the final checkpoint from the September 17–18 streaming run initialized
from ckpt34. Both tests used strength-limited Stockfish 2400, 40,000 nodes/move,
the initial board, and 100 games. The evaluator source, configuration and Stockfish
binary were frozen from the previous actor204 comparison.

| Search | Wins | Draws | Losses | Score | Actor204 score |
|---|---:|---:|---:|---:|---:|
| Gumbel, 512 simulations | 35 | 29 | 36 | 49.5% | 65% |
| Halving, budget 2,048 | 45 | 32 | 23 | 61% | 67% |

All 200 games completed, with zero incomplete games in the successful runs.
Gumbel used 24 concurrent games and took 16m11s including startup. Halving initially
exhausted GPU memory at concurrency six before any games finished. Its preserved
failure log identifies suffix-KV concatenation as the failing allocation. Retrying
at concurrency four completed in 22m32s, with the search budget and checkpoint
unchanged. Halving runtime is therefore not directly comparable with the old
six-game run. Total wall time including the failed attempt and retry was 40m36s.

An independent game bootstrap, stratified by model color, gives these exploratory
95% intervals for actor147 minus actor204: Gumbel −27 to −4 percentage points;
halving −16.5 to +4.5 points. These assume independent games and exclude run-level
variation. Gumbel is a concerning negative result; halving does not establish a
regression at this sample size. Neither establishes an improvement.

Historical ckpt34 results were 50.5% for Gumbel-512 from the initial board and
roughly 65–66% for halving. These are historical references, not fresh matched
baseline reruns. Overnight held-out screens against ckpt34 also failed to establish
a gain (47.5%, 51.5%, 45% across three 100-game screens).

Actor147 has 3,076 optimizer steps and 2,091,655 training exposures. Actor204 had
5,196 steps and 3,090,338 exposures. The new night exceeded the old night's added
steps, but not the old model's cumulative training. Consequently this comparison
does not isolate the causal effect of the diversity change.

Raw results, frozen commands/source, retry history, and bootstrap script/results:
`artifacts/eval/actor147_streaming_sf2400_searches_2026-09-18/`.

## Matched search timings

The existing `scripts/bench_self_play.py --component search` harness evaluated
actor147 at concurrency 24, top-m 16, depth limit 32, and the production numerical
settings. There were 96 fixed starts: 24 initial boards with distinct search RNG
identities, plus 24 positions from each streamed human bucket (1–30, 31–70, 71–120).
The human starts came from published block two without advancing the training
cursor. Each budget used one warm-up and three measured repeats. All trials
completed all positions and simulations without errors; results within each
budget were identical across repeats.

| Simulations | Median time / 96 searches | Range | Searches/s | Relative time |
|---|---:|---:|---:|---:|
| 128 | 2.37s | 2.27–2.62s | 40.6 | 1.00× |
| 256 | 4.79s | 4.55–5.43s | 20.0 | 2.03× |
| 512 | 10.86s | 9.84–11.35s | 8.8 | 4.59× |

These are fixed-start search throughput measurements, not per-game move latency
or end-to-end training rates. Game evolution, replay I/O, training and screens are
excluded. A higher-budget collector needs its own whole-game validation before
adopting a nightly throughput forecast.

Relative to 128, selected moves changed on 7/96 positions at 256 and 14/96 at 512.
Mean policy total-variation distance was 0.135 and 0.180 respectively. Changed
targets are not proof of better targets. The depth limit also became active:
zero cutoffs at 128, 38 at 256 across one position, and 426 at 512 across 20
positions. This is another reason not to assume budget alone determines quality.

Commands, seed manifest, raw trials, metadata and summary:
`artifacts/benchmarks/actor147_search_budgets_2026-09-18/`.

## Evidence about the data and learning

The run completed 11,199 games with launch buckets 2,800 / 2,800 / 2,799 / 2,800,
and generated 1,024,191 positions. Input block waits totaled about 0.065 seconds.

The final active replay window contains 548 games and 49,852 positions. Its games
are nearly evenly split by starting bucket (140 / 137 / 136 / 135), while position
contributions are 17,635 / 14,196 / 10,922 / 7,099. Longer continuations naturally
produce more supervised positions. It includes 3,777 positions before ply 20,
12,479 at plies 20–59, 21,899 at 60–119, and 11,697 at 120+. This is a snapshot of
the final replay window, not the whole night's exposure distribution.

The first versus final 100 training batches show value loss decreasing from
5.24 to 0.73 and predicted draw probability rising from nearly zero to 20.3%
(observed final draw fraction 21.8%). The policy-target KL averages were 0.646 and
0.620. These are on changing training data; they do not establish held-out
calibration, policy improvement, or chess strength. In particular, matching the
overall draw rate does not establish correct position-level draw predictions.

## Recommended next experiment

Test a 256-simulation branch against a continued 128-simulation control, starting
both from the same actor147 recovery state. Preserve the optimizer, input-cursor
snapshot, starting mixture, learning rate, replay size and reuse settings in
isolated run directories. Compare over equal GPU hours, and report both strength
and fresh-position/exposure counts. This tests the compute tradeoff directly.

Continuing 128 alone for another night is a reasonable lower-cost control: one
night of the broader distribution is not enough to conclude it has plateaued,
and cumulative training remains below actor204. However, repeated nights should
be governed by fixed evaluation milestones rather than loss reduction alone.

256 is the first budget to test because it costs about twice as much search time;
512 costs about 4.6 times as much in this probe. Do not infer a corresponding
strength gain. Resetting to ckpt34 should be a separate experiment: current results
do not establish a regression from ckpt34 that warrants discarding actor147's
learning. A confirmed regression would change that choice.

If uniformly deeper search proves too expensive, mixed search budgets are a
research-backed later option. [KataGo's paper](https://arxiv.org/html/1902.10565v5#S3.SS1)
uses occasional full searches among faster moves and records only the full-search
turns for training. Applying that approach here would require explicit target
masking and exposure accounting; it is not equivalent to mixing cheap and expensive
targets indiscriminately. [Gumbel AlphaZero](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)
was designed to improve learning with limited simulations, so a fixed minimum
such as 512 is not established by that literature for this model and hardware.

As of this assessment, the existing timer still resumes the original
128-simulation streaming run at 22:00 Toronto and stops by 08:00. No larger-budget
training branch or restart has been launched.

## Authorized schedule update

After reviewing the assessment, the user selected 256 simulations for tonight's
resume. The existing timer now points to
`artifacts/self_play/actor147-streaming-s256-2026-09-18/`, starting September 18 at
22:00 Toronto and stopping by September 19 at 08:00. The original 128-simulation
run is preserved. Only the search simulation budget changes in the new branch;
24 concurrent games and the other training/search settings remain the same.

The fork retains actor147, 3,076 optimizer steps, 2,091,655 exposures, optimizer
and scheduler tensors, all RNG/sampler state, replay, and streaming cursor at
sequence 11,199. Old replay and the partial collection retain their 128-simulation
targets; newly collected targets use 256. Checkpoint configuration and progress
metadata were explicitly migrated to the new run and recorded in
`operation/fork-manifest.json`. All other checkpoint fields were checked for exact
equality, and copied replay/stream files were verified by SHA256.

The normal CUDA runtime and `Stage2Trainer.resume` successfully restored the new
checkpoint without advancing training or the data cursor. The nightly launcher
was verified to choose `--resume` with deadline `2026-09-19T08:00:00-04:00`.
`operation/resume-verification.json` records these checks. The installed service
also requires `state.json` to exist before starting, preventing an accidental
fresh initialization if the prepared continuation state is missing. Service
validation and timer inspection passed; training is scheduled, not running yet.

## Superseding request: fresh ckpt34, scale 1.0, immediate start

The user subsequently selected a fresh run from ckpt34 with 256 simulations and
`value_scale=1.0`, then requested an immediate start rather than waiting for the
nightly timer. The service now runs
`artifacts/self_play/ckpt34-streaming-s256-q1-2026-09-18/`. It started September 18
at 12:45:10 Toronto, with deadline September 19 at 08:00. Root candidates remain
16, maximum depth 32, concurrency 24, and the four-way streaming mixture and
learning settings are unchanged. The prepared actor147 continuation is preserved
but is no longer selected by the service. The active service will not be started
a second time by its 22:00 timer while it is already running.

`operation/initialization-verification.json` confirms exact initial model equality
with ckpt34 after removing `_orig_mod.` prefixes, an empty optimizer state, zero
steps/exposures, and empty replay/sampler queue. The original training stream
starts afresh for this experiment; its cursor is independent of previous runs.

The first collection completed 83 games and 6,660 usable positions in 511.68s:
46,857 positions/hour, median move latency 1.556s, p95 1.978s. All games completed:
76 checkmates and seven draws. There were 1,191 depth cutoffs among 1,704,960
simulations. The first training phase completed 22 updates / 13,320 exposures in
18.46s of measured update time, publishing actor1. Including input/model startup
and the first training phase, the initial rate was 40,124 positions/hour. These
are first-phase measurements, not an overnight average or a matched throughput
comparison with the old run. Raw evidence is in `operation/first-phase-report.json`.

A preceding isolated actor147 probe compared scales 0.1 and 1.0 at a fixed 128
simulations on the same 96 starts. Median search times were 2.706s and 2.852s;
selected moves differed on 25/96 positions. Mean target entropy fell from 0.555
to 0.057 nats, with over-95%-probability targets on 28 versus 87 positions. This
shows stronger concentration, not better targets or playing strength. Probe
artifacts: `artifacts/benchmarks/actor147_qscale_2026-09-18/`.

The user raised the paper's Figure 9 as motivation for inexpensive self-play and
larger evaluation budgets, with 128 simulations as a fallback if 256 is too slow.
No budget reduction has been applied as of the first-phase report. The Figure 9
comparison is at equal frames, not equal elapsed time; its 9x9 Go results do not
establish the optimal budget for this chess model.

## Current run: fresh ckpt34, 128 simulations, scale 1.0

The user selected the cheaper 128-simulation experiment after discussing Figure 9.
The 256 run was stopped, retaining actor1 and its phase-one recovery checkpoint
(22 updates). Systemd stopped the inhibitor wrapper and killed its remaining
processes, so the in-progress second collection did not complete a graceful drain;
the existing published checkpoint/replay/journal remain preserved. For future
graceful manual stops, signal the training child identified in session `launch.json`
and wait for it to save and exit before stopping the inhibitor service.

The replacement service started September 18 at 13:00:30 Toronto, using
`artifacts/self_play/ckpt34-streaming-s128-q1-2026-09-18/`. It initializes from
ckpt34 with an empty optimizer/replay and independent fresh streaming cursor,
128 simulations, scale 1.0, 16 root candidates, depth 32 and 24 concurrent games.
It retains the September 19 08:00 deadline. All other learning settings remain
unchanged. This supersedes the 256 run and the earlier actor147 continuation.

The hyperparameter review identified these priorities for separate experiments:

- Exploration: self-play currently plays the Gumbel search-selected action. The
  paper additionally describes early-game visit-count sampling; that is not
  implemented as an acting-temperature schedule here. Gumbel noise already gives
  stochastic exploration, so its absence is not equivalent to no exploration.
- Target sharpness and update size: scale 1.0 made the probe's targets much sharper.
  Keep LR 1e-5 and reuse 2 initially; monitor policy-target KL, target/model entropy,
  gradient clipping and held-out strength before increasing either.
- Value/policy loss balance: value weight stays 1.0. Earlier diagnostic gradient
  probes showed value gradients could dominate, but the prior late-stage weight-0.1
  fork did not repair strength. No optimal replacement weight is established.
- Replay: the 50,000-position window trades freshness against retention. Its size
  and the replay reuse ratio should be tuned separately from search changes.
- Search: c_visit 50 and root candidates 16 remain reasonable starting settings.
  The current production workspace explicitly rejects depths above 32; the earlier
  suggestion to compare depth 48 requires an implementation extension first.
- Evaluation: hold search budget, scale, noise and opponent protocol fixed across
  checkpoints when measuring training gains. Evaluation-budget tuning is separate.

These are tuning candidates, not additional changes to the active run.

User agreed to retain LR 1e-5, scale 1.0 and 128 simulations for at least
three hours before reviewing learning progress (approximately September 18
16:00 Toronto, with evaluation at the next eligible phase boundary). Review
smoothed policy-target KL, policy/value losses, target entropy, gradient norms
and the scheduled strength screen. Changing replay and target distributions
mean flat raw losses alone do not establish an inadequate learning rate.
If fitting remains stalled while training is numerically stable, consider a
controlled LR 3e-5 comparison from a saved state. Instability is a reason to
diagnose the run, not automatically increase LR. No automatic LR change is
enabled by this review plan.

### Fresh LR 1e-4 experiment (supersedes the LR 1e-5 observation plan)

The user subsequently authorized a fresh ckpt34 start at LR 1e-4 and scale 1.0.
The LR 1e-5 training child received SIGTERM directly and drained successfully:
service exit 0 at September 18 13:32:37 Toronto, no pending streaming launches.
Its actor6, 125 updates, 81,442 exposures and final collected replay remain in
the original directory; the last collection is saved at the train phase for resume.

The replacement `ckpt34-streaming-s128-q1-lr1e4-2026-09-18` started at
13:32:54 Toronto with fresh weights from ckpt34, fresh optimizer/replay and its
own streaming cursor. LR is constant 1e-4; search remains 128/16/32, scale 1.0,
c_visit 50, and all other learning/collection settings are retained. The existing
nightly service now targets this directory. Deadline remains September 19 08:00.
Three-hour screens remain enabled, so the first is due around 16:33 at an eligible
phase boundary. This supersedes the earlier 16:00 review time.

MiniZero source comparison (reviewed September 18):
https://github.com/rlglab/minizero/blob/main/minizero/config/configuration.cpp
defaults to SGD LR 0.02, momentum 0.9, and documents typical Adam/AdamW LR
0.001. Its Gumbel constants are sample size 16, visit constant 50, scale 1.
The learner supports SGD/Adam/AdamW and StepLR with step_size 1,000,000 and
gamma 0.1. These are framework settings, not validated rates for our pretrained
StableAdamW chess model. No optimizer or schedule changes were adopted.

### Continue ckpt34 optimizer and schedule (latest user decision)

The user superseded the fixed-LR experiment with continuation of ckpt34's saved
LR and schedule. Its actual saved LR is 0.0004151649149149149, not 0.0005.
The linear OneCycleLR schedule is at update 340,000 of 1,000,000, decaying toward
0.00025. The new run is `ckpt34-streaming-s128-q1-resumeopt-2026-09-18`.
Search remains 128 simulations, scale 1.0, top-m 16 and depth 32. Replay and
stream cursor start fresh; weights, StableAdamW moments and schedule continue
from ckpt34. The optimizer's saved Triton backend settings are also retained.
The brief LR 1e-4 run stopped gracefully after 35 games / 2,416 positions and
zero optimizer updates, preserving its collected replay.

Added explicit `--initialize-optimizer` support to the runner and nightly
launcher, with model/group/LR checks. Checkpoints record scheduler type so
subsequent self-play resume restores OneCycleLR rather than LambdaLR. Existing
constant-LR checkpoints still resume using LambdaLR. Metrics now include the LR
used for each update. Self-play step/exposure counters start at zero independently
of the inherited optimizer/scheduler clock.

Regression tests check exact subsequent updates against uninterrupted training,
then exact save/resume continuation. A disposable GPU test with the actual ckpt34
optimizer also passed: finite weights after an update, next LR
0.00041516466466466465 and scheduler step 340001. Evidence and frozen inputs are
in the new run's `operation/` directory. The nightly service now targets this
run and retains the September 19 08:00 Toronto deadline and three-hour screens.

### Return to constant LR 1e-4 (latest user decision)

The inherited-optimizer run's first screen (iteration 43) scored 22% against
ckpt34 over 50 color-swapped pairs, paired interval 15.5–29%. Its observe-only
mode recorded the `rollback_stop` recommendation but continued. The user then
requested stopping this run and starting fresh ckpt34 weights with constant
LR 1e-4. The new directory is
`ckpt34-streaming-s128-q1-constant1e4-2026-09-18`, with fresh optimizer, replay and
stream cursor. Search remains 128 simulations, scale 1.0, 16 candidates, depth32.
The initial LR 1e-4 directory from earlier today is preserved independently.
The old optimizer/scheduler import option remains available for reproducing and
resuming archived runs; this new run uses the existing constant-LR path only.

Root candidate count discussion: the paper's Algorithm 2 samples m candidates
without replacement using Gumbel + logits across all actions. Section 7.1 reports
Go ablations at m=4,8,16,32 and m=min(n,16) for other Go experiments; it does not
separately specify the chess m. Thus 16 is an established reference setting,
not a demonstrated chess optimum. At a fixed 128 simulations, 12 trades breadth
for more effort per candidate; 24 trades effort per candidate for breadth.
The user asked about these alternatives but did not request changing m.

The user then increased the requested simulation budget to 200 while retaining
constant LR 1e-4, scale 1.0 and m=16. The final replacement directory is
`ckpt34-streaming-s200-q1-constant1e4-2026-09-18`. It initializes fresh weights
from ckpt34 with empty optimizer/replay and an independent streaming cursor;
it does not continue the regressed actor or import the supervised optimizer.
The September 19 08:00 deadline and three-hour evaluation cadence remain.
Screens use the configured 200 simulations for both candidate and baseline;
comparisons across runs therefore also differ in evaluation search budget.

### Final overnight choice: 64 simulations, constant LR 1e-4

The user subsequently chose 64 simulations to collect more self-play positions
overnight, retaining scale 1.0, c_visit 50, 16 candidates and constant LR 1e-4.
The 200-simulation run stopped gracefully at iteration 0/train after eight updates
and 5,190 exposures; its checkpoint/replay remain available separately.
`ckpt34-streaming-s64-q1-constant1e4-2026-09-18` starts fresh from ckpt34 with
empty optimizer/replay and its own stream state. It supersedes all earlier run
targets for the nightly service and current TensorBoard exporter. Deadline remains
September 19 08:00 Toronto, with observe-only screens every three hours using
64 simulations for both actors. No claim is established that pretraining removes
the need for 50M positions or that longer training alone will repair regression.
