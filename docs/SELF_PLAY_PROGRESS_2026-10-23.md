# Self-play progress and debugging lessons — 2026-10-23

EMA-2000 is the strongest completed SF2600 aggregate recorded so far: **60.5%
over 100 games (47 wins, 27 draws, 26 losses)**. The previous direct test,
EMA-1085, scored 49% (34/30/36). This is a promising continuing candidate, with
supporting improvement in both raw policy quality and value accuracy. Strength
does not rise at every checkpoint, and neither test is a precise rating estimate.

The [committed measurement record](results/self-play-2026-10-23.json) retains
the original aggregate reports and manifests, resolved evaluation settings,
live training configuration, head-to-head history, golden-set report and
throughput window. Source artifact paths and report hashes identify the local
evidence; model weights, replay and game traces remain uncommitted artifacts.

## What was measured

| Checkpoint | Opponent | Games | Score |
|---|---|---:|---:|
| EMA-1085 | Strength-limited SF2600 | 100 | 49.0% |
| EMA-2000 | Strength-limited SF2600 | 100 | **60.5%** |
| EMA-1000 | Starting EMA-554 | 400, 200 opening pairs | 63.25% |
| EMA-1250 | Starting EMA-554 | 400, 200 opening pairs | 63.5% |
| EMA-1500 | Starting EMA-554 | 400, 200 opening pairs | 65.5% |
| EMA-1750 | Starting EMA-554 | 400, 200 opening pairs | 66.125% |
| EMA-2000 | Starting EMA-554 | 400, 200 opening pairs | **67.5%** |

Both SF matches used Gumbel 512 simulations, top-m 16, depth 32, raw Q at
value scale 0.5, FP32, TF32 off and zero exploration. Stockfish used
`UCI_LimitStrength=true`, `UCI_Elo=2600`, one thread, 64 MiB hash and
`Limit(time=5, nodes=40000)`. There were eight concurrent games, two 50-game
batches with seeds 1042/1043, no random opening plies and a 512-ply limit.
The EMA-2000 evaluation source was `b7ee94c`, runtime `shared-search-v2`.

At score p = 0.605, `2600 + 400 log10(p / (1-p))` gives **2674**.
An approximate 95% interval is **2616–2736**: encode game scores as 1/0.5/0,
estimate their sample variance, form `p ± 1.96 × SE` (52.34%–68.66%), then
transform the endpoints. This assumes independent games and measures sampling
uncertainty only. Opening dependence, opponent calibration and engine randomness
are not accounted for. This is performance relative to this specific SF benchmark,
not FIDE Elo or full-strength Stockfish at a fixed node budget.

The two fixed-network baselines are different: **53,250** is the original
supervised checkpoint's update number; **EMA-554** is the prior self-play run's
EMA used to initialize the current run. The current run restarts its iteration
counter. Screens against 53,250 and head-to-heads against EMA-554 are not
interchangeable, and neither score can be converted directly into an SF score.

## Throughput was the largest practical accelerator

The earlier reported pace was about 100k completed games/day. In the measured
two-hour window on October 21, the run advanced from iteration 1103 to 1155,
about **16.7k games/hour or 400k/day**, including alternating training and routine
side jobs. A 100k-game experiment therefore takes about six hours instead of a
day. Longer exclusive evaluations and downtime reduce realized daily totals.
This is a wall-clock observation, not a matched hardware/configuration ablation.

The useful engineering changes were reducing Python work and repeated allocation
in the hot search path, while preserving search semantics:

- `34ff836`: Rust tree ownership and the simulation loop, plus bulk decode-wave
  staging. The commit records identical CPU collection over six configurations
  (4,558 searches), GPU collection parity on actor-940 BF16 (1,101/1,101), and
  collection benchmarks increasing from about 71–72k to 81–84k neural evals/s.
- `499a10d`: one allocation per node's edge arrays, flat tree buffers, and an
  exact fixed-point fast path for completed-Q summation. Recorded search CPU
  cost fell from 6.3 to 3.8 microseconds/leaf with identical collection records.
- `b7ee94c`: capture KV stacking/scatter, legal-logit gathering and result packing
  inside decoder graphs, reducing eager submissions between decode waves.
- Five workers × 32 concurrent games under CUDA MPS, one CPU thread each,
  reusable inference workspaces, carry-over games and cheaper decided-game
  playouts keep useful work flowing through a 16 GB RTX 5070 Ti.

More games made it practical to test longer learning horizons and inspect more
checkpoints. Target construction, learning rate, batch size and EMA horizon also
changed. We cannot assign the strength gain to throughput alone or claim the
native optimizations increased learning per game. Retest parity whenever moving
arithmetic or ownership; a faster search with changed labels is another experiment.
See the [collection performance handoff](COLLECTION_PERFORMANCE_HANDOFF_2026-10-17.md).

## Is the policy actually helping?

Earlier debugging found that a lower policy-target KL did not imply better moves.
On 600 positions in the [October 11 plateau investigation](GUMBEL_PLATEAU_QUESTION_2026-10-11.md),
KL fell 0.860 → 0.697 while prior regret was essentially flat (0.081 → 0.083).
Search-target regret improved 0.027 → 0.020. Only about 8% of target/prior KL was
seed noise. That evidence favored investigating absorption/generalization rather
than assuming all target disagreement was noise or blindly increasing search.

The current fixed golden set tells a better story. EMA-2000 versus EMA-554,
4,110 positions from 822 source games, no search:

| Metric, lower is better | EMA-554 | EMA-2000 | Paired delta, 95% game-bootstrap interval |
|---|---:|---:|---|
| Expected prior move regret | 0.11623 | **0.10852** | −0.00770 [−0.00947, −0.00580] |
| Prior argmax move regret | 0.07922 | **0.06954** | −0.00968 [−0.01518, −0.00474] |
| Value expected-score MAE | 0.16422 | **0.15311** | −0.01111 [−0.01341, −0.00875] |

Both heads now improve against a fixed external reference. This supports policy
learning in move quality, not merely closer imitation of a moving teacher.
It does **not** isolate the policy head's contribution to match strength; that
requires a controlled policy/value ablation. Do not compare absolute regret
between this golden set and the earlier 600-position sample.

## Learning and target fixes that mattered

Gumbel's all-simulation WDL average includes visits spent on moves sequential
halving subsequently rejects. Use the WDL under the returned improved policy
(`value_search_source = "improved_wdl"`) for the search-value mixture and auxiliary
TD targets. Under training noise, the October 18 audit reduced label expected-score
MAE from about 0.166 to 0.138 and removed most of its negative bias. These label
metrics are distinct from the golden network metrics above. See [search studies](SEARCH_METHODS.md).

Frozen-replay experiments also exposed learning damage: at LR 2e-4, held-out-game
KL worsened from 0.77 to 1.07 and golden value MAE from 0.164 to 0.169 after six
passes. LR 5e-5 reduced that damage (about 0.79 KL, 0.158 MAE). More training on
the same data was not automatically useful. Increasing the improved-search WDL
mix to 0.5 preserved search move quality better than outcome-heavy labels, even
when the latter looked better in value calibration alone. The diagnostics are
retained locally under `artifacts/eval/learning-diagnostics-2026-10-19/` and their
summary is in `config/self_play.toml`.

The live candidate uses LR 5e-5, improved-WDL mix 0.5, reuse 2, a 500k-position
FIFO window, policy surprise weighting and auxiliary TD horizons 0.8/0.9/0.95.
Gradient accumulation increased from 4 to **16** around iteration 620; the
repository default is still 4. EMA decay remains 0.995 **per optimizer step**:
fewer, larger steps lengthen its horizon in games/iterations (roughly 20 rather
than five iterations). Carry weights, optimizer, EMA, replay and stream identity
when resuming; record every deliberate recipe migration. Collector configuration
uses BF16; the parent's run-start runtime description can report FP32 and must
not be substituted for worker precision. The SF evaluations explicitly use FP32.

## Debugging the next plateau or regression

1. **Verify the run and measurement identity.** Inspect state, checkpoint,
   config/CLI overrides, seed manifest, runtime revision and resolved precision.
   Compare completed matches under the same opponent, budget, exploration,
   openings and stopping rules. Preserve EMA and raw weights before pruning.
2. **Check useful wall-clock throughput.** Count completed, usable games and
   positions across collection, drain, training, restart and evaluation time.
   Neural evals/s and GPU utilization alone can hide long games, discarded
   continuations, compilation, startup, or CPU submission gaps. Profile workers,
   not just the coordinator; use GPU timing for asynchronous kernels.
3. **Separate teacher quality from absorption.** On a frozen position set,
   compare raw prior, improved policy and selected search move against the
   external reference. Measure regret, best-move agreement and good-move mass
   alongside KL/entropy. If the teacher improves but the held-out prior does not,
   test optimization and generalization before buying more simulations.
4. **Use game-held-out frozen replay.** Verify the model can overfit a small
   sample, then compare unseen games, not random positions from the same games.
   Match data/exposures when varying LR, accumulation, target temperature or loss
   weights. Report actual optimizer steps and the resulting EMA horizon.
5. **Audit value semantics and calibration separately.** Check side-to-move sign,
   terminal WDL, mixture source, draw bias and auxiliary horizons. Compare played
   move regret as well as MAE/Brier; calibration can improve while search ranking
   worsens. Min-max Q scaling can amplify near-ties: inspect surprise by Q range,
   not just its average. See [policy weighting](POLICY_SURPRISE_WEIGHTING.md).
6. **Check diversity and repetition.** Inspect start buckets, stream cursor,
   repeated lines and termination rates. A new run can replay its parent's
   starts unless stream history is explicitly carried. PUCT's deterministic
   most-visited self-play previously collapsed onto a repeated drawn line;
   exploration and move sampling are part of teacher quality.
7. **Confirm strength externally.** Fixed-baseline paired matches and golden
   metrics are useful frequent signals; direct SF matches are the external
   check. Review intervals and raw/EMA differences. A promising 100-game record
   warrants continued measurement, not automatic promotion or a precise Elo claim.

## Operational recovery and evaluation scheduling

The run supervisor captures checkpoints before pruning, screens every 155
iterations, checks the golden set every 25, and plays EMA-554 every 250.
The requested SF2600 test at 2000 was a **one-off**, not a recurring schedule.
Head-to-heads and SF matches require exclusive GPU use: a previous head-to-head
beside five collectors caused OOMs. Routine side jobs must also fit the memory
budget; a run's normal peak is not spare capacity guaranteed for an evaluator.

For an exclusive evaluation, set `RUN/STOP`, send SIGTERM to the training
coordinator and wait for a clean drain/checkpoint and watchdog exit. Capture EMA,
run the recoverable batched match, then clear only the evaluation-owned pause
and restart the watchdog with the exact run config. The live supervisor's
`PAUSED` marker distinguishes its automatic evaluation pauses from a user's stop.
Do not clear an unexplained STOP, halted state or pause automatically.

If training stops unexpectedly, inspect `state.json`, `watchdog.log`, the last
traceback and supervisor actions before restarting. The watchdog retries crashes
at most three times/hour; the supervisor has its own limited restart policy.
Those limits should expose persistent faults rather than hide them. Resume from
the saved state; never fabricate labels for incomplete games or discard replay
and stream progress merely to get the process running again.
