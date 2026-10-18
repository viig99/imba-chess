# Search-method study

The user paused all ordinary self-play training on 2026-10-18 to make CPU and
GPU resources available for this study. Do not restart those runs until the
study is complete or the user requests a restart. The user coordinated the
pause with the separate Claude session. Host inspection confirmed no training
or watchdog process remained, GPU utilization was 1%, and CUDA was available.

The study compares the current Gumbel baseline with Lc0-style PUCT, parent
variance-scaled PUCT, and action variance-aware PUCT-V. Use the existing native
chess implementation, decoder, scheduler, collector processes, and replay
targets. Keep policy selection, exploration, and target construction explicit.

Build a versioned Stockfish reference set with complete histories, diverse
source games, per-game tuning/test splits, and every legal move scored. Cache
engine identity, settings, raw centipawn/mate scores, WDL, PVs, and work counts.
Pilot analysis budgets before committing to the full labelling cost. Treat
Stockfish WDL as engine-reference estimates, not observed game outcomes.

Report separately: (1) frozen-network move/target/value quality, (2) matched
collection throughput and actual inference rows, and (3) learning after matched
training exposure and compute. A better frozen-network search does not establish
better learning. Keep the locked test set out of tuning and oracle labels out of
training. Preserve the initial 600-position study for historical comparison.

## Reference set

`artifacts/eval/golden-v1-2026-10-18/positions.json` freezes 5,000 positions
from 1,000 complete games: 750 held-out human games, 200 prior self-play games,
and 50 games against Stockfish. Five well-spaced, deduplicated roots per game
retain all preceding moves for the history-dependent model. The full source
games are stored alongside the roots. A deterministic split by source game gives
890 tuning positions and 4,110 test positions. The phase mix is 1,227 opening,
3,091 middlegame, and 682 endgame positions.

The reference builder uses ten independent single-threaded Stockfish processes,
full strength, root analysis at 1,000,000 nodes, and a separate forced search of
every legal move at 200,000 nodes. It records the last completed exact score
iteration rather than accepting an unfinished aspiration-window bound. Engine
binary hash, settings, position digest, and individual scores are versioned.
The append-only writer is locked, fsynced, and resumable; concurrent readers do
not modify its output. Labels live in `reference-200k/labels.jsonl`.

The full reference completed successfully in 4,915 seconds (about 82 minutes).
Validation found all 5,000 roots and all 156,227 legal moves, with 3,789
informative roots and 995 exactly flat roots. The final labels SHA-256 is
`9386d7b21772c0384123b09f07d4c9e54a8f6b2a49ed91b2f15c81094a74b358`.

A paired 48-position pilot at half versus full budget had mean absolute
per-move expected-score disagreement 0.00694. The lower-budget best move lost
0.00358 expected score on average under the higher budget, with a maximum of
0.1155. These finite searches are a strong reusable reference, with uncertainty
that matters when differences are small.

## Search and target implementations

All methods share the existing Rust node statistics, move generation,
selection/backup interface, Python scheduler, compiled GPU leaf decoder,
workspace, root-history handling, multi-process collection, and replay schema.
PUCT adds a native selection rule. Variance variants add online second moments
only when requested. Gumbel's statistics and selection behavior are preserved.
The common search driver still orchestrates traversal in Python.

Lc0-style PUCT uses a visit-dependent exploration coefficient and parent-based
first-play urgency; it is not a complete Lc0 engine port. Parent variance scaling
normalizes empirical return variance against a regularizing prior. PUCT-V uses
an action-variance exploration term plus a logarithmic bias term. These are
independent search ablations, without assumptions about their chess strength.

PUCT + RPO uses the same PUCT traversal and most-visited played move. It changes
only the final root learning target: maximize expected search Q minus
`lambda * KL(network prior || target policy)`. A scalar bisection normalizes
the resulting distribution. This is a root-target ablation, not a complete RPO
or RMCTS search implementation. It should add little runtime compared with 256
simulations, but its production cost is measured rather than assumed.

## Evaluation and selection

The frozen network is always the raw `actor-000550.pt` from
`plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run`, with BF16 inference. A position's
reference regret is `max_a SF_score(a) - SF_score(chosen move)`. Target regret is
the policy-weighted average of those action regrets. Lower is better. Also
report near-best probability mass, blunder rate, entropy, visited actions,
policy change from the prior, and value error against reference WDL. KL of
target versus prior measures change, not quality; Q values are not probability
distributions to compare using KL.

After exploratory 160-root runs, use the first 500 predetermined tuning roots
at 256 simulations for the expanded hyperparameter and root-target checks.
Each method gets exactly the same roots, complete histories, and fixed GPU
root batch grouping. Partial reference coverage never substitutes later roots
for requested roots. Select settings using tuning data, then freeze them before
learning forks and test evaluation. Report the full 5,000 roots with the tuning
and test splits separated; the 4,110 test roots are the independent comparison.
Confidence intervals use paired bootstrap resampling of source games, because
five positions from one game are correlated. Tuning intervals are exploratory
and do not account for selecting the best configuration from a grid.

Initial 160-root results, before expanded tuning:

| Method | Played-move regret | Target regret |
|---|---:|---:|
| Gumbel | 0.06340 | 0.05978 |
| PUCT, c=1 | 0.05329 | 0.07147 |
| Parent variance PUCT, c=1.75 | 0.04321 | 0.06813 |
| PUCT-V, bias=3 | 0.05446 | 0.07176 |
| PUCT c=1 + RPO target, regularization=0.25 | 0.05329 | 0.05451 |

The promising move selection and weaker visit targets are why root-target
construction needs a separate ablation. These results do not demonstrate better
neural learning.

The expanded 500-root check spans 177 source games (365 human, 119 self-play,
16 Stockfish-match roots). The grid and target ablation took 144 and 141 seconds
respectively, including model loads and warm-ups:

| Frozen finalist | Played-move regret | Target regret |
|---|---:|---:|
| Gumbel | 0.04837 | 0.04670 |
| PUCT, c=1 | 0.04134 | 0.05779 |
| Parent variance PUCT, c=1.75 | 0.03874 | 0.05535 |
| PUCT-V, bias=3 | 0.03960 | 0.05883 |
| PUCT c=1 + RPO target, regularization=0.25 | 0.04134 | 0.04751 |

The apparent RPO target advantage on 160 roots did not persist on 500 roots.
Its target-regret difference from Gumbel is +0.00081, with exploratory paired
95% interval [-0.00628, +0.00750]. Its played-move difference is -0.00703,
interval [-0.01565, +0.00197]. The candidate remains interesting for move quality
and throughput, but target quality is unresolved and better learning is unproven.

`frozen-selection.json` records the selection before any test evaluation.
`config/search_study_finalists.toml` locks all five arms. The primary candidate
is PUCT + RPO; the other search rules remain fixed comparison ablations.
The all-position report explicitly includes separate tune/test metrics and
paired intervals. An additional finalist check with two training-noise seeds
describes target behavior under actual collection exploration without changing
the frozen settings.

Before the held-out method comparison, the user explicitly requested the
variance-scaled PUCT + RPO combination. Revision 2 of `frozen-selection.json`
adds a sixth frozen arm with regularization 0.015625, chosen from the cached
500-root tuning ablation (target regret 0.04372, played-move regret 0.03874).
The initial selection is preserved as `frozen-selection-initial.json`. The
additional combination participates in the noise check, full-set comparison,
production throughput benchmark, and learning forks. The original five settings
remain unchanged. There are five learning forks: Gumbel, PUCT, parent variance
PUCT, PUCT + RPO, and parent variance PUCT + RPO.

The fresh six-arm check on 500 tuning roots is complete, with evaluation noise
disabled plus two training-exploration seeds. Mean target regrets over those two
training seeds are:

| Method | Mean training-noise target regret |
|---|---:|
| Gumbel | 0.04599 |
| PUCT | 0.06524 |
| Parent variance PUCT | 0.06021 |
| PUCT-V | 0.06594 |
| PUCT + RPO | 0.04860 |
| Parent variance PUCT + RPO | 0.04581 |

The new combination roughly matches Gumbel and repairs much of the visit-target
gap. The earlier cached 6% target advantage is not a robust finding. All six
methods had bit-identical root log priors within this fresh paired run. Its
three-variant root batching differs from the earlier one-variant grid; BF16
numerics and discrete tree decisions can shift across those batching layouts.
Compare methods within one matched run, and rely on the held-out and actual
learning tests rather than differences between successive tuning reports.

## Completed held-out search comparison

The locked six-arm comparison completed in 323 seconds. The independent test
split contains 4,110 roots from 822 source games; the 890 tuning roots are
reported separately in `locked-search-all-s256/summary.json`.

| Method | Test played-move regret | Test target regret |
|---|---:|---:|
| Gumbel | 0.04328 | 0.04512 |
| PUCT | 0.04060 | 0.05759 |
| Parent variance PUCT | 0.03974 | 0.05525 |
| PUCT-V | 0.03989 | 0.05903 |
| PUCT + RPO | 0.04060 | 0.04921 |
| Parent variance PUCT + RPO | 0.03974 | 0.04372 |

The variance + RPO combination has 8.2% lower mean played-move regret and 3.1%
lower mean target regret than Gumbel. Its paired move-regret difference is
-0.00354, interval [-0.00662, -0.00080]. The target-regret difference is -0.00140,
interval [-0.00415, +0.00116], so target superiority is unresolved. These are
source-game-clustered intervals and the multiple reported comparisons remain
exploratory. The combination shares variance PUCT's played moves by design.
Its search-WDL expected-score MAE is 0.13556 versus Gumbel's 0.16519, a frozen
search estimate comparison rather than evidence of learned value improvement.

PUCT + RPO at regularization 0.25 has worse held-out target regret than Gumbel
(paired difference +0.00409, interval [+0.00148, +0.00651]). This supports keeping
the variance + sharper-RPO combination in the actual learning screen, without
claiming the frozen target proxy proves it will learn better.

## Throughput and learning

Production collection benchmarks use five workers, 32 games per worker, the same
checkpoint and starts, a 30-second warm-up, then two measured phases targeting
20,000 completed positions each, with carry-over enabled. Actual GPU inference
rows exclude cached-root accounting in search results. Completed usable and
policy-eligible positions are reported separately. Workspace counters report
payload bytes transferred/copied; they do not measure hardware memory bandwidth
utilization. All current benchmarks share the load from the ten reference
workers. Each is a short measurement, not a repeated performance distribution.

The first measurements were 50,847 actual neural rows/sec and 777,989 usable
positions/hour for Gumbel, versus 55,451 rows/sec and 827,599 positions/hour for
PUCT c=1. These correspond to approximately 9.1% and 6.4% improvements.
Parent variance PUCT measured 54,460 rows/sec and 799,484 usable positions/hour.
PUCT + RPO measured 54,551 rows/sec and 807,821 usable positions/hour: 7.3% more
neural rows/sec than Gumbel and 1.6% fewer than plain PUCT. Its completed
policy-eligible throughput was 602,142 positions/hour versus Gumbel's 545,519.
PUCT-V measured 54,923 rows/sec and 839,499 usable positions/hour, with 615,944
policy-eligible positions/hour. The first five benchmark arms ran while the
reference builder was active (PUCT-V overlaps its final work). The added parent
variance PUCT + RPO benchmark runs after reference completion; treat small
performance differences across those CPU-load conditions as preliminary.
The initial added-combination run measured 81,444 actual rows/sec and 1,193,980
usable positions/hour after Stockfish had released the CPUs. This is not a
matched speedup against the earlier Gumbel run. A separate follow-up queue waits
until learning and raw evaluation finish, then benchmarks Gumbel and the
combination back-to-back under the same idle-CPU conditions. Its status is
`idle-benchmark-queue.json` and outputs go to `throughput-idle/`.
See `artifacts/eval/search-study-2026-10-18/throughput/*/result.json` for measured
phases, worker stages, GPU samples, and byte counters.

Isolated learning forks start from identical model, optimizer, scheduler, and
EMA state. They use the same fixed cohorts of starting positions and the normal
training recipe, with three milestones of 320 games and nominally 50,000 training
exposures each. Actual exposures can exceed the nominal budget by the final
optimizer step and are recorded. Whole cohorts finish under one actor; carry-over
is disabled for these forks. Human source identities used in the golden set,
including parent sources of golden self-play games, are excluded from starts.
Stockfish scores are never training labels. Each arm has its own replay and
checkpoints; ordinary training artifacts remain untouched.

`scripts/evaluate_search_learning.py` compares raw network policies and values,
without MCTS, at each milestone against the initial network and the Gumbel fork.
Report gains against both actual exposures and collection-plus-training seconds.
A single short fork per method can screen candidates; it cannot establish
long-run learning superiority or replace repeated seeds and playing-strength
matches.

Early learning progress: Gumbel completed 960 games and 152,777 supervised
exposures in 67 optimizer steps; PUCT completed 960 games and 154,193 exposures
in 67 steps. Collection plus training took 282.8 and 308.5 seconds respectively,
including each arm's training compilation. PUCT produced 57,708 completed
full-search positions versus Gumbel's 49,369, so a fixed game count does not imply
equal collection work. Both completed without numerical failures. These are
progress and workload observations; trained-weight held-out evaluation has not
yet run, and own-target training losses must not be used to rank the methods.

## Completed trained-weight screen

All five learning forks and the raw held-out evaluator have now completed.
The evaluator took 103 seconds, covering the initial network and all three
milestones of every arm on the same 4,110 held-out roots. No MCTS is used here.
Final training exposures range from 152,777 to 154,676 (66--67 optimizer steps).

| Network | Raw argmax move regret | Raw policy expected regret | Raw value MAE |
|---|---:|---:|---:|
| Initial actor-550 | 0.08257 | 0.11756 | 0.16602 |
| Gumbel fork | 0.07973 | 0.11095 | 0.16230 |
| Parent variance PUCT + RPO fork | 0.07961 | 0.11105 | 0.16502 |
| Parent variance PUCT fork | 0.07903 | 0.10715 | 0.16228 |
| PUCT + RPO fork | 0.08196 | 0.11345 | 0.15866 |
| PUCT fork | 0.07942 | 0.11236 | 0.16152 |

The strongest short-run raw-policy result is parent variance PUCT **without
RPO**, despite its weaker frozen visit-target expected regret. Its final raw
policy expected regret is 3.4% below Gumbel's; paired difference -0.00381,
source-game bootstrap interval [-0.00474, -0.00288]. At the preceding milestone
it already reached 0.10932 after 103,847 exposures and 225.6 collection/training
seconds, versus Gumbel's final 0.11095 after 152,777 exposures and 282.8 seconds.
Its raw argmax move advantage over Gumbel remains unresolved (difference -0.00070,
interval [-0.00398, +0.00288]). Its value MAE is practically equal to Gumbel's.

The variance + RPO fork roughly ties Gumbel on final raw policy expected regret:
difference +0.00009, interval [-0.00096, +0.00109]. Its raw move difference is
-0.00012, interval [-0.00371, +0.00354]. Its value MAE is higher in this screen.
Thus its better frozen search-target score did not produce a measurable
short-run raw-policy advantage over Gumbel.

Plain PUCT and PUCT + RPO have worse raw policy expected regret than Gumbel at
this milestone. PUCT + RPO has the lowest mean value MAE, illustrating a
policy/value tradeoff rather than an overall winner. These confidence intervals
resample held-out source games, not training seeds. One short training seed
cannot establish long-run learning or playing-strength superiority; none of
these results are Elo estimates. Raw policy expected regret measures the
learned network's probability distribution, not the MCTS teaching target.

## Second training seed and paired reporting

The user requested a repeat with another random seed. Seed 43 repeats all five
learning methods from the identical actor-550 model and optimizer state, with
the same three 320-game phases, nominal 50,000 exposures per phase, and 5 x 32
collector layout. Generated configuration copies change only `run.seed` from
42 to 43; assertions compare every other parsed field with the frozen originals.
The production runtime seeds Python and Torch, while source cohorts, search
exploration, game IDs, and replay shuffling use the new seed. All methods within
seed 43 share the same source-cohort selection and launch identities.

`scripts/repeat_search_learning.py --seed 43` runs Gumbel, variance PUCT + RPO,
variance PUCT, PUCT + RPO, then PUCT. It runs raw held-out evaluation after each
completed arm, reusing cached predictions for earlier arms. This prioritizes the
Gumbel-versus-variance-plus-RPO comparison. No additional post-training MCTS is
performed for this learning screen. Outputs live under `seed-43/`.

`scripts/report_learning_seed_intervals.py --second <seed-43-directory> --watch`
updates `paired-two-seed-intervals.json` as trained-weight predictions become
available. It averages paired candidate-minus-Gumbel effects over the two seeds,
then jointly resamples the 822 held-out source games, retaining all positions
and both seeds together. Its 95% intervals quantify held-out sampling uncertainty
conditional on these two trained runs. Individual seed effects are shown
separately; two seeds are insufficient for a reliable training-seed confidence
interval. The repeat also writes `two-seed-summary.json` after all arms finish.

The matched idle-CPU throughput repeat is complete: Gumbel achieved 79,357 actual
neural rows/sec, 1,236,817 usable positions/hour, and 861,357 policy-eligible
positions/hour. Variance PUCT + RPO achieved 80,045 rows/sec, 1,157,926 usable
positions/hour, and 886,129 policy-eligible positions/hour. Neural-row throughput
is effectively similar in this short comparison (+0.9%); completed usable
throughput is lower (-6.4%) while policy-eligible throughput is higher (+2.9%).
The earlier large apparent speedup under unequal Stockfish CPU load must not be
used as the method's performance advantage.

## Final two-seed result

Seed 43, all incremental held-out evaluations, the combined report, and the
paired interval reporter completed successfully. The policy learning advantage
of variance-scaled PUCT **without RPO** repeated: raw policy expected regret
0.10715 and 0.10752 versus Gumbel's 0.11095 and 0.11195. The two-seed means are
0.10733 versus 0.11145, a 3.7% reduction. The paired mean difference is -0.00412,
conditional held-out source-game 95% interval [-0.00485, -0.00338]. Mean training
exposures are virtually equal, and mean collection-plus-training time is 1.2%
higher for variance PUCT (303.07 versus 299.55 seconds).

Variance PUCT + RPO has mean policy regret 0.11182, roughly tied with Gumbel
(difference +0.00037, interval [-0.00035, +0.00115]). Its value MAE is higher in
both seeds: mean difference +0.00350, interval [+0.00142, +0.00542]. Plain PUCT
and PUCT + RPO have worse mean raw policy expected regret than Gumbel in both
seeds. No trained candidate has a clear paired argmax-move advantage over
Gumbel in this screen. The appropriate next learner to investigate is plain
variance-scaled PUCT, without claiming a proven playing-strength gain.

The concise final report is
`artifacts/eval/search-study-2026-10-18/TWO_SEED_REPORT.md`. Paired intervals are
in `seed-43/paired-two-seed-intervals.json`; their uncertainty scope is conditional
on these two trained models, not a reliable estimate over new training seeds.

## Commands and artifacts

The main entry points are `scripts/build_golden_set.py`,
`scripts/compare_search_methods.py`, `scripts/bench_collection.py`,
`scripts/train_search_study.py`, and `scripts/evaluate_search_learning.py`.
`scripts/run_search_study.py` queues GPU work sequentially. Its optional
`--tuning-reference` runs the 500-root checks first. Use `--skip-learning` while
reviewing tuning results; learning requires frozen configuration choices.
With `--evaluation-reference` and `--evaluation-arms`, the queue runs the locked
full-set comparison before learning, then raw held-out policy/value evaluation
after every learning arm completes. `--exploration-seeds 2` adds the finalist
noise check first.

An additional CPU-only diagnostic, `rpo-small-lambda-tune.json`, reapplies RPO
to cached tuning trees. It reproduces the live regularization=0.25 policies to
1e-12. Reducing regularization to 0.015625 gives target regrets 0.04458 for PUCT,
0.04372 for parent variance PUCT, and 0.04482 for PUCT-V. The cached variance
result motivates the sixth frozen arm described above; the other small-lambda
settings remain diagnostic. Sharpening a target can improve its expected SF
score without improving the resulting network; even a one-hot most-visited
target can benefit this metric. Treat these as learning hypotheses, not
demonstrated learning gains.

Outputs are under `artifacts/eval/search-study-2026-10-18`. Search result
manifests guard checkpoint, code, settings, reference identity, and exact root
selection. The 160-root runs are `tune-grid-s256` and `tune-targets-s256`; expanded
runs are `tune-500-grid-s256` and `tune-500-targets-s256`. Learning outputs are
under `learning/<method>`. The successful small end-to-end collect/train smoke
run is `learning-smoke-puct-rpo`.
