# Search methods: production usage and study results

All commands below run from the repository root. Gumbel remains the default.
The added methods use the same native chess transitions and node statistics,
compiled single-leaf CUDA decoder, KV workspace, root carry, batch scheduler,
collection workers, and replay schema. They need no checkpoint conversion.
Ordinary self-play has not been restarted as part of this integration.

**Current learning candidate: variance-scaled PUCT without RPO.** It improved
raw policy expected Stockfish regret in both short training seeds. RPO remains
available for ablations; its frozen-target advantage did not become a repeated
learning advantage. No method has established an Elo gain in this study.

## Supported choices and exact presets

| Public CLI name | Training config | Traversal | Policy target | Main preset |
|---|---|---|---|---|
| `gumbel` | `config/self_play.toml` | Existing Gumbel | Completed-Q improved policy | 256 simulations, value scale 0.5 |
| `puct` | `config/self_play_puct.toml` | Lc0-style PUCT | Visit distribution | cpuct 1.0 |
| `puct_variance` | `config/self_play_puct_variance.toml` | Parent variance-scaled PUCT | Visit distribution | cpuct 1.75 |
| `puct_v` | `config/self_play_puct_v.toml` | Action variance-aware PUCT-V | Visit distribution | c1 √2, c2 3.0 |
| `puct_rpo` | `config/self_play_puct_rpo.toml` | Same PUCT traversal | Root RPO | cpuct 1.0, RPO factor 0.25 |
| `puct_variance_rpo` | `config/self_play_puct_variance_rpo.toml` | Same variance PUCT traversal | Root RPO | cpuct 1.75, RPO factor 0.015625 |

PUCT presets use 256 simulations and depth 32, cpuct base 38,739, coefficient
factor 3.89, FPU reduction 0.33, variance prior 0.25 with weight 8, and training
Dirichlet alpha 0.3 / fraction 0.25. Full settings are in the files above.
The configurations otherwise copy the established collection and learning recipe;
workers and game concurrency remain execution options. Lc0-style describes the
selection rule and parent FPU, not a complete Lc0 engine implementation. Variance
scaling is an interpretable parent-return-variance adaptation, not a complete
KataGo implementation. PUCT-V follows arXiv:2512.21648 (Weichart): Eq. 22 in v1,
renumbered Eq. 17 in v3, using ln(N + 1) for the paper's log N. Its preset c2 = 3.0
is tuned on our [-1, 1] value scale, which equals the paper's c2 = 1.5 on [0, 1],
not the paper default. One deviation from Lc0 remains: FPU uses the mean of child
returns as parent Q, without the node's own network value.

**RPO changes the returned policy / teaching target. At fixed weights and search
settings it does not change tree visits or the most-visited move played during
evaluation.** It maximizes `Q · target − λ KL(prior || target)` by scalar
bisection, with completed Q for unvisited actions. Its λ is the configured factor
times the dynamic cpuct coefficient times `sqrt(simulations)/(simulations + legal_moves)`.
The current collector plays the returned move directly. At identical weights,
adding RPO therefore changes labels rather than moves; trajectories can diverge
after the changed targets update the weights.

## Train and resume

Start a separate run when changing search method. Reusing an existing output with
a different search config fails the resume identity check. For a checkpoint with
compatible optimizer state, preserve that state explicitly:

```bash
.venv/bin/python scripts/run_self_play.py \
  --config config/self_play_puct_variance.toml \
  --initialize artifacts/self_play/plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run/actor-000550.pt \
  --initialize-optimizer --collect-workers 5 --concurrent-games 160 \
  --output artifacts/self_play/variance-puct-new

# Later resume with the same config. Pass it through the watchdog as well.
scripts/self_play_watchdog.sh artifacts/self_play/variance-puct-new -- \
  --config config/self_play_puct_variance.toml \
  --collect-workers 5 --concurrent-games 160
```

Replace the config with `self_play_puct_variance_rpo.toml` for the RPO ablation,
or the other files in the table. Omit `--initialize-optimizer` only when a fresh
optimizer is intended. These are normal production runs, continuous by default;
use `--no-continuous` for the configured duration. The existing monitor seed
manifest and training corpus are required as described in the README.

The TOML `[search]` section uses canonical `algorithm = "puct_variance"` with
`target_policy = "rpo"` to enable RPO; the `_rpo` suffix is a public inference CLI
alias. Collection retains seeded root exploration. Standalone evaluations and
normal checkpoint screens explicitly disable exploration. Fast decided-game
playouts also work with these methods; `fast_top_m` only applies to Gumbel.
The new methods retain the existing maximum CUDA workspace depth of 32.

## Inference and matches

A trained checkpoint can be evaluated with any supported search regardless of
which search generated its training data. Keep weight effects and search effects
separate when comparing results.

```bash
# Recoverable SF2600 / 40k-node evaluation; shared BF16 inference and batching.
.venv/bin/python scripts/run_stockfish_eval.py \
  --checkpoint artifacts/self_play/variance-puct-new/actor-000100.pt \
  --config config/imba_chess_v4_aux3.toml \
  --out artifacts/eval/variance-puct-sf2600 -- \
  --model-move-policy puct_variance \
  --search-config config/self_play_puct_variance.toml \
  --simulations 512 --inference-dtype bfloat16

# Mixed-search match: each side can choose its own method and parameters.
.venv/bin/python scripts/match_two_checkpoints.py \
  --config config/imba_chess_v4_aux3.toml \
  --checkpoint-a artifacts/self_play/gumbel-new/actor-000100.pt \
  --checkpoint-b artifacts/self_play/variance-puct-new/actor-000100.pt \
  --model-move-policy-a gumbel --model-move-policy-b puct_variance \
  --search-config-b config/self_play_puct_variance.toml --simulations 256 \
  --inference-dtype-a bfloat16 --inference-dtype-b bfloat16 \
  --games 200 --concurrent-games 8 --device cuda \
  --output-json artifacts/eval/gumbel-vs-variance.json
```

`eval_vs_stockfish.py` accepts the same `--model-move-policy`, `--search-config`,
`--simulations`, and precision flags for a single run. Without `--search-config`,
PUCT methods use the presets above. With it, the `[search]` section is authoritative
and must agree with the selected traversal and target type; `--simulations`
overrides only the budget. The inference `--config` remains the model/repo config,
not the self-play config. Gumbel's legacy `--gumbel-simulations` remains supported;
use one simulation flag. Halving retains `--search-budget` in neural evaluations.
Mixed matches also support `--simulations-a/b` and `--search-config-a/b`.

Evaluation JSON saves the selected name, resolved parameters, budget/unit, and
precision. The recoverable Stockfish wrapper hashes a supplied search config and
rejects changed contents on resume. Use a fresh output for a changed method.
`eval_self_play.py --config <self-play TOML>` also supports these methods for
restartable color-swapped checkpoint pairs or its independent Stockfish protocol;
its `--best` opponent uses the same configured search, not a per-side override.
The older `calibrate_stockfish_nodes.py` CLI retains its Gumbel/halving choices.

For Python callers, select the canonical runtime backend and pass explicit config:

```python
from imba_chess.eval.inference_runtime import load_runtime
from imba_chess.eval.search_options import puct_config, search_backend

method = "puct_variance_rpo"
config = puct_config(method, simulations=256)
runtime, _ = load_runtime(repo_config=repo_config, checkpoint=checkpoint,
                          device="cuda", algorithm=search_backend(method))
# Submit the generator to the existing BatchScheduler / runtime executors.
generator = runtime.search(board=board, history=history, actor_id="actor",
                           game_id="game", config=config, noise=0.0)
```

Omit explicit noise only when training exploration is wanted. Search returns the
existing GumbelResult-compatible move, policy, visits, Q, WDL, and counters. RPO
changes `policy`; its runtime backend remains `puct` or `puct_variance`.

## Golden reference and metrics

The reusable golden set contains 5,000 positions from 1,000 complete games:
750 held-out human games, 200 older self-play games, and 50 Stockfish-match games.
Five spaced roots per game retain full preceding histories; full source games
are saved too. Tuning/test split is by source game: 890 / 4,110 positions.
The full mix is 1,227 opening, 3,091 middlegame, and 682 endgame roots.

Ten single-thread Stockfish 18 processes annotated at full strength: 1M nodes per
root plus independent 200k-node forced searches for every legal move. All 156,227
legal moves have labels. Completion took 4,915 seconds (~82 minutes). The data
includes 3,789 informative and 995 exactly flat roots; finite engine searches
have uncertainty and are not observed outcomes or an Elo ground truth.

Let `s(a)` be Stockfish's expected score from the player-to-move perspective,
`P(win) + 0.5 P(draw)`, and `r(a) = max_b s(b) − s(a)`.

- **Move regret:** `r(played move)`; lower is better.
- **Policy expected regret:** `sum_a policy(a) r(a)`; measures the entire teaching
  distribution in frozen search, or the learned network distribution in raw eval.
- **Best move:** a move tied for best under this finite reference, not exact PV match.
- **Blunder rate:** fraction with move regret at least 0.05.
- **Good mass:** policy probability within 0.02 of the reference best.
- **Value MAE/MSE:** absolute/squared error of network or search expected score
  against the independent root reference. This is not three-component WDL MAE.
  *All-simulation* search value is `search_wdl`, the mean leaf WDL over every
  simulation. Gumbel spends only ~25% of its simulations on the move it plays
  (PUCT ~67-71%), so its all-simulation value includes moves that sequential halving
  discards and is biased low (-0.078 expected score). *Played-move Q* is the played
  edge's mean value, `(1 + q) / 2`; use it to compare value estimates across methods.
- **KL(target || network prior):** size of a policy update, not oracle quality.
  Q-values are scores, not probability distributions; we do not take KL of Q-values.

The frozen actor is `plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run/actor-000550.pt`.
The initial raw network had policy regret 0.11756, argmax move regret 0.08257,
and expected-score value MAE 0.16602 in the raw-learning evaluation.
Arms were selected on the first 500 tuning positions, then frozen before held-out
comparison. Inference was BF16, 256 simulations, zero exploration, with identical
planned root grouping. The full test is 4,110 roots from 822 source games.

## Full frozen-search held-out results

Lower regret and value errors are better; higher best-move rate and good mass are better.

| Method | Move regret | Target regret | Best move % | Blunder % | Good mass % | All-sim value MAE | Played-Q value MAE | Played-Q value MSE | Target entropy | KL to prior | Visited actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gumbel | 0.04328 | 0.04512 | 65.20681 | 16.25304 | 77.31639 | 0.16519 | 0.13710 | 0.03643 | 0.52953 | 0.98897 | 14.90754 |
| PUCT | 0.04060 | 0.05759 | 66.00973 | 15.57178 | 74.58238 | 0.13684 | 0.13555 | 0.03567 | 0.93436 | 0.40479 | 6.07470 |
| Variance PUCT | 0.03974 | 0.05525 | 65.79075 | 15.40146 | 75.21413 | 0.13556 | 0.13427 | 0.03522 | 0.80020 | 0.49425 | 5.18905 |
| PUCT-V | 0.03989 | 0.05903 | 66.27737 | 15.42579 | 74.19603 | 0.13836 | 0.13694 | 0.03591 | 0.95675 | 0.36380 | 6.04501 |
| PUCT + RPO | 0.04060 | 0.04921 | 66.00973 | 15.57178 | 76.54295 | 0.13684 | 0.13555 | 0.03567 | 0.64588 | 0.89847 | 6.07470 |
| Variance PUCT + RPO | 0.03974 | 0.04372 | 65.79075 | 15.40146 | 77.51120 | 0.13556 | 0.13427 | 0.03522 | 0.23085 | 1.27709 | 5.18905 |

Variance PUCT reduces played-move regret by 8.2% versus Gumbel, paired source-game
95% interval for the difference [-0.00662, -0.00080]. Its visit target is worse
than Gumbel's. RPO retains the same played moves and reduces target regret to
0.04372; its target advantage over Gumbel is unresolved (difference -0.00140,
interval [-0.00415, +0.00116]). PUCT-V also improves played-move regret but has
worse visit targets; it was not advanced to a learning fork.

The all-simulation value column favours PUCT by construction (see the metric
definitions above). On the played-move Q, every method is within 0.003 MAE of
Gumbel's 0.13710: no method has a clear value-estimation advantage. Training labels
inherit this: `value_search_mix` and the auxiliary TD targets consume `search_wdl`,
so Gumbel runs train on a search value (0.165 MAE) no better than the raw network
(0.166), while PUCT runs get the improved one. Learning comparisons between the two
therefore differ in value-label quality as well as policy targets.

## Two-seed learning screen

The following completed screen tests self-play generation followed by training,
not fixed-dataset distillation. Each method starts from the identical frozen
checkpoint and optimizer/EMA/scheduler state, then generates its own continuations.
No Stockfish labels are used for training. Golden source identities are excluded
from training starts. The study uses uniform replay sampling and explicit exposure
budgets with carry-over disabled, so this is a controlled short screen rather
than an exact replay of the production fresh-first/carry-over recipe.


Both seeds (42 and 43) and all five learning methods completed successfully.
Each fork starts from the same actor-550 model and optimizer state, plays 960
games over three phases, and trains for approximately 150,000 supervised
position exposures (65--67 optimizer steps). Methods within a seed share source
cohorts and launch identities; their generated games can diverge. All milestone
evaluations use raw network predictions without MCTS on 4,110 held-out positions
from 822 source games. Stockfish scores are evaluation labels only.

### Main finding

**Variance-scaled PUCT without RPO produces the strongest short-run raw policy
in both training seeds.** Its mean policy expected regret is 3.7% below Gumbel's.
Variance-scaled PUCT + RPO does not improve raw policy learning over Gumbel in
this screen and has higher mean value error. Better frozen search targets did
not predict the best trained network.

### Raw policy expected Stockfish regret

Lower is better. This scores the learned policy distribution, not a search
teaching target or the network's chosen move alone.

| Method | Seed 42 | Seed 43 | Mean | Difference from Gumbel, paired 95% interval |
|---|---:|---:|---:|---:|
| Gumbel | 0.11095 | 0.11195 | 0.11145 | Baseline |
| Variance-scaled PUCT | 0.10715 | 0.10752 | **0.10733** | **-0.00412 [-0.00485, -0.00338]** |
| Variance-scaled PUCT + RPO | 0.11105 | 0.11260 | 0.11182 | +0.00037 [-0.00035, +0.00115] |
| PUCT | 0.11236 | 0.11349 | 0.11293 | +0.00147 [+0.00083, +0.00212] |
| PUCT + RPO | 0.11345 | 0.11456 | 0.11400 | +0.00255 [+0.00188, +0.00326] |

The individual policy-regret improvements for variance-scaled PUCT are -0.00381
and -0.00443 versus the corresponding Gumbel seed. Its mean value MAE difference
is +0.00100, interval [-0.00087, +0.00292]: no clear value advantage or penalty.

### Chosen moves, value, and cost

| Method | Mean argmax move regret | Mean value MAE | Mean collection + training seconds |
|---|---:|---:|---:|
| Gumbel | 0.08012 | 0.16165 | 299.55 |
| Variance-scaled PUCT | 0.07915 | 0.16265 | 303.07 |
| Variance-scaled PUCT + RPO | 0.08087 | 0.16515 | 291.70 |
| PUCT | 0.07950 | 0.16104 | 293.32 |
| PUCT + RPO | 0.08166 | 0.16236 | 293.80 |

No candidate's mean argmax move advantage over Gumbel is clear in these paired
intervals. Variance + RPO's value MAE is higher in both seeds: mean difference
+0.00350, interval [+0.00142, +0.00542]. PUCT + RPO's first-seed value advantage
did not repeat in the second seed. The main repeated gain is soft-policy quality,
not a demonstrated improvement in playing strength.

Mean training exposures are almost equal for Gumbel (154,369) and variance PUCT
(154,247). Variance PUCT uses about 1.2% more collection-plus-training time in
this screen. Timing includes training compilation and has only two samples.

### Training diagnostics at the final phase

Exposure-weighted means within phase three, then averaged across the two seeds.
These are on each method’s own self-play data and targets, not the golden set.

| Method | Total loss | Policy CE | Value CE | Target KL | WDL Brier | Target entropy | Network entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gumbel | 2.27719 | 1.44272 | 0.38813 | 0.78676 | 0.20914 | 0.65596 | 1.56390 |
| Variance PUCT | 2.38922 | 1.51781 | 0.45135 | 0.51677 | 0.24426 | 1.00104 | 1.58818 |
| Variance PUCT + RPO | 2.46011 | 1.50607 | 0.43556 | 1.24412 | 0.23530 | 0.26194 | 1.68573 |
| PUCT | 2.51958 | 1.67604 | 0.45384 | 0.46348 | 0.24351 | 1.21256 | 1.72179 |
| PUCT + RPO | 2.52212 | 1.56324 | 0.45023 | 0.86919 | 0.24572 | 0.69405 | 1.71106 |

Total loss includes weighted policy, value and auxiliary components. Policy CE
depends strongly on target entropy; a lower CE against a different target is not
a fair learning score. WDL Brier here measures outcome-label fit on generated
games; held-out value MAE above compares expected score against Stockfish.

### Throughput

The matched idle-CPU benchmark compared Gumbel with variance PUCT + RPO using
five workers x 32 games, 30 seconds of warm-up, and two measured phases.

| Method | Actual neural rows/sec | Usable positions/hour | Policy-eligible positions/hour |
|---|---:|---:|---:|
| Gumbel | 79,357 | 1,236,817 | 861,357 |
| Variance PUCT + RPO | 80,045 | 1,157,926 | 886,129 |

The combination has similar neural-row throughput (+0.9%), lower completed
usable throughput (-6.4%), and higher policy-eligible throughput (+2.9%) in this
short measurement. Earlier large apparent gains under unequal Stockfish CPU
load should not be used as method speedups. Plain variance PUCT's earlier
benchmark under shared Stockfish load measured 54,460 rows/sec versus Gumbel's
50,847; it was not part of this later idle-CPU pair.

### Interpretation and next priority

The intervals jointly resample source games, keeping all correlated positions
and both training seeds together. They quantify held-out sampling uncertainty
conditional on these two trained runs, with Stockfish labels treated as fixed.
They do not quantify confidence over arbitrary future training seeds or engine
label error. Two short training runs are a consistency check, not a long-run
learning or Elo result.

The next learning comparison should prioritize **Gumbel versus variance-scaled
PUCT without RPO** over a longer training budget. Keep the starting checkpoint,
exposure budgets, and evaluation cohort matched. The current results do not
justify selecting RPO as the stronger learner solely from frozen-target regret.

Detailed source results: `raw-learning-test/summary.json`,
`seed-43/raw-learning-test/summary.json`,
`seed-43/paired-two-seed-intervals.json`, and `seed-43/two-seed-summary.json`.

## Reproduce and extend the measurements

Historical output directories contain frozen manifests and must not be reused
for new code or settings. Use new output names. The full set is for locked
comparisons; tune parameters on the tuning partition only.

```bash
# Frozen-search comparison; config/search_study_finalists.toml contains all six arms.
.venv/bin/python scripts/compare_search_methods.py \
  --positions artifacts/eval/golden-v1-2026-10-18/positions.json \
  --reference artifacts/eval/golden-v1-2026-10-18/reference-200k \
  --checkpoint artifacts/self_play/plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run/actor-000550.pt \
  --arms config/search_study_finalists.toml --split test --simulations 256 \
  --concurrent-games 32 --out artifacts/eval/search-next/frozen

# Controlled collection + learning fork. Repeat for each config under matched seeds.
.venv/bin/python scripts/train_search_study.py \
  --config config/self_play_puct_variance.toml \
  --initialize artifacts/self_play/plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run/actor-000550.pt \
  --golden-positions artifacts/eval/golden-v1-2026-10-18/positions.json \
  --phases 3 --games-per-phase 320 --exposures-per-phase 50000 \
  --workers 5 --concurrent-games 160 --out artifacts/eval/search-next/learning/puct_variance

# Evaluate all completed milestone weights without search.
.venv/bin/python scripts/evaluate_search_learning.py \
  --positions artifacts/eval/golden-v1-2026-10-18/positions.json \
  --reference artifacts/eval/golden-v1-2026-10-18/reference-200k \
  --initial artifacts/self_play/plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run/actor-000550.pt \
  --learning artifacts/eval/search-next/learning --split test \
  --out artifacts/eval/search-next/raw

# Throughput: same machine load, workers, weights, warm-up, and completion budgets.
.venv/bin/python scripts/bench_collection.py \
  --config config/self_play_puct_variance.toml \
  --checkpoint artifacts/self_play/plain-s256-lr2e4-from-rgsc-it441-2026-10-16/run/actor-000550.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --workers 5 --concurrent-games 160 --warmup-seconds 30 \
  --fresh-positions 20000 --phases 2 --carry-over \
  --output artifacts/eval/search-next/throughput-variance
```

Raw network evaluation isolates learning; it does not establish search-time
playing strength. A longer matched Gumbel-versus-variance-PUCT training comparison
should follow, with intermediate raw held-out evaluations. At selected milestones,
play paired matches with a common inference search to isolate weight quality,
then compare each method's own search to measure end-to-end deployment strength.
Keep exposure and wall-time comparisons separate. More training seeds are needed
before claiming robust learning gains.

The byte counters describe payload copies/transfers, not measured hardware memory
bandwidth utilization. Use usable and policy-eligible completed positions/hour
alongside actual neural rows/second; row throughput alone can hide longer games.

## Artifacts and validation

Local artifact root: `artifacts/eval/search-study-2026-10-18/`.

| Evidence | Relative path |
|---|---|
| Frozen six-arm results | `locked-search-all-s256/summary.json` |
| First-seed raw predictions | `raw-learning-test/summary.json` |
| Second-seed raw predictions | `seed-43/raw-learning-test/summary.json` |
| Paired source-game intervals | `seed-43/paired-two-seed-intervals.json` |
| Combined per-seed and mean metrics | `seed-43/two-seed-summary.json` |
| Learning loss / policy / value traces | `learning/<method>/train.jsonl` and `seed-43/learning/<method>/train.jsonl` |
| Frozen choices | `frozen-selection.json` |

Golden roots are in `artifacts/eval/golden-v1-2026-10-18/positions.json`;
labels are in its `reference-200k/labels.jsonl`. Labels SHA-256:
`9386d7b21772c0384123b09f07d4c9e54a8f6b2a49ed91b2f15c81094a74b358`.
Data and weights remain local ignored artifacts; this document preserves the
main tables in git. Training loss, policy CE/entropy/KL, value CE and Brier are
diagnostics tied to each method's changing data/targets; cross-arm loss alone
cannot rank learning capability. Use the common held-out regret and value metrics.

Integration validation: 2,785 default tests passed, 37 extended tests deselected.
Tests cover native/scalar selection agreement, budgets and terminal backups,
RPO normalization and unchanged traversal, seeded exploration, shared inference
dispatch, config identity, CLI presets, zero-noise evaluation, and saved settings.
Real BF16 CUDA smokes exercised two concurrent games with variance PUCT + RPO
against Stockfish, and a mixed Gumbel-versus-variance-PUCT checkpoint match with
a local opening. Games were intentionally capped and are not strength evidence.

See the [detailed dated study log](SEARCH_METHOD_STUDY_2026-10-18.md) for tuning,
phase-level results, earlier experiments, and reference-budget checks.
