# Search methods: production usage and study results

All commands below run from the repository root. Gumbel remains the default.
The added methods use the same native chess transitions and node statistics,
compiled single-leaf CUDA decoder, KV workspace, root carry, batch scheduler,
collection workers, and replay schema. They need no checkpoint conversion.

**Status (2026-10-19): Gumbel is the default and the live recipe, now trained on
`improved_wdl` at `value_search_mix = 0.25`. Whether variance-scaled PUCT learns
better is open.** Variance-scaled PUCT without RPO is still the only method with
a repeated learning gain: raw policy regret in both short seeds of the screen
below. That screen predates two changes that bear on it:

- Gumbel's value label was fixed. Its old `search_wdl` label was about as poor
  as the raw network (see the value table below).
- PUCT self-play played the most-visited move. The first real PUCT run (2026-10-18)
  collapsed its initial-board games onto one repeated drawn line. Even on the same
  weights at iteration 0, 19.5% of its games ended in threefold repetition, against
  7.4% for Gumbel. Self-play now samples moves, through `sample_plies` and the late
  sampling options (see Train and resume).

A matched comparison is staged but not launched: both arms use `improved_wdl` at mix
0.25 and differ only in `[search]`. As an inference search, variance PUCT is weaker
on Gumbel-trained weights. On ema-554 at 512 simulations it scored 0.365 against
Stockfish 2600 over 200 games; earlier checkpoints scored about 0.44 with Gumbel.
It visits about 5 root moves, against Gumbel's 15, and misses low-prior best moves.
Evaluate with Gumbel. As a teacher it is also weaker under training noise. Every PUCT
visit target scores 0.013 to 0.019 more target regret than Gumbel's, and Lc0's
self-play FPU of 0 makes it worse. See the 2026-10-19 rerun below.

RPO is not advanced. Its preset factor (1/64) was tuned on frozen target regret,
which rewards concentrating mass on the best-Q move. It produces near one-hot
targets: entropy 0.23 against Gumbel's 0.53. It tied Gumbel on raw policy and had
higher value error. A retry should choose the factor for a Gumbel-like target
entropy, not for frozen regret. No method has established an Elo gain over Gumbel.

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
returns as parent Q, without the node's own network value. `[search]` options
`lc0_parent_q = true` (Lc0's parent Q, own evaluation counted as a visit) and
`paper_log_n = true` (PUCT-V's log N) select the exact forms; both default to false,
which keeps the study arithmetic and existing run identities. Neither has yet been
measured on a real network.

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

PUCT self-play chooses the played move with three `[search]` options. All of
them leave the training target unchanged, and they apply only while root noise
is on, so evaluations always play the most-visited move. Gumbel needs none of
them: its noise already varies the played move at every ply.

- `sample_plies`: game plies below it play a visit-proportional sample
  (AlphaZero's opening temperature).
- `late_temperature` and `late_value_cutoff`: from ply `sample_plies` on, play a
  `visits^(1/T)` sample among visited moves whose Q is within the cutoff of the
  most-visited move's.

All three default to 0, which keeps the most-visited move. They are dropped from
the run identifier at 0. `self_play_puct_variance.toml` sets 30, 1.0 and 0.05.
These settings were calibrated on stored targets after ply 30. They depart from
the most-visited move at 24.8% of positions, for a mean Q loss of 0.001 and none
above 0.05. Gumbel's played move departs from its greedy choice at 26.7%, with a
mean loss of 0.006 and 4.4% above 0.05. Whether they bring PUCT's threefold
rate down to Gumbel's has not yet been measured in games. Start buckets cover
plies 0, 1-30, 31-70 and 71-120, so `sample_plies` alone never reaches half of
the starts.
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

Every search now also records `improved_wdl`, the root-player WDL of its returned
policy: `sum_a policy(a) WDL_a`, with unvisited moves taking the WDL form of
completed Q's mixed value. Its W - L is the improved policy's completed-Q value; for
a PUCT visit target it equals `search_wdl`. `[learning] value_search_source =
"improved_wdl"` makes `value_search_mix` and the auxiliary TD targets use it; the
default `"search_wdl"` keeps existing runs' identity. Gumbel, same frozen actor and
test roots, expected-score error against the Stockfish root:

| Gumbel exploration | `search_wdl` MAE / bias | `improved_wdl` MAE / bias | Difference, 95% game bootstrap |
|---|---:|---:|---:|
| None (evaluation) | 0.1652 / -0.0779 | 0.1375 / -0.0037 | -0.0277 [-0.0309, -0.0246] |
| Training noise, seed 0 | 0.1660 / -0.0803 | 0.1376 / -0.0041 | -0.0284 [-0.0315, -0.0253] |
| Training noise, seed 1 | 0.1661 / -0.0801 | 0.1377 / -0.0039 | -0.0284 [-0.0316, -0.0253] |

The raw network scores 0.1660 / -0.0333. Results: `artifacts/eval/gumbel-improved-wdl-noise-2026-10-18/`.

### Training-noise rerun of every fixed variant (2026-10-19)

Self-play trains on searches run with root noise. This rerun scores those searches
directly, with every fix since the study above in place: Gumbel's `improved_wdl`,
the exact Lc0 parent Q (`lc0_parent_q`), PUCT-V's paper log N (`paper_log_n`), and
self-play FPU settings. Setup: same frozen actor-000550 and 4,110 test roots, 256
simulations, bf16. Each root has one noise-off search and two training-noise seeds,
and every arm uses the same noise draws. Differences are paired by source game, in
the same order as the table (negative favours the row).

| Variant | Target regret, training noise | vs variance PUCT | vs Gumbel | Visited root moves | Target entropy | Value label MAE |
|---|---:|---:|---:|---:|---:|---:|
| Gumbel | **0.0458** | -0.0132 | — | 14.9 | 0.52 | 0.1377 |
| Variance PUCT (recipe) | 0.0591 | — | +0.0132 | 6.9 | 0.88 | 0.1358 |
| Variance PUCT, Lc0 parent Q | 0.0588 | -0.0003 | +0.0130 | 6.8 | 0.88 | 0.1355 |
| Variance PUCT, self-play FPU 0 at the root | 0.0756 | +0.0165 | +0.0297 | 18.7 | 1.31 | 0.1375 |
| Variance PUCT, self-play FPU 0 everywhere | 0.0744 | +0.0153 | +0.0285 | 18.8 | 1.30 | 0.1391 |
| Lc0 self-play constants (PUCT, cpuct 1.2, no growth, FPU 0) | 0.0795 | +0.0204 | +0.0336 | 19.2 | 1.44 | 0.1399 |
| PUCT | 0.0635 | +0.0044 | +0.0176 | 8.7 | 1.08 | 0.1371 |
| PUCT, Lc0 parent Q | 0.0633 | +0.0043 | +0.0175 | 8.7 | 1.08 | 0.1368 |
| PUCT-V | 0.0648 | +0.0057 | +0.0190 | 8.6 | 1.10 | 0.1387 |
| PUCT-V, paper log N and Lc0 parent Q | 0.0645 | +0.0055 | +0.0187 | 8.6 | 1.10 | 0.1385 |
| PUCT + RPO | 0.0497 | -0.0094 | +0.0039 | 8.7 | 0.64 | 0.1363 |
| Variance PUCT + RPO | 0.0445 | -0.0146 | -0.0014 | 6.9 | 0.23 | 0.1349 |

The difference columns average the two seeds. Each seed's 95% interval excludes zero,
with two exceptions. Lc0 parent Q is significant in one seed only. Variance PUCT +
RPO against Gumbel is within noise, for example [-0.0047, +0.0013] in seed 0. The value
label is `improved_wdl` for Gumbel. For PUCT's visit target it equals `search_wdl`.

- Under training noise, every PUCT visit target is 0.013 to 0.019 worse than Gumbel's.
  Gumbel's target barely changes with noise on (0.0456 noise off, 0.0458 on).
  Variance PUCT's gets 0.004 worse.
- Self-play FPU 0, Lc0's self-play setting, is rejected. It nearly triples the
  visited root moves, but the policy target is raw visit counts. At 256 simulations,
  the visits it spends on bad moves enter that target directly. Lc0 self-play
  searches far more nodes per move. The option was not merged.
- The exact Lc0 parent Q and paper log N change target regret by 0.0003 or less.
  Both stay off by default.
- On the real network, the variance factor on cpuct averages 0.40 (median 0.37, p10
  0.21, p90 0.63), so variance PUCT's effective cpuct is about 0.7.
- Only RPO matches Gumbel's target regret, and only with near one-hot targets.
  Those targets did not train a better network in the screen below.
- Value labels are now within 0.005 MAE across methods.

Reruns with identical inputs are bit-identical (Gumbel: 12,330 of 12,330 searches).
Noise-off and training-noise searches share bf16 batches. An arm whose training-noise
searches differ therefore perturbs a few noise-off searches (29 to 69 of 4,110 roots,
no move changes), even though noise-off code paths are unchanged. Results, the
analysis script and the per-root variance log are in
`artifacts/eval/puct-selfplay-fpu-2026-10-19/`.

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

Update, 2026-10-19: this screen ran with Gumbel's old `search_wdl` value label
and with PUCT playing the most-visited move after the opening. Both have since
changed (see the status at the top), so the gain above is not yet a result
under the current recipe. Compare at matched hours, and evaluate both arms with
Gumbel. Also watch the raw policy's golden-set regret and the threefold rate.

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
