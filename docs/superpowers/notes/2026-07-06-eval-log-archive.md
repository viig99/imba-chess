# Eval log archive — details pruned from the README (2026-07-06)

The README's results were consolidated into a single progression table. This
file preserves the full per-run diary entries, per-color splits, secondary
observations, and superseded usage examples verbatim, so nothing is lost for
future analysis or agents working on the project. Numbers here are final;
interpretation bullets reflect understanding at the time of each run.

## Protocol (applies to everything below)

Stockfish over UCI at 0.05s/move, `UCI_Elo` limited per segment, 100 games
per configuration, seed 42, colors alternating. Score = (wins + 0.5 × draws)
/ games; ±~0.05 standard error at 100 games.

## Results vs Stockfish 1400 (v3 pipeline)

Setup: `greedy` / `value_search_d2` on checkpoint `best_hr10_checkpoint_6`
(hr@10 = 0.9131, epoch 6); `value_search_halving` on the run's eventual best
checkpoint, `best_hr10_checkpoint_10` (hr@10 = 0.9304).

| Move selection | W / D / L | Score rate |
|---|---|---|
| `greedy` | 7 / 28 / 65 | 0.21 |
| `value_search_d2` (K=16, λ=0.05) | 22 / 16 / 47 @ 85 games | 0.34 (final, 100 games) |
| `value_search_halving` (N=256, `halving_rounds=0` auto) | 88 / 7 / 5 | **0.915** |

By color (`value_search_halving`): white 46/2/2 (score 0.94), black 42/5/3
(score 0.89) — the black-side weakness visible under `greedy`/`d2` is
essentially gone.

Takeaways at the time:

- The raw policy is markedly stronger than the previous run's (greedy 0.145
  → 0.21) — attributable to the v3 changes: placement-aware board encoding,
  game-level shuffle buffer, corrupt-game rejection.
- Search still multiplies the policy: d2 adds +0.13 over greedy on the same
  checkpoint, clearing the +0.05 gate that justified building the halving
  search.
- Halving search adds another large jump over d2 (0.34 → 0.915, ~+330 Elo
  over the same opponent at the two checkpoints tested) — SF1400 is
  saturated as an eval opponent for this policy.

## Results vs Stockfish 1800 (v3 budget scaling)

Setup: checkpoint `best_hr10_checkpoint_12` (hr@10 = 0.9349). Run on the
prefix-cache decode path (~12 s/game at budget 512/depth 6 — vs ~44 s/game
the uncached path needed for budget 256/depth 4).

| `value_search_halving` config | W / D / L | Score rate | ≈ Elo vs SF1800 |
|---|---|---|---|
| budget 256, depth 4 (defaults) | 38 / 17 / 45 | 0.465 | −24 |
| budget 512, depth 6 | 45 / 22 / 33 | 0.560 | +42 |
| budget 1024, depth 6 (ckpt_13, hr@10 0.9361) | 48 / 23 / 29 | **0.595** | +67 |

Understanding at the time:

- **Search does most of the lifting.** The full progression on v3
  checkpoints vs SF1400 is greedy 0.21 → d2 0.34 → halving-256 0.915. A pure
  imitation policy plays ~1170-Elo-equivalent chess; the same network under
  budgeted value search plays ~1800+.
- **The budget curve is flattening — the value oracle is becoming the
  bottleneck.** 256→512(+depth) bought +0.095; 512→1024 bought only +0.035
  (well inside noise, and the 1024 run even had a slightly newer checkpoint
  in its favor). Per the decision rule (stop buying search at the first flat
  doubling), further budget scaling is no longer the lever: the next Elo
  lives in better value labels.
- Draws grow with opponent strength (17–23 at 1800 vs 7 at 1400):
  converting drawn endgames is exactly where outcome-label noise hurts most
  — consistent with the flattening curve.
- Per-color splits at 1800 fluctuate across runs (e.g. 0.43/0.69 at
  512/d6) — 50 games/color is noisy; treat asymmetries as variance until
  they repeat.
- v3 checkpoints remain evaluable via `--config config/imba_chess_v3.toml`
  after the v4 architecture change.

## v4 trunk + value-net blend (SF1800)

Setup: v4 checkpoint `best_hr10_checkpoint_12` (hr@10 = 0.9468; 768d ×
8-layer trunk, `value_loss_weight` 1.0, from a still-running training job)
with the distilled value net (3.5M params, one epoch over ~200M
engine-evaluated positions). α = `value_net_alpha`: 0 = pure model value
head, 1 = pure distilled net.

α sweep at budget 256/depth 4 — run with a **mid-training** net checkpoint
(~45% trained, val soft-CE 0.6526 vs ~0.61 finished):

| α | W / D / L | Score rate |
|---|---|---|
| 0 | 51 / 18 / 31 | 0.600 |
| 0.25 | 44 / 30 / 26 | 0.590 |
| 0.5 | 41 / 23 / 36 | 0.525 |
| 1.0 | 15 / 26 / 59 | 0.280 |

Re-run at budget 1024/depth 6 with the **finished** net:

| α | W / D / L | Score rate |
|---|---|---|
| 0 | 56 / 19 / 25 | 0.655 |
| 0.25 | 62 / 15 / 23 | **0.695** |

Lessons recorded at the time:

- **Trunk scale beat label distillation to the punch.** v4's own value head
  (bigger trunk, doubled value loss weight) already fixed most of what the
  distillation was built to fix: at matched search (256/d4), v3 scored 0.465
  and v4 scores 0.600 — above v3's best-ever 0.595 that needed 4× the
  budget.
- **Oracle quality sets the search's exchange rate for compute.** v3's
  budget curve flattened (+0.035 for the last doubling); v4's is alive again
  (0.600 @ 256/d4 → 0.655 @ 1024/d6). The flattening was the value head, not
  the search.
- **An underfit oracle is worse than none.** The mid-training net degraded
  play monotonically in α. A direct probe showed it scoring a clean
  knight-up position at scalar +0.44 when its own training target (SF17
  win-rate model at +300cp) says ≈ +1.0. Never evaluate a half-trained
  oracle and conclude anything about the method.
- **Light mixing wins; heavy mixing loses.** Working hypothesis for the
  α=1.0 failure: Stockfish's win-rate targets encode *value under
  near-perfect play*, which rounds small advantages to draws — the wrong
  semantics against a fallible opponent — and the net sees history-free
  analysis positions, off-distribution from the search tree. At α=0.25 the
  model head stays in charge and the net acts as a second opinion: more
  wins, not more draws (62/15/23 vs 56/19/25).
- This is the λ=0 lesson again from a new angle: offline label accuracy is
  not in-search usefulness; oracles must be judged by play.

## Results vs Stockfish 2000 (full α grid)

Same setup (v4 `checkpoint_12`, finished net, 1024/depth 6):

| α | W / D / L | Score rate | ≈ Elo vs SF2000 |
|---|---|---|---|
| 0 | 41 / 22 / 37 | 0.520 | +14 |
| 0.25 | 49 / 29 / 22 | **0.635** | **+96** |
| 0.5 | 39 / 28 / 33 | 0.530 | +21 |
| 1.0 | 24 / 29 / 47 | 0.385 | −81 |

- **The blend's edge grows with opponent strength**: +0.040 over pure model
  head at SF1800, +0.115 at SF2000 (~2.3 SE — no longer a noise candidate).
  A clean humped curve peaking at α=0.25 in both settings.
- Working hypothesis for the mechanism: against stronger opponents there are
  fewer free wins from blunders, so value accuracy matters more — and the
  net's "value under strong play" semantics become more correct as the
  opponent approaches the play its labels assume. Testable corollary: the
  optimal α rises with opponent Elo.
- Even pure net (α=1.0, finished, 1024/d6) is respectable now at 0.385 —
  the earlier 0.280 collapse was substantially the underfit checkpoint, not
  the concept.
- Net position: the α=0.25 system scores 0.635 vs SF2000 (0.05s/move) —
  roughly a 2100-Elo-equivalent player on this ladder, built from a
  27M-param imitation policy, a 3.5M distilled value net, and a 1024-node
  search.

## Results vs Stockfish 2200 (α probes)

Same setup (v4 `checkpoint_12`, 3.5M net, 1024/depth 6). First rung where
the system scores below 0.5.

| α | W / D / L | Score rate | as White | as Black |
|---|---|---|---|---|
| 0.25 | 35 / 21 / 44 | 0.455 | 17/5/28 (0.39) | 18/16/16 (0.52) |
| 0.15 | 37 / 24 / 39 | 0.490 | 19/11/20 (0.49) | 18/13/19 (0.49) |

- α=0.15 vs 0.25: +0.035, inside 1 SE — a statistical tie, but the
  "optimal α rises with opponent Elo" prediction from the 1800→2000 trend
  called for 0.35 > 0.25 > 0.15 and is not supported. Revised working
  model: shallow α optimum in [0.1, 0.3], location not predictably tied to
  opponent strength.
- The reversed color split in the α=0.25 run (Black 0.52 > White 0.39) did
  NOT repeat at α=0.15 (0.49/0.49 dead even) — stays classified as
  variance per the two-runs rule.
- Implied absolute rating across rungs (score → Elo diff + rung label):
  ~1943 from SF1800, ~2096 from SF2000, ~2193 from SF2200 — monotone rise,
  consistent with UCI_Elo rung compression at 0.05s/move (labels are
  ordinal here, not absolute; SF's UCI_Elo is calibrated at long time
  controls, and the weakening mechanism is score-weighted random root-move
  picks over a full-strength search — measured depth 9–12, ~30–85k
  nodes/move at our settings, Stockfish 18).

### SF2200, second round: epoch-14 trunk + budget 2048 / depth 8

v4 `best_hr10_checkpoint_14` (hr@10 = 0.9500, two epochs newer), budget
2048, depth 8 (first depth-8 run; the even-horizon rule kept):

| α | W / D / L | Score rate |
|---|---|---|
| 0 | 37 / 28 / 35 | 0.510 |
| 0.15 | 38 / 27 / 35 | **0.515** |

- First score above 0.5 at this rung — but the combined lever pull
  (budget ×2 + depth 6→8 + two more trunk epochs + hr@10 0.9468→0.9500)
  bought only +0.025 over the e12/1024/d6 α=0.15 result (0.490), inside
  noise. Diminishing returns on every axis at once around ~2200.
- The blend edge vanished at this config: α=0.15 ≈ α=0 (+0.005). Pattern
  across the ladder: the better the trunk's own value head, the less the
  distilled net adds (+0.115 @2000/e12 → +0.035 @2200/e12 → ~0 @2200/e14).
  Consistent with trunk scale progressively eating the net's margin.
- Draw share rose with the deeper search (27–28 vs 21–24 at 1024/d6) —
  the usual signature of two more accurate players.
- Attribution between checkpoint/budget/depth was not decomposed (an
  e14 @ 1024/d6 leg would separate them); with the combined gain this
  small, the decomposition was judged not worth the eval time.

### SF2200, third round: epoch-23 trunk (Elo-weighted value loss) + budget 2048 / depth 8

v4 `best_hr10_checkpoint_23` (hr@10 = 0.9564), same 2048/depth-8 search as
the e14 round. This checkpoint is the first one trained past the
Elo-weighted value loss change (d45abc2): training resumed from epoch 18
with the same per-token Elo scale that weights policy CE now also
multiplying the progress-weighted value CE (`elo_loss_weight_strength`/
`alpha` unchanged, so stronger players' outcomes count as lower-noise value
labels).

| α | W / D / L | Score rate |
|---|---|---|
| 0 | 44 / 31 / 25 | **0.595** |
| 0.15 | 46 / 20 / 34 | 0.560 |

- Pure-head score jumped 0.510 → 0.595 (+0.085) over the e14 checkpoint at
  the *identical* search config — the largest single-lever gain measured at
  this rung, and the first clear evidence the Elo-weighted value loss (or
  the extra epochs, not decomposed) is paying off, not just holding flat.
- The blend flipped sign: at e14 it was a wash (α=0.15 ≈ α=0, +0.005); at
  e23 it actively costs 0.035 (0.560 vs 0.595). Full trend across the
  ladder: +0.115 @2000/e12 → +0.035 @2200/e12 → ~0 @2200/e14 → **−0.035
  @2200/e23**. The distilled net's value has now crossed from neutral to
  net-negative as the trunk's own head strengthens — worth revisiting
  whether α=0.15 is still the right default, or whether the net should be
  dropped for this checkpoint family.
- Draw share fell back to 20 at α=0.15 (from 27 @e14) despite rising to 31
  at α=0 — the two configs are no longer just "same players, more draws";
  the α=0 line is now both scoring higher and drawing more, consistent with
  a genuinely stronger, more decisive pure head.
- Not decomposed: how much of the +0.085 comes from the Elo-weighted value
  loss specifically vs. five more epochs of plain training — an e18 (the
  resume point) @ 2048/d8 leg would isolate it.

## Historical: the λ sweep (v2 checkpoint)

An earlier sweep on a pre-v3 checkpoint (not comparable to the numbers
above) established two durable design facts, both baked into the current
defaults: **value-dominant scoring beats policy-dominant scoring** by ~+140
Elo inference-only (0.23 → 0.405 vs SF1400), and **λ = 0 collapses** (0.12)
— optimizing purely against the learned value head over-exploits its noise
(Goodhart), so the policy log-prob prior is a necessary regularizer, not a
cosmetic tiebreak. Score was flat across λ ∈ [0.05, 0.2];
`value_rerank_lambda = 0.05` has been the default since.

## Superseded usage examples (removed from README for brevity)

Value search (d2) against Elo-limited Stockfish:

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --model-move-policy value_search_d2 \
  --value-rerank-top-k 16 \
  --value-rerank-lambda 0.1 \
  --stockfish-limit-strength --stockfish-elo 1400 \
  --games 100
```

Halving with the beam-attribution override (`--halving-rounds 1` = pure
prior-guided beam, for attributing gains between tree depth and the
elimination feedback loop):

```bash
python scripts/eval_vs_stockfish.py \
  --checkpoint artifacts/checkpoints/best_hr10_*.pt \
  --model-move-policy value_search_halving \
  --search-budget 256 \
  --halving-rounds 1 \
  --stockfish-limit-strength --stockfish-elo 1400 \
  --games 100
```

## Evaluation-during-training details (condensed in README)

- `fast_val` / `fast_test`: every `[training].eval_every_steps` over the
  first `fast_val_max_games` / `fast_test_max_games` games.
- `full_val`: every `[training].full_val_every_epochs` over
  `[dataset].val_max_games`; `full_test`: in `--eval-only` mode over
  `[dataset].test_max_games`.
- Metrics: `loss_ce`, `ppl`, `top1/top3/top5_acc`, `hr@10`, `mrr`,
  `token_count`, `game_count`.
- Best checkpoints selected by `hr@10` from `full_val`; last checkpoints by
  step cadence. On `--resume`, model/optimizer/scheduler/scaler/trainer
  state are restored and an immediate `fast_val`/`fast_test` health check
  runs.
