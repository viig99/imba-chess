# Why we can't run RL with rollouts: the throughput bottleneck, measured (2026-08-20)

Question asked: what actually prevents a closed RL/ExIt loop over search
rollouts? Answered by measurement, not estimate. All numbers on the local
RTX 3070 Ti (8GB), checkpoint_23, fp32:

- `scripts/rollout_budget_sweep.sh` — rollout cost vs `search_budget`
- `scripts/bench_root_eval.py` — where a single root forward's time goes, and
  a reproduction of the Inductor crash that keeps it uncompiled
- `scripts/rollout_equivalence_gate.sh` — flex-vs-dense rollout equivalence

## 1. It is not the network, and not the data pipeline

Streaming `Lichess/standard-chess-games` through `LichessDataset.stream()` with
the production filters (`min_avg_elo=2000`, `min_time_control_sec=180`) and the
pinned seed:

| games | elapsed | marginal rate |
|---|---|---|
| 500 | 32.3s | — |
| 8,000 | 58.3s | **288 games/s** steady-state |

(First game arrives after 30.5s of one-off parquet index/metadata fetch.)

**288 games/s = 1.04M games/hr.** The rollout generator consumes ~1,100
games/hr. The data pipeline is ~**940x faster than the thing it feeds**. A
faster internet link changes nothing about rollout throughput.

Two places the link *does* matter, neither of them this bottleneck:
- Training at `num_workers=0` (forced by the sharding bug, see
  `rollout-sharding-alignment-bug` memory) serializes stream fetch into the
  train loop: 51.96 games/step ÷ 288 games/s = **~180 ms/step** of
  un-overlapped fetch on top of ~110 ms/step of compute, which is exactly the
  measured 110.8 → ~280 ms/step regression. Materializing the ~300k-game
  seeded prefix locally removes that without reintroducing the sharding bug.
- Remote GPU rental sync becomes free. But note the 2026-07-15 finding: the
  rented 5090 was **2.5-3x slower per shard** than local because the workload
  is CPU/overhead-bound and vast.ai cgroup-limited us to 23 cores. Rent cores,
  not FLOPs.

## 2. Search budget is not the main lever either — root_eval is a floor

`scripts/rollout_budget_sweep.sh`, 30 games, G=8, fp32, fixed seed. Numbers are
from each run's own `--profile` total (29 games / 323 labeled positions), which
excludes ~30s of one-off startup that wall time carries:

| budget | search evals | root_eval | search_gpu | bookkeeping | s/game | games/hr | s/label |
|---|---|---|---|---|---|---|---|
| 2048 (production) | 660,942 | 33.5s (35.7%) | 31.8s (33.9%) | 28.4s (30.3%) | 3.23 | 1,113 | 0.290 |
| 512 | 165,334 | 28.6s (58.3%) | 12.6s (25.7%) | 7.7s (15.7%) | 1.69 | 2,126 | 0.152 |
| 128 | 41,295 | 27.0s (73.1%) | 8.0s (21.7%) | 1.8s (4.9%) | 1.27 | 2,829 | 0.114 |
| 32 | 10,308 | 23.9s (80.2%) | 5.3s (17.6%) | 0.5s (1.8%) | 1.03 | 3,503 | 0.092 |

(The 2048 row's 1,113 games/hr independently reproduces the nightly cron's own
measured 1,030-1,114 games/hr — the bench is calibrated against production.)

**`root_eval` is budget-independent**: 33.5s → 23.9s while search evals fall
**64x**. Cutting the budget 64x buys only **3.1x** total, and leaves root_eval
at 80% of all time. Python `search_bookkeeping` — the thing two prior
optimization projects targeted (cozy-chess adoption, the proposed PyO3 crate) —
collapses to **1.8%** in the regime RL would actually use. It is a non-issue
there.

This matters because the literature RL target needs *few* simulations: Gumbel
MuZero (Danihelka et al., ICLR 2022) gets a policy-improvement target at 16-50
sims. We pay 2048 evals/position. So the budget *should* come down — and when
it does, root_eval is 100% of the problem.

## 3. root_eval is per-call overhead, not compute

`scripts/bench_root_eval.py`, eager (as the rollout generator runs it —
`generate_search_rollouts.py` hardcodes `compile_model=False`):

| games | plies | tokens | blockmask | model fwd |
|---|---|---|---|---|
| 1 | 40 | 40 | 0.35 ms | **94.83 ms** |
| 1 | 80 | 80 | 0.34 ms | **94.09 ms** |
| 1 | 160 | 160 | 0.35 ms | **89.88 ms** |
| 8 | 80 | 640 | 0.37 ms | 96.85 ms |
| 8 | 160 | 1280 | 0.35 ms | 125.81 ms |
| 32 | 80 | 2560 | 0.52 ms | 332.71 ms |

**Flat ~95 ms from 40 to 160 tokens — a 4x change in work costs nothing.** The
forward is pure dispatch/launch overhead below ~1280 tokens. Block-mask
construction is 0.4%; the reverted `_IncrementalRootCache` (2026-07-15) was
attacking sequence *length*, which was never the cost.

Why each labeled position pays a nearly-full 95 ms: with `--sample-every-n-plies 8`,
at any scheduler tick typically only one of the G=8 concurrent games sits on a
sampled ply, so root evals merge only ~**1.3 at a time** (23.9s ÷ 95 ms ≈ 251
calls for 323 positions). The G-game scheduler batches search waves well and
root evals barely at all.

## 4. Candidate A (REJECTED — see 4b): fixed-shape padding + torch.compile

Compiled, one static shape: **4.69 ms vs 94.83 ms — 20x.** But compiling a
*second* distinct shape crashes:

```
torch._inductor.exc.InductorError: LoweringException: AssertionError
  args[3]: Subgraph(name='sdpa_score0', ...)
  args[4]: (s37, s71, ...)   # symbolic shapes
  ...  Subgraph(name='sdpa_mask0', ...)
```

Inductor cannot lower `flex_attention`'s mask/score subgraphs under dynamic
symbolic shapes. That is the real reason `compile_model=False` is hardcoded
here, and it is the same family as the 2026-07-18 `dynamic=True` recompilation
storm.

A now-deleted throwaway bench tested the implied fix: pad every root batch to
one of a few **fixed** token counts (expressed as one extra trailing document
in `seq_offsets`, so the jagged block mask isolates the padding), then
`torch.compile(model, dynamic=False)` sees exactly one static graph. Numbers
below are kept because they are what candidate B had to beat.

**Correctness:** padded vs unpadded logits on the real tokens agree to
`max|Δlogit| ≈ 4e-6` (fp32 noise) across every workload tested — padding
documents do not leak.

**Throughput**, six different real workloads per bucket (all compile clean, no
Inductor crash):

| pad bucket | compiled | eager | speedup |
|---|---|---|---|
| none (single static shape, 40 tok) | 4.69 ms | 94.8 ms | **20.2x** |
| 512 | 12.6-13.9 ms | 96-103 ms | 7.2-8.2x |
| 768 | 16.0-22.3 ms | 101-109 ms | 4.5-6.8x |
| 1024 | 22.9-34.5 ms | 117-124 ms | 3.4-5.4x |

Compiled cost tracks bucket size (real compute); eager is flat (pure
overhead). So a **tier of small buckets** — {128, 256, 512}, each its own
compiled graph — is the design: an average 78-ply game lands in the 128
bucket at roughly 5-6 ms, i.e. **~12-15x off root_eval**.

## 4b. Spike: dense-mask SDPA beats bucketed compile (2026-08-20, same day)

Before committing to the padding+compile design above, we spiked the
alternative: keep eager, but replace `flex_attention` with
`F.scaled_dot_product_attention` plus a materialized additive mask, on the
**inference path only**. Rationale: flex_attention exists to avoid
materializing an S x S mask, and at inference lengths (<= 512 tokens) a
`[1, 12, S, S]` fp32 mask is ~12 MB -- cheap. Its reason for existing does not
apply here, but its eager Python dispatch cost (~11-12 ms per layer x 8
layers = the measured ~95 ms) very much does.

What has to be replicated exactly: the jagged doc-causal mask, and the
`score_mod`'s **per-head** relative-position bias
`_ps_w[h, clamp((k-q) + max_seq_len - 1, 0, 2*max_seq_len - 2)]`. That bias is
additive and per-head, so it cannot fold into a bool mask -- the SDPA
`attn_mask` must be additive float `[1, H, S, S]`. It is Toeplitz and
data-independent, so it is precomputable per shape.

**Arbitration against a float64 reference** (isolated attention op) --
establishes neither implementation is wrong:

| impl | S=80 | S=320 |
|---|---|---|
| flex_attention | 1.03e-6 | 8.13e-7 |
| SDPA (EFFICIENT) | 1.67e-6 | 1.89e-6 |
| SDPA (MATH) | 7.84e-7 | 8.09e-7 |

**Full 8-layer model, SDPA vs flex** (fp32, EFFICIENT backend):

| S | max abs dlogit | top-1 agree | top-5 set agree | flex | SDPA | speedup |
|---|---|---|---|---|---|---|
| 40 | 8.3e-6 | **100%** | **100%** | 82.9 ms | 2.17 ms | **38.2x** |
| 80 | 7.7e-6 | **100%** | **100%** | 75.2 ms | 2.74 ms | **27.4x** |
| 160 | 1.3e-5 | **100%** | **100%** | 76.4 ms | 3.93 ms | **19.5x** |
| 320 | 2.2e-5 | **100%** | **100%** | 79.3 ms | 6.79 ms | **11.7x** |
| 640 | 4.2e-5 | **100%** | **100%** | 82.7 ms | 16.48 ms | **5.0x** |

Better than bucketed compile (4.69-13.9 ms) at every size, with no compile
step, no bucket management, no Inductor fragility, and no warmup. **Adopt B,
drop A.**

**Bug found while spiking, worth a permanent test:** `_ps_w` is a *per-layer*
parameter. The first version built one mask from `layers[0]` and shared it
across all 8 layers, producing a deterministic `max|dlogit| ~3.0` and only
77-94% top-1 agreement -- identical under both SDPA backends, which is what
revealed it as a logic bug rather than kernel precision. Any implementation
MUST build the additive bias per layer, and the test suite should pin that.

**Scope limit:** inference only. In training S can be ~4,000 tokens, where
`[1, 12, S, S]` is ~800 MB. Training stays on flex_attention.

**Honest ceiling:** this fixes root_eval, which is 80% of rollout time at
budget 32 but only 36% at production budget 2048. Projected: at budget 2048
~1,113 -> ~1,700 games/hr (1.5x); at budget 128 ~2,829 -> ~9,700 games/hr;
at budget 32 ~3,503 -> ~15,400 games/hr. The large numbers still require the
budget reduction, which still needs its own label-quality validation.

## 4c. Implemented (2026-08-20)

Adopted B. The whole change is three pieces and threads no flags:

- `create_batch_dense_mask` (`hstu_model.py`) — `[S, S]` bool, doc-causal,
  admitting exactly what `create_batch_block_mask` admits.
- `SequentialTransductionUnitJagged.forward` dispatches on mask type: a plain
  `Tensor` takes the SDPA path, a `BlockMask` takes flex. `_additive_mask`
  folds that layer's own `_ps_w` relative-position bias into an additive
  `[1, H, S, S]` mask.
- `_forward_model` builds the dense mask instead of a BlockMask — one line,
  and it is the only production call site that changed.

No new config knobs, no changes to `merged_executors.py`,
`generate_search_rollouts.py`, or `hstu_model.forward`'s layer loop. The
rollout script's `compile_model=False` is now simply correct rather than a
workaround.

**The selection criterion is batch SHAPE, not train-vs-eval.** The additive
mask is quadratic in S, so dataset-sized batches must keep BlockMask:
`ignite_evaluator` (fast-val hr@10), `eval_value_loss`, and
`eval_value_by_progress` all run ~4,000-token batches and correctly still
build a BlockMask. `_additive_mask` raises above a 256 MiB budget so getting
this wrong fails loudly instead of allocating 805 MiB per layer.

**Unexpected bonus — the dense path is compile-robust.** Under
`torch.compile(dynamic=True, fullgraph=False)` (eval's exact setting) it
generalizes across varying token *and* document counts with identical argmax
and <= 1.1e-5 drift. That is precisely what flex could not do, and it means
eval's `compile = true` is no longer sitting on an Inductor crash.

Tests: `tests/test_dense_attn_mask.py`, 7 cases — doc-causal semantics,
equivalence against `BlockMask.mask_mod` itself, flex-vs-dense logit
agreement, `return_kv` parity, the per-layer-bias regression pin, the size
guard, and compile-across-shapes. Suite 291 -> **297 passed**.

**Verified so far:** unit + integration tests, and the compile-robustness
check. **Still unverified (needs a free GPU):** the 20-game end-to-end
move-agreement gate (`scripts/rollout_equivalence_gate.sh`, thresholds >= 99%
agreement / p99 |dv| <= 1e-3), the throughput re-measurement, and a smoke run
of `eval_vs_stockfish.py` — eval compiles in production, so its numbers must
not be trusted until it has actually been run on this path.


## 4d. Measured after the fix (2026-08-20) — all gates PASS

Same bench, same 30 games / fixed seed / G=8 / fp32, before vs after:

| budget | s/game | games/hr | speedup | root_eval | labels/hr | fresh steps/hr |
|---|---|---|---|---|---|---|
| 2048 (prod) | 3.23 → **1.52** | 1,113 → **2,373** | **2.13x** | 33.5s → **1.0s** (33.5x) | 26,427 | 21.4 → **45.7** |
| 512 | 1.69 → **0.55** | 2,126 → **6,525** | **3.07x** | 28.6s → 0.9s (31.8x) | 72,675 | 40.9 → **125.6** |
| 128 | 1.27 → **0.27** | 2,829 → **13,385** | **4.73x** | 27.0s → 0.8s (33.8x) | 149,077 | 54.5 → **257.6** |
| 32 | 1.03 → **0.21** | 3,503 → **17,115** | **4.89x** | 23.9s → 0.8s (29.9x) | 190,623 | 67.4 → **329.4** |

Every cell beat the projection in 4b (which guessed ~1,700 / ~9,700 / ~15,400).
`root_eval` is now 2.2-13.9% of time instead of 35.7-80.2%.

**Search behaviour is unchanged, verified two ways:**
- Wave and eval counts match the pre-change run *exactly* at all four budgets
  (465/660,942 · 395/165,334 · 306/41,295 · 231/10,308) — the search made the
  same choices throughout the tree, not just at the leaves.
- `scripts/rollout_equivalence_gate.sh` (20 games, budget 2048, G=6, flex vs
  dense in the same build): 209 rows both arms, **100.0000% best-arm move
  agreement**, p99 |Δv| = **8.19e-7**, max 9.84e-7 — three orders of magnitude
  inside the 1e-3 gate.

**Eval smoke test PASSES** (`artifacts/eval/sdpa_smoke10.json`): 10 games vs
SF2200, exit 0, `compile = True` confirmed in `run_config`, legal coverage
1.0000, no Inductor crash. So the dense path is compile-safe on CUDA too, not
just CPU. (The 0.70 score on 10 games is statistically meaningless — SE ≈ 0.15
— and is not a strength claim.)

**The bottleneck has moved, and it moved differently per budget:**

| budget | largest bucket now | second |
|---|---|---|
| 2048 | `search_bookkeeping` **53.0%** (Python) | `search_gpu` 44.6% |
| 512 | `search_gpu` 60.3% | bookkeeping 33.6% |
| 128 | `search_gpu` 71.7% | bookkeeping 17.1% |
| 32 | `search_gpu` 77.8% | bookkeeping 6.8% |

This partially **reverses** section 5's ranking. At production budget 2048,
Python `search_bookkeeping` is once again the single largest bucket (53%),
which is exactly the criterion the 2026-07-18 spec set for reopening the PyO3
Stage-3 discussion. At the low budgets an RL loop would use, it stays small
(6.8% at budget 32) and `search_gpu` — the `forward_decode` wave path, which
this change did not touch — dominates instead.

One cost to note: peak VRAM rose to ~7.6 GiB at G=8/budget 2048 (the additive
mask is real memory). It completed, but that is close to the 7.66 GiB card;
`concurrent_games = 6` remains the safe local default.

## 5. What this adds up to

Ranked by size, at the search budget an RL loop would actually use:

1. **Eager flex_attention per-call overhead** — ~95 ms/call, 73-80% of rollout
   time at budget ≤128. Fix: fixed-shape padding + `dynamic=False` compile.
   Measured 7-20x depending on bucket. **Not yet built.**
2. **Root-eval batch merging** — ~1.3 positions/call because sampled plies are
   sparse and unsynchronized across the G games. Densifying
   `--sample-every-n-plies` toward 1 both multiplies labels per game and makes
   every tick a full G-wide merge. Free; changes only a flag and the scheduler's
   tick condition.
3. **`search_budget` 2048 → ~128** — 2.5x today, worth more once (1) lands, and
   defensible against Gumbel MuZero's 16-50 sims. **Quality risk: unmeasured.**
   Needs a label-agreement check against 2048 before adoption.
4. **`num_workers=0` training tax** — ~180 ms/step of un-overlapped stream
   fetch. Fix: local corpus materialization, or fix the sharding math.
5. **Python `search_bookkeeping`** — 30% at budget 2048, **1.8% at budget 32**.
   Do not spend more effort here; the PyO3 Stage 3 NO-GO from 2026-07-18 is
   even more strongly NO-GO in the low-budget regime.

Order-of-magnitude, combining (1)+(2)+(3) and nothing else: ~1,100 games/hr →
plausibly ~9,000 games/hr, and labels/hr from ~12,400 to well over 100,000.
That is the difference between one ExIt iteration per night and one per hour.

**Caveat on all of it:** throughput is necessary, not sufficient. The
measurement floor documented in the same session (identical checkpoint_23
scoring 0.5850 and 0.5975 on the same 200-game protocol) means even a working
RL loop cannot currently be *evaluated* below ~60 Elo of effect. Fixing
throughput without fixing the ruler produces faster nulls.

## Provenance

- `artifacts/budget_sweep/summary.txt` + per-budget logs (30 games each).
- Data-stream rate: inline benchmark, 8,000 games, `config_kl01.toml`'s
  `[dataset]` section, seed 42.
- `scripts/bench_root_eval.py` is committed and rerunnable; `--compile`
  reproduces the Inductor crash on the second shape.
- The A-vs-B spike scripts were throwaway and are deleted; their numbers are
  transcribed in sections 4 and 4b.
