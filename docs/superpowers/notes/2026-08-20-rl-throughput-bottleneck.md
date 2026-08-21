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

## 6. Correctness review: does generation search like inference? (2026-08-20)

Requested check before optimizing further: during rollout generation, do we
genuinely spend the search budget, and do we pick the best move the same way
inference does? (Training/label-consumption deliberately out of scope.)

**Three drivers share one generator.** `search._halving_stepwise` is driven by
`scripts/generate_search_rollouts.py:305` (generation),
`scripts/eval_vs_stockfish.py:543` (inference), and
`search.select_value_search_halving:867` (sync wrapper). Selection is therefore
**identical by construction** — `best = max(survivors, key=score)` at
`search.py:831` is shared code, not duplicated logic.

**The budget is genuinely spent.** Measured over 48,186 real rollout rows
(`artifacts/rollouts/nightly/unaligned_archive_20260713_20260715/session_20260715_230000.parquet`,
all `search_budget=2048`), summing the recorded per-arm `arm_evals_spent`:

| metric | value |
| --- | --- |
| evals spent / position | mean 2039.3, median 2048, p10 2047 |
| budget realization | **mean 99.6%, median 100.0%** |
| arms / position | mean 15.68, max 28 |

So `max_depth=4` does **not** starve frontiers at budget 2048; the earlier
worry that the tree runs out of nodes before the budget is spent is dead.

**One real divergence: root arm selection.**
`generate_search_rollouts.py` defaults `--gumbel-root-sampling` to **True**
(line 459-461) and no launcher overrides it; `eval_vs_stockfish.py` passes no
`rng` and `HalvingConfig.gumbel_root_sampling` defaults **False**. So
generation samples root arms via Gumbel-Top-k while inference takes a
deterministic top-m-by-prior cut. Scope and consequence, same 48,186 rows:

- Live in **87.2%** of positions (the other 12.8% have fewer mapped legal
  moves than `top_m=16`, where Gumbel-Top-k selects *every* move and the
  permutation is irrelevant).
- But it changes the emitted label in **0.04%** of positions: the chosen best
  arm sits at prior rank 0 (median) / 3 (p90), and is at rank >=16 — i.e.
  outside what a deterministic top-16 cut would have contained — only 0.04% of
  the time.

Verdict: **benign.** Generation and inference agree on the selected move in
~99.96% of positions despite exploring different arm sets, so the Gumbel
default buys exploration diversity in the per-arm value rows essentially for
free. Left as-is.

**Bonus signal, same query:** search overrides the policy's own top-prior arm
in **39.9%** of positions, so the labels carry real information beyond the
policy that generated them.

**One real bug, fixed.** `search.py`'s budget-starvation fallback read:

```python
best = max(survivors, key=lambda arm: arm.score)
if best.score == float("-inf"):
    # Budget starvation: fall back to the highest-prior candidate.
    best = arms[0]
```

`arms` follows `picks = order[:top_m]`. Under `_prior_order` (inference)
`arms[0]` genuinely is the highest-prior candidate, but under Gumbel-Top-k
(**generation's default**) `order` is a permutation, so `arms[0]` is an
arbitrary draw and the fallback returned a random move while its comment
claimed otherwise. Now `max(arms, key=lambda arm: arm.root_log_prior)`.
Deterministic inference play is bit-identical (`arms[0]` is already the global
prior argmax there), pinned by
`tests/test_search_stepwise.py::test_budget_starvation_is_unchanged_for_deterministic_inference`;
the Gumbel path is pinned across 24 seeds by
`test_budget_starvation_falls_back_to_highest_prior_arm`, which failed on
20/24 seeds before the fix. Low blast radius (only the ~0.4% of positions that
starve, and generation drops those rows via the `backed_value is None` filter
at `generate_search_rollouts.py:328`), but it was silently wrong.

## 7. Measurement hygiene: this laptop cannot see sub-1.2x effects while the desktop is busy

Two consecutive paired A/B/B/A runs of the same two CPU optimizations both
failed their own drift control, on provably identical work
(`waves={324} evals={428016}` in every arm):

| run | total | decode_prep (contains the changes) | search_gpu (**cannot** contain them) |
| --- | --- | --- | --- |
| first | 1.177x | — | **1.270x** |
| second | 1.161x | 1.183x | **1.223x** |

`search_gpu` times only `model.forward_decode_grouped`; no Python-side change
can touch it. Its moving 1.22-1.27x is proof of contamination, and in the
second run the bucket containing the changes moved **less** than the bucket
that cannot — the signature of a uniform machine-wide slowdown, not a code
win. Raw sequence 81.8 -> 71.5 -> 83.4 -> 98.1 s shows a *superlinear* ramp;
A/B/B/A cancels drift only when it is **linear** (A at positions 1,4 and B at
2,3 share the same mean position, 2.5).

Cause, confirmed by the user: a **YouTube live stream in Firefox** running in
the background — already visible in the profile as `_ssl._SSLSocket.read`
13.45 s (11% of profiled time), plus `nvidia-smi` showing 81% GPU utilization
with **zero compute processes** (pure compositing) and firefox + four Isolated
Web Content processes at ~49% CPU.

Rules adopted:

- **Discard the first run.** A cold process pays CUDA context creation and
  kernel autotune: 66.5 s vs 49.3 s for the *identical* arm. A/B/B/A cannot
  cancel a one-off spike. `scripts/bench_python_opts_paired.sh` now runs a
  throwaway arm first.
- **Always report `search_gpu` as the drift control** and refuse to read
  `total` unless it is ~1.00x.
- **No browser/stream/GPU-monitor during timing runs.** Note `gnome-shell`
  alone holds ~82% SM on this box even when idle (`nvidia-smi pmon`), so it is
  a constant floor, not a variable — tolerable, unlike a stream.
- Sub-1.2x CPU wins are **not measurable end-to-end here**; size them with an
  isolated interleaved microbenchmark and combine with the cProfile share.

Strategic consequence: chasing individual ~1.05x Python wins is unfalsifiable
on this rig. Prefer structural changes large enough to clear the noise floor.
## 8. decode_project, decomposed — and two rejections (2026-08-21)

With `decode_prep`/`decode_project` finally timed (section 7), `decode_project`
was the largest bucket: 23.1s of 74.5s (30.9%), 54 us/node. "Largest bucket" is
not a design, so it was decomposed with wall-clock accumulators inside the real
consume path (`scripts/probe_project_phases.py`, 137,210 nodes) rather than
cProfile — **cProfile inflates this workload ~2x** (349 us/node profiled vs 174
us/node clean) and had mis-sized `torch.cat` by half.

| phase | us/node | share |
| --- | --- | --- |
| `kv_path` (2.05 `torch.cat`/node) | 15.02 | 25.6% |
| `d2h` (per-wave `.float().cpu()`) | 12.49 | 21.3% |
| `assemble` (value scalar + PositionEval) | 11.90 | 20.3% |
| `mapping` (memoized id/UCI loop) | 7.47 | 12.8% |
| `gather` (`torch.tensor` + `index_select`) | 4.57 | 7.8% |
| `sortlists` (canonical UCI sort + reorder) | 3.01 | 5.1% |
| `softmax` (`log_softmax` + `.tolist()`) | 2.42 | 4.1% |
| `movegen` | 1.51 | 2.6% |
| `kv_stack` | 0.21 | 0.4% |

**Done:** `assemble`'s value scalar plus `gather` and `softmax` are now one
call per wave instead of ~4 per node (commit `bd8ee83`). Measured 3.042x on
that phase (11.70 -> 3.85 us/node, +3.36s per 20-game run) and **bit-identical
on real labels**: 100.0000% best-arm agreement over 209 positions, max |delta|
exactly 0.0 on backed values and all 3,261 arm log-priors.

**Rejected — FBGEMM jagged ops.** The proposal targeted the jagged->padded
merge. That is only part of `decode_prep` (15.3%), and the `cat`/`pad` portion
specifically is ~3.0s wall of 74.5s = **4.0%, ceiling 1.04x**. A
CUDA-version-coupled dependency for <=4% loses to spending the same effort on
a 30.9% bucket.

**Rejected — batching the `kv_path` cats.** The obvious fix (group wave rows by
parent path length, one `stack`+`cat` per depth, hand out views) makes every
node's `path_kv` a view into a shared group tensor, coupling lifetimes on the
largest VRAM consumer in the process: at L=8, H=12, d=64, fp32, k+v, a path
token is 48 KiB, so 2048 evals x 8 games at mean path 2-3 is **1.5-2.25 GiB**
on a 7.66 GiB card that already peaks at 7.6 GiB. Storing paths pre-padded to
`max_depth+1` is worse (3.75 GiB). Trading OOM risk at the production
`concurrent_games` for ~12 us/node (6.9%, itself under the noise floor) is a
bad deal. Left alone.

**Also dead: the "lambda sort" idea.** `sorted(..., key=lambda i: ucis[i])`
looked like ~122 Python calls per node; it is **n calls, not n log n** —
`sorted` decorates once per element. Measured, `key=ucis.__getitem__` and
`sorted(zip(...))` are both *slower* (4.20/4.18 vs 3.23 us/node) with identical
output. No headroom.

### What is actually left, ranked

> **Superseded by section 9.** The ranking below rests on a CPU 67.6% / GPU
> 32.4% split derived from `perf_counter` buckets that were measuring CUDA
> *enqueue* time. `torch.profiler` puts real device work at 17.1%, so the
> pipelining ceiling is ~1.21x, not ~1.48x.

1. **CPU/GPU overlap (pipelining) — the biggest remaining lever.** CPU is 67.6%
   and GPU 32.4%, strictly alternating: `d2h` at 21.3% of `decode_project` is
   the first CUDA sync after the forward launch, i.e. 5.3s of GPU wait booked as
   CPU work, which only happens because nothing else runs during it. Wave N+1
   depends on wave N inside a game, so the overlap has to come from splitting
   the G concurrent games into two staggered groups: group A's Python work runs
   while group B's forward is on the GPU. Ceiling `1/max(0.676, 0.324)` =
   **~1.48x**, no new dependency and no VRAM increase. Smaller waves cost some
   GPU efficiency (157 us/row at 512-1023 rows vs 110 at 2048-4095), so budget
   for GPU share rising toward ~42%; still ~1.48x.
2. **Rust/PyO3 for `mapping` (7.47 us/node) plus the 21.3% `search_bookkeeping`
   bucket** — ceiling 1.27x for the tree half alone (not the 1.86x previously
   assumed, which was an artifact of the untimed-bucket bug).
3. `movegen`/`sortlists`/`kv_stack` — 7.2 us/node combined. Not worth it.


## 9. torch.profiler: the buckets were measuring enqueue time (2026-08-21)

Every timing bucket in this project is a bare `time.perf_counter()` delta with
no `torch.cuda.synchronize()`. Per the PyTorch benchmarking guide that measures
how long the CPU took to **enqueue** CUDA work, not how long the GPU took to
run it. That single fact explains every attribution puzzle in sections 7-8:
`search_gpu` wraps only `forward_decode_grouped`, so it saw launch time;
the real device time surfaced at whatever happened to sync next, which is why
`d2h` looked like 12.49 us/node of "CPU" work and why hoisting work above that
sync moved 6.7 us/node out of `d2h` and straight into `kv_path`.

`scripts/profile_torch_waves.py` measures it properly (`torch.profiler`,
CPU+CUDA activities, `schedule(skip_first=24, warmup=2, active=12)` so CUDA
context init and early autotune waves are not read as workload). Over 12
decode waves:

| | |
| --- | --- |
| Self CPU time total | **2.467 s** |
| Self CUDA time total | **422.6 ms** |
| **GPU busy** | **17.1%** |

So this loop is far more CPU-bound than the buckets implied (they said GPU
32.4%). Two plans die on this number:

- **CPU/GPU pipelining ceiling is ~1.21x, not ~1.48x.** Total device work is
  422 ms against 2,467 ms of host work; perfectly hiding *all* of it saves at
  most that 422 ms. Not worth a two-group scheduler rewrite.
- **TF32 (`set_float32_matmul_precision('high')`) is pointless here.**
  `aten::mm` is 61 ms of 422 ms GPU = **2.5% of wall**. There is no version of
  that trade that justifies risking label precision, which is why fp32 was
  chosen in the first place.

The confirmed top cost is thousands of tiny tensor ops per wave -- 36,671 over
12 waves, ~3,056 per wave, essentially all KV-cache plumbing:

| op | calls | CPU total | share |
| --- | --- | --- | --- |
| `aten::copy_` | 10,373 | 360.7 ms | 14.6% |
| `aten::to` | 3,889 | 294.2 ms | 11.9% |
| `aten::pad` / `constant_pad_nd` | 9,242 | 178.9 ms | 7.3% |
| `aten::cat` | 17,056 | 151.2 ms | 6.1% (and 23.4% of all GPU time) |

### Rejected after measuring (`scripts/bench_wave_suffixes.py`)

`_wave_suffixes` owns the `pad`+`stack` share, so three replacements were timed
with `torch.utils.benchmark.Timer` (which synchronizes CUDA correctly -- a bare
`perf_counter` here would repeat the original mistake). All three produce a
byte-identical padded batch (`torch.equal` vs current: True):

| variant | B=990 | B=330 |
| --- | --- | --- |
| A current (`F.pad` per node + `stack`) | 4.62 us/node | 1.336 ms |
| B preallocated `new_zeros` + slice copy | 3.98 us/node (1.16x) | 1.517 ms (**slower**) |
| C **NestedTensor** -> `to_padded_tensor` | **10.70 us/node (2.3x slower)** | 3.206 ms |

- **NestedTensor: rejected.** It is the right tool for *attention over* ragged
  sequences (the tutorial's 597us vs 951us), but as a pad+stack replacement the
  prototype API's bookkeeping costs more than the padding it avoids.
- **Preallocation: rejected.** 1.16x at B=990 inverts to a regression at
  B=330, i.e. it depends on wave size, and is worth ~0.27s of a 66s run.

`_wave_suffixes` is therefore left alone.

### What the profiler says is actually left

`ProfilerStep` self CPU 403 ms (16.35%) plus `decode_wave` self CPU 446 ms
(18.08%) = **~34% of wall in the Python interpreter itself** -- movegen, vocab
mapping, and search bookkeeping. That is the Rust/PyO3 target, and it is now
the only remaining lever with real headroom. Ranked, with honest ceilings:

1. **Python -> Rust for movegen/mapping + tree bookkeeping** — ~34% of wall.
2. `torch.compile(mode="reduce-overhead")` / CUDA graphs to kill launch
   overhead on the ~3,056 small ops per wave. Attractive in principle, but wave
   shapes vary every tick so it would recompile or re-capture constantly, and
   Inductor already crashed on the second shape in this codebase (section 4).
   Needs a shape-bucketing scheme first.
3. Pipelining — ~1.21x ceiling, large rewrite. Deprioritized.



## Provenance

- `artifacts/budget_sweep/summary.txt` + per-budget logs (30 games each).
- Data-stream rate: inline benchmark, 8,000 games, `config_kl01.toml`'s
  `[dataset]` section, seed 42.
- `scripts/bench_root_eval.py` is committed and rerunnable; `--compile`
  reproduces the Inductor crash on the second shape.
- The A-vs-B spike scripts were throwaway and are deleted; their numbers are
  transcribed in sections 4 and 4b.
