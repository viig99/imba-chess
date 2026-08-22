# Handoff: rollout-generation performance (toward a fused rollout + training loop)

Written 2026-08-21. Audience: an engineer or model picking this up cold.
Companion deep-dive with all raw numbers:
`docs/superpowers/notes/2026-08-20-rl-throughput-bottleneck.md` (sections 1-9).

---

## 1. The end goal, and why generation throughput is the whole problem

We want an **ExIt-style loop that generates search-labelled rollouts and trains
on them in the same process**: search produces a better-than-policy move
distribution, the policy distils toward it, the improved policy makes the next
round of search better.

The loop is **generation-bound, not training-bound**. Measured cost constants
(`docs/superpowers/notes/rollout-training-cost-constants.md`): training a label
is ~281x cheaper than producing it. So a fused loop's iteration time is
essentially generation time, and every throughput question is a generation
question.

Rough target: **~250k labels/hr** makes one meaningful experiment per night.
Current rate is **~10k labelled positions/hr** (see §3), so this needs roughly
an order of magnitude, of which maybe 2-3x has been identified so far.

**Correctness is not negotiable.** The product of generation is *training
labels*. A "faster" generator that shifts labels is worse than a slow one,
because it silently poisons every downstream experiment. Every change in this
area must pass the label gate in §7.

---

## 2. Generation methodology (what the code actually does)

Entry point: `scripts/generate_search_rollouts.py`.

### Per game
1. Replay a real Lichess game (streamed from HuggingFace) ply by ply.
2. At every Nth ply (`--sample-every-n-plies`, effectively every 8th), stop and
   **search** the position.
3. Emit one row per searched position (`RolloutRow`) containing the search's
   chosen move, its backed value, and per-arm detail.

### Per searched position
1. **Root forward** (`WorkRequest("root_eval")`): one model forward over the
   game prefix, producing policy logits, value logits, and the per-layer K/V
   cache that becomes the shared **prefix** for every search node.
2. **Search** (`search._halving_stepwise`, a generator): a
   sequential-halving / "MCTS-lite" tree search.
   - Root arms = `top_m=16` candidate moves, chosen by **Gumbel-Top-k** over
     policy priors (Danihelka et al., ICLR 2022) — `--gumbel-root-sampling`
     defaults **True** for generation — plus every *forcing* move (captures and
     checks) appended unconditionally.
   - `rounds = ceil(log2(num_arms))` rounds; each round splits the remaining
     budget across surviving arms, then eliminates the worst half by score
     (`backed_value + lam * prior`, `lam=0.05`).
   - The generator **yields `EvalRequest(batch=[(handle, board), ...])`** and is
     resumed with the corresponding `PositionEval`s. It never calls the model
     itself — this is what allows cross-game batching.
3. Selection: `best = max(survivors, key=score)`, shared by all three drivers
   (see §4), so generation and inference select identically.

### Cross-game batching
`BatchScheduler` (`src/imba_chess/eval/batch_scheduler.py`) runs `G` game
coroutines in **lockstep ticks**. Each tick: advance every slot to its next
`WorkRequest`, group requests by kind, run **one merged executor call per
kind**, scatter results back. Finished games are emitted in dataset-stream
order via a hold-back buffer so `--flush-every-games` / resume semantics are
unchanged.

So one "decode wave" is: the union of all G games' currently-popped search
nodes, evaluated in a single `forward_decode_grouped` call. Each node is one
new token whose K/V context is `prefix` + its own root->node ancestor path.

### Measured shape (budget 2048, G=8)

| quantity | value |
| --- | --- |
| searched positions / game | 10.45 |
| node evals / searched position | 2047.9 (budget is essentially fully spent) |
| **node evals / game** | **21,503** |
| nodes / decode wave | ~1,321 |
| legal moves / node | ~26-31 |
| **C-level torch/FFI calls per node** | **~494** |
| cozy-Board FFI crossings per node | ~97 |

One label costs ~2,048 model-evaluated nodes. That ratio is the fundamental
cost driver, and it is *deliberate*: see the label-quality data in §6.

---

## 3. Current state: where the time goes

Clean single-run bucket breakdown, 20 games, budget 2048, G=8. **Bucket shares
within one run are drift-immune** (everything scales together), which is why
they are quoted as shares rather than seconds.

| bucket | share | what it is |
| --- | --- | --- |
| `decode_project` | ~28-31% | per-node movegen + vocab mapping + legal-logit projection (CPU) |
| `search_gpu` | ~28-31% | the `forward_decode_grouped` call |
| `search_bookkeeping` | ~20-21% | heap/tree management inside `_halving_stepwise` (CPU) |
| `decode_prep` | ~12-15% | board encode + jagged->padded suffix K/V build (CPU) |
| `root_eval` | ~1.5% | root forward |
| `batch_build`, `ply_bookkeeping` | ~0.1% | negligible |

**But the buckets lie about CPU vs GPU.** They are bare `perf_counter` deltas
with no `torch.cuda.synchronize()`, so `search_gpu` measures kernel *enqueue*
time. `torch.profiler` with CUDA activities
(`scripts/profile_torch_waves.py`) gives the truth over 12 waves:

| | |
| --- | --- |
| Self CPU total | **2.39-2.47 s** |
| Self CUDA total | **381-423 ms** |
| **GPU busy** | **~17%** |

This is a **CPU-bound, launch-overhead-dominated workload**. Top host costs,
all KV/tensor plumbing, ~3,056 small ops per wave:

| op | calls / 12 waves | CPU total |
| --- | --- | --- |
| `aten::copy_` | 10,373 | ~310-361 ms |
| `aten::to` | 3,889 | ~294 ms (the `.cpu()` sync) |
| `aten::cat` | 17,056 | ~144-151 ms (also ~22% of all GPU time) |
| `aten::pad` | 9,242 | ~179 ms (**now removed**, see §5) |

Roughly **34% of wall time is the Python interpreter itself**
(`ProfilerStep` self 403 ms + `decode_wave` self 446-472 ms) — movegen, vocab
mapping, tree bookkeeping. That is the single largest remaining target.

Current arena cutover profile (same 12-wave harness, idle GPU) reduced self CPU
from **1.407 s to 1.003 s** and self CUDA from **297.4 ms to 239.0 ms**.
The 17,056-call `aten::cat` hotspot disappeared. Peak allocated VRAM at G=8
fell from **6,340.9 MiB to 4,165.7 MiB**; see §5.

Data loading is **not** a bottleneck (288 games/s steady state, ~940x faster
than generation), but the HF/fsspec streaming machinery does show up as
`_ssl._SSLSocket.read` and pathlib churn *outside* the timed buckets. Local
corpus materialization is a separate, known lever.

---

## 4. Code map

| path | role |
| --- | --- |
| `scripts/generate_search_rollouts.py` | generation entry point, `_TimingStats` buckets, game coroutines |
| `src/imba_chess/eval/batch_scheduler.py` | lockstep G-game scheduler (torch-free by design) |
| `src/imba_chess/eval/merged_executors.py` | merges G payloads into one root-eval / decode-wave call; owns the timing buckets |
| `src/imba_chess/eval/position_evaluator.py` | `CachedPositionEvaluator`: prefix K/V, shared per-turn `_KVArena`, decode request/result flow |
| `src/imba_chess/eval/search.py` | `_halving_stepwise` + scoring/elimination; `HalvingConfig` |
| `src/imba_chess/eval/cozy_bridge.py` | python-chess <-> owned native binding conversion, legal-move projection (`project_legal_moves`), native terminal detection |
| `src/imba_chess/model/hstu_model.py` | model; `create_batch_block_mask` (flex) and `create_batch_dense_mask` (SDPA) |
| `src/imba_chess/model/hstu_attention.py` | attention layer; dispatches SDPA vs flex_attention on mask type |
| `scripts/eval_vs_stockfish.py` | inference/play path — the other `_halving_stepwise` driver |

Three drivers share `_halving_stepwise`: generation, `eval_vs_stockfish.py`,
and the sync wrapper `select_value_search_halving`. **Selection logic is
therefore identical by construction** — don't duplicate it.

### Attention paths (important, easy to break)
Mask type is chosen by **batch shape**, not train-vs-eval:
- **dense `[S,S]` bool -> SDPA**: play/search batches (a few hundred tokens).
- **`BlockMask` -> flex_attention**: dataset-sized batches (~4,000 tokens),
  i.e. training *and* dataset eval (`ignite_evaluator`, `eval_value_loss`,
  `eval_value_by_progress`).

Those bullets describe full-sequence forwards, including root evaluation.
Cached one-token `forward_decode*` does manual prefix/suffix `einsum`, masking,
and softmax. The former FBGEMM operation packed ragged ancestor K/V into a
dense suffix for that manual decode; it did not feed Q/K/V into SDPA.

`flex_attention` is **not dead code**. At S=4,000 the dense additive mask would
be `12 * 4000^2 * 4` = **768 MB per layer**, over the 256 MiB guard in
`_additive_mask`. Dropping it would also require reimplementing the per-head
relative-position bias (`_ps_w` via `_generate_rab_score_mod`), which
`varlen_attn` cannot express. CPU now routes to dense/SDPA because torch 2.13
removed FlexAttention's CPU backward.

---

## 5. What has landed (all bit-identical unless noted)

Chronological, with measured effect. `f16b2cf..c210e49`.

| commit | change | measured |
| --- | --- | --- |
| `f957f0c` | dense-mask **SDPA inference path** replacing eager flex_attention below the wave boundary | `root_eval` 33.5s -> 1.0s (**33.5x**), total 2.13x |
| `f957f0c` | memoized cozy move -> (vocab id, UCI) lookup *(superseded -- removed in the native projection cutover, §12)* | 311 -> 154 ns/move (**2.02x**) |
| `1818a19` | **bug fix**: budget-starvation fallback returned a *random* arm under Gumbel (claimed "highest-prior") | failed 20/24 seeds before fix |
| `1818a19` | `decode_prep` bucket — the decode executor's CPU work was timed by **nothing** | exposed 46% of real work |
| `7fced57` | split into `decode_prep` / `decode_project` | — |
| `bd8ee83` | **batched** per-node value softmax + gather + log_softmax into one call per wave | 11.70 -> 3.85 us/node (**3.04x**) |
| `509a425` | hoist logits-independent movegen above the first CUDA sync | `d2h` 12.49 -> 5.84 us/node |
| `3bf08e3` | torch 2.13 compat: CPU attention -> SDPA; `torch.profiler` + `Timer` harnesses | 6 test failures fixed |
| `c210e49` | **fbgemm** `jagged_to_padded_dense` for the wave's suffix K/V; `path_kv` stored **token-first** `[t,L,H,d]` | 8.03 -> 0.87 us/node (**~9x**), `decode_prep` 14.8 -> 11.0s |
| current | replace persistent per-node full-path K/V copies with one shared append-only `_KVArena`; batch ancestor indices before one gather | 12-wave self CPU 1.407 -> 1.003 s (**1.40x**), self CUDA 297.4 -> 239.0 ms (**1.24x**), peak allocated VRAM 6,340.9 -> 4,165.7 MiB (**-34.3%**); 20-game G=8 labels and `arm_evals_spent` bit-identical |

Also fixed: `encode_cozy` (cached constants + inlined `scan_forward`) at
**1.346x** in isolation.

**Honest note on magnitudes**: several of these are individually 1.03-1.05x
end-to-end and are *not* separately visible in wall time on this machine (§6).
They were kept because each was measured in isolation and verified exact.

---

## 6. Measurement methodology — read this before trusting any number

This is the most expensive lesson in the project. Four separate paired
benchmarks were invalidated before the cause was understood.

1. **Never use bare `perf_counter` around GPU code.** CUDA is async; you
   measure enqueue time. Use `torch.utils.benchmark.Timer` (synchronizes) or
   `torch.profiler` with `ProfilerActivity.CUDA`. The project's own
   `_TimingStats` buckets have this flaw — treat them as *shares*, not as a
   CPU/GPU split.
2. **This machine has a ~±20% wall-time noise floor.** Anything below ~1.2x is
   **not measurable end-to-end here.** Size such changes with an isolated,
   interleaved microbenchmark and report the isolated number plus its share of
   wall time.
3. **Desktop contention is the dominant confound.** A background YouTube
   stream in Firefox (visible as `_ssl._SSLSocket.read` in profiles, and as
   ~80% GPU utilization with *zero* compute processes) produced apparent 1.2-1.4x
   "regressions" that did not exist. **Verify the machine is idle**
   (`nvidia-smi` util ~0%, no browser) before quoting wall time.
4. **Paired beats unpaired.** Ratios measured back-to-back in one process are
   robust to contention; comparisons across sessions/runtimes are not. A
   "torch 2.13 is 1.36x slower" conclusion was drawn and then **retracted** for
   exactly this reason.
5. **Discard the first run.** Cold CUDA context + autotune cost 66.5s vs 49.3s
   for an identical arm. A/B/B/A cancels *linear* drift only.
6. **Use `search_gpu` as a drift control** in paired runs: a CPU-side change
   cannot alter it, so if it moves >~1.0x the comparison is contaminated.
7. **cProfile inflates this workload ~2x** (349 vs 174 us/node) and its
   `cumtime` is invalid here (`_halving_stepwise` is a generator resumed
   re-entrantly, so pstats treats it as recursive). **Only self-time is
   usable**, and even that mis-sized `torch.cat` by 2x.

### Budget choice (settled; don't relitigate without new data)
`search_budget = 2048` is deliberate. Label agreement against a 2048 oracle:
512 -> 71.8%, 128 -> 62.2%, 32 -> 55.1% best-move agreement (mean |dvalue|
0.072 / 0.095 / 0.125). At budget 32 there are only ~2.5 evals per arm across
~15.7 arms: a confidently wrong teacher. Cheap budgets are faster and produce
bad labels.

---

## 7. Correctness gates (run these on any generation change)

**Label differential.** Generate 20 games at a fixed seed and diff the parquet
against a known-good baseline:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/generate_search_rollouts.py \
  --config config/imba_chess_exit_seeded_rollout.toml \
  --checkpoint artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt \
  --output-path /tmp/candidate.parquet \
  --max-games 20 --search-budget 2048 --concurrent-games 8 \
  --dtype float32 --sample-seed 42 --profile --profile-every-games 20
```

Compare: `best_arm_move_uci` agreement, `max |d best_arm_backed_value|`,
`max |d arm_log_prior|`, and **`arm_evals_spent` element-for-element** — that
last one proves the search trajectory is unchanged node-for-node, which is much
stronger than "the final labels happen to match".

- **Project gate**: >=99% move agreement and p99 |dvalue| <= 1e-3.
- **Prefer bit-identical.** Every optimization listed in §5 achieved exactly
  0.0 delta; if yours doesn't, understand precisely why before accepting it.
- Also check `waves=` and `evals=` are unchanged (identical work).

**Test suite**: `.venv/bin/pytest -q` — currently **335 passed**. Notable
guards: `tests/test_batched_projection.py` (padding cannot leak into a short
row), `tests/test_dense_attn_mask.py` (dense == flex), `tests/test_actor_worker.py`
(the eval path's independent move-mapping twin is cross-checked against
`_project_legal_logits_cozy`), `tests/test_search_stepwise.py` (starvation
fallback, 24 seeds).

---

## 8. Ranked next directions, with honest ceilings

Ceilings are `1/(1-share)`, i.e. what *perfectly eliminating* the cost would
give. Treat them as upper bounds, not estimates.

1. **Python -> Rust (PyO3) for the per-node hot path.** ~34% of wall time is
   the interpreter. Two sub-targets:
   - ~~`decode_project`'s movegen + vocab mapping~~ **DONE** (§12): one FFI
     crossing per node, returning all legal moves with their vocab ids in one
     call, via `cozy_bridge.project_legal_moves`. Measured **2.90 vs 11.63
     us/node (4.02x)** isolated, worth ~6.9% end-to-end — below the noise
     floor, so do not expect to see it in wall clock. The remaining
     `decode_project` cost is tensor construction and log_softmax, which the
     projector does not touch and which batching would target instead.
   - `search_bookkeeping` (~20% bucket): the tree/arena port. Note the older
     "1.86x" estimate for this was an artifact of the untimed-bucket bug;
     **true ceiling is ~1.27x** for the tree half alone. **This is now the
     largest remaining Python bucket and the next target.**
   - The byte-identical Python oracle argument still holds for the *remaining*
     sub-targets, and it is why the projection cutover kept an independent
     oracle in `tests/test_native_projection.py` rather than letting the Rust
     become its own only reference. Do the same for the tree port.
   - Cheaper first step: a **coarse forcing-set call** (`is_capture_cozy` +
     `gives_check` + `_forcing_index_set_tree`, ~9.9M crossings -> 1 per node).
2. **CUDA graphs / `torch.compile(mode="reduce-overhead")`** for the ~3,056
   small ops per wave. Directly targets the launch-overhead-dominated profile.
   Blocker: wave shapes vary every tick, so it would recompile/recapture
   constantly, and Inductor already crashed on the second shape in this
   codebase. **Needs a shape-bucketing scheme first** (pad wave sizes to
   powers of two?).
3. **Exploit the arena's memory headroom deliberately.** G=12 now completes at
   budget 2048 with peak allocated/reserved VRAM **6,187/6,696 MiB**, versus
   the former G=8 peak near the card limit. On the fixed 20-game gate it cut
   merged waves from **324 to 271**; wall time moved only ~1.1x, below this
   machine's trustworthy end-to-end threshold. The capacity is real; larger
   batches are an enabler for shape bucketing, not a claimed standalone win.
4. **CPU/GPU overlap (pipelining).** Ceiling only **~1.21x** (GPU is 17% busy;
   total device work is ~422 ms against ~2.4 s of host work). Requires two
   staggered game groups and a launch/collect split of the executor protocol.
   Large rewrite, modest payoff — deprioritized. *An earlier ~1.48x estimate
   was based on the enqueue-time buckets and is wrong.*
5. **Local corpus materialization**, removing HF/fsspec streaming (`_ssl` reads
   + pathlib churn) from the loop. Also unblocks raising `num_workers` (see §9).

### Rejected, with evidence — don't redo these
- **NestedTensor for suffix packing**: measured **2.6x slower** than the former
  pad+stack path (21.08 vs 8.03 us/node), with identical output. If a future
  design again has a true values+offsets jagged representation, prefer FBGEMM
  over NestedTensor. The current arena instead has indexed ancestor rows and
  gathers the final dense suffix directly; no jagged conversion remains.
- **Preallocated `new_zeros` + slice copy**: 1.30x at B=990 but *slower* at
  B=330; wave-size dependent, ~0.4% of wall.
- **TF32 (`set_float32_matmul_precision('high')`)**: `aten::mm` is only ~61 ms
  of ~400 ms GPU = **~2.5% of wall**. No version of that trade justifies
  risking label precision. fp32 is a deliberate label-quality choice (bf16
  gave only 90.4% move agreement, p99 |dv| 0.131).
- **FlashAttention for the search path**: FA rejects a non-null `attn_mask`,
  requires fp16/bf16, and FA3/CUTE-DSL are Hopper+ (this GPU is sm_86). Moot
  anyway now that attention is ~2% of time.
- **The "lambda sort" in the projection**: `sorted()` calls its key **n times,
  not n log n**. `key=ucis.__getitem__` and `sorted(zip(...))` both measured
  *slower* than the existing lambda. No headroom.
- **Lowering `search_budget`**: see §6.

---

## 9. Environment and operational constraints

- **GPU**: RTX 3070 Ti Laptop, **sm_86**, **7.66 GiB**. The arena cut peak
  allocated VRAM at G=8 from **6,340.9 to 4,165.7 MiB**; G=12 is now tested
  safe at **6,187 MiB allocated / 6,696 MiB reserved** for fp32, budget 2048.
  Keep `PYTORCH_ALLOC_CONF=expandable_segments:True`. Configuration defaults
  remain conservative because the same `[eval_vs_stockfish]` setting also
  controls actor-mode evaluation.
- **torch 2.13.0+cu130**. FBGEMM-GPU is no longer a runtime dependency:
  `_KVArena` stores each node's own K/V once and gathers dense ancestor
  suffixes by index.
- **`num_workers=0` in training is load-bearing, not an oversight.** It works
  around a sharding-alignment bug (`torch_iterable.py:42-48`:
  `num_shards = world_size * num_workers`, parquet-file-level sharding vs an
  unsharded generator) that produced **0.0% rollout coverage on all 4 shards**.
  Raising it silently re-breaks rollout training. Costs ~180 ms/step.
- **fp32 is deliberate** for label quality (see §8).
- Do not run GPU work while the machine's owner is gaming; and **verify the
  desktop is idle before quoting any wall-time number** (§6).

## 10. Known limitations inherited from the README

- Training is single-process (no end-to-end DDP launcher).
- No legal-move masking in the prediction head during training; legality is
  enforced at inference only.
- Prefix K/V caching is **per-turn**: rebuilt each model turn, no cross-turn
  reuse. (Cross-*game* batching now exists — that README line is stale.)
- The big model's value labels are raw game outcomes (noisy). Search-backed
  value distillation (ExIt Phase 1a) is not yet a net win.
- Checkpoints predating the placement-aware board encoding / 1,970-token vocab
  are incompatible.

## 11. Open question the next owner should settle first

**Evaluation power, not throughput, may be the real blocker.** The same
checkpoint scored 0.5850 and 0.5975 on the same 200-game protocol — a 0.0125
spread against SE=0.0297, i.e. ordinary sampling noise. But it means effects
below ~60 Elo currently cannot be measured at all. Fixing throughput without
fixing the ruler produces *faster nulls*. The minimal fix is more games
(800/arm -> SE ~0.015, ~2.5 h/arm), which is itself a generation-throughput
problem — so the two goals are coupled.

---

## 12b. Generation speedup: measured end to end (2026-08-21)

Three changes landed after the native binding cutover, all gated on
**bit-identical** 20-game rollout labels (324 waves / 428,016 evaluations
unchanged in every case):

1. `3192671` forcing detection fused into the native projector -- the scan
   re-walked the (board, moves) pair projection had just walked.
2. `0f0a02b` `push_and_classify` -- edge push and terminal detection in one
   native call, replacing ~15.5 `Board.pieces()` FFI crossings per node.
3. `292277b` wave row indices derived once per wave instead of once per layer
   -- `nonzero` was the largest single torch op in a rollout profile (1.019s /
   2,608 calls) and each call was also a device->host sync.
4. `0cb1067` `PositionEval` carries the vocab id the projector already
   computed, so `extend()` stops re-deriving it from the UCI string (270k
   hash+lookup per 4-game run); and `encode_cozy` moves to Rust -- it was the
   largest Python function left at 0.743s self, one call per node with ~10 FFI
   crossings each.

**Paired A/B/B/A, A = `d4d1a3e`, B = `292277b`, same machine, same session**
(changes 1-3):

| bucket | before | after | ratio |
| --- | --- | --- | --- |
| `search_gpu` | 16.1s | 11.1s | **1.46** |
| `search_bookkeeping` | 15.8s | 10.4s | **1.52** |
| `decode_project` | 8.3s | 11.8s | 0.70 |
| `decode_prep` | 5.8s | 6.5s | 0.88 |
| `root_eval` (control) | 0.9s | 0.9s | 1.00 |
| **total** | **47.0s** | **40.8s** | **1.152** |

Within-arm spread was 1.4s (A) and 0.2s (B) against a 6.2s difference, and the
control bucket came in at exactly 1.00, so **this one is resolved** -- unlike
the single-shot comparisons, which were repeatedly contaminated by drift of
0.66x-1.18x in `search_gpu`.

**`decode_project` going 0.70 is expected and is partly an artifact.** Forcing
detection genuinely moved into it (change 1). But change 3 also removed syncs
from the layer loop, so the stall moved to the next sync point -- the `.cpu()`
inside `consume_decode_result`, which `decode_project` times. The buckets are
enqueue-timed (section 6, rule 1), so moving syncs reshuffles attribution
without changing work. Read the total, not the split.

### Change 4 measured separately, and only at bucket level

**Paired A/B/B/A, A = `292277b`, B = `0cb1067`:**

| bucket | before | after | ratio |
| --- | --- | --- | --- |
| `decode_prep` | 11.2s | 8.9s | **1.264** |
| `search_bookkeeping` | 9.9s | 10.3s | 0.97 |
| `decode_project` | 23.9s | 24.8s | 0.96 |
| `search_gpu` | 19.7s | 20.4s | 0.97 |
| total | 67.0s | 66.8s | 1.004 |

**Read only the `decode_prep` row.** That session ran at ~67s where the same
code had measured ~41s an hour earlier -- roughly 60% slower machine-wide, with
no compute process on the GPU and `nvidia-smi` reporting 82% "utilization",
the phantom-load signature in section 6 rule 3. Everything inflated together,
so the total is unresolvable and the other buckets sit inside the drift band.

`decode_prep` is where `encode_cozy` lives, and its arm ranges do not overlap
(A 10.8/11.7 vs B 9.1/8.7), so **1.26x there is real** -- about 2.3s, which is
~5.6% of a healthy ~41s run but only 3.4% of that session's 67s.

The `extend` vocab-id half would surface in `search_bookkeeping` and did not
(0.97). It was sized at ~1-2% beforehand, i.e. below what any session here has
resolved. It is kept because it is free and provably identical, not because it
was measured.

**Do not compare totals across sessions.** Observed spread for materially the
same code across this work: 40s to 74s. Only same-session paired ratios mean
anything, and even those need the control buckets checked.

### Not done: batching the per-group decode loop

Padding every prefix to `maxP` and masking, so the whole per-group loop
collapses into one call, is **not bit-identical**. It changes the softmax and
output-einsum reduction lengths; a probe measured weights and outputs differing
by ~1e-7 at prefix lengths 37, 200 and 511. The search is a chain of argmax
comparisons over 2,048 nodes and 8 layers, so that can flip a near-tie and
change the tree.

Only the exact half was taken (change 3). Doing the rest requires an explicit
decision to retire the exact-label gate for this path and replace it with a
label-quality gate (agreement against a 2048 oracle plus a strength run). Do
not do it silently.

---

## 12. Owned native binding: complete (2026-08-21)

Stage 1 of the native-binding plan is **done and gated**. Production projects
legal moves through Rust, the third-party binding is gone, and both the exact
label gate and the speed gate passed. What remains is optional follow-on work,
listed at the end.

### Repository state

Everything below is pushed; nothing is left uncommitted.

- Feature branch `feat/imba-chess-native`, worktree
  `/home/vigi99/CodeDir/imba-chess/.worktrees/imba-chess-native`
  - `dfc16f8` perf: project legal moves in Rust
  - `d573821` test: validate native move projection
  - `91e1200` Merge main (seeded-stream generation + success-path hard exit)
  - `922b7c0` perf: use owned native chess projection
  - `83054fa` docs: refresh the native-binding resume checkpoint
  - `0ee7eac` fix: bound requires-python to PyO3's real ceiling
  - `6489040` fix: repair the scripts the projection cutover broke
- `main` (also pushed):
  - `cd32640` feat: spread rollout coverage with sparse game sampling
  - `a0ff7f0` fix: point the nightly rollout at a seed-pinned config
  - `590838a` fix: hard-exit on success, not only on crash

The main checkout no longer carries uncommitted generation work; the only
untracked file left there is `scripts/bench_decode_wave.py`, deliberately kept
out of git as a self-declared throwaway probe.

### What is now true of the code

`cozy_bridge.project_legal_moves(board, vocab)` is the **single** production
projection path. Search wave consumption, the per-node logit gather, and the
actor worker all route through it, and everything below the call is Rust:
movegen, castling normalization, vocabulary lookup, and the canonical UCI sort
in one FFI crossing. One `MoveProjector` per vocabulary is cached in a
`WeakKeyDictionary`, because building one walks the whole ~1,970-entry label
space and must not happen per node.

`cozy-chess-py` is gone -- from `pyproject.toml`, `uv.lock`, and the venv.
`grep -rn cozy_chess src/ tests/ scripts/` returns nothing. Deleted with it:
`_CASTLE_RAW_TO_UCI`, `_MOVE_ID_MEMO`, `_cozy_move_id_and_uci`,
`_legal_moves_ids_ucis`, `actor_worker._legal_vocab_projection`, and
`tests/test_native_binding_parity.py`.

### Evidence in hand

- **Label and work equality: PASS.** Pre- and post-cutover 20-game rollouts on
  a common stream produce **bit-identical** output across all 19 parquet
  columns, including every array column (`best_arm_move_uci`,
  `best_arm_backed_value` at max |delta| exactly 0.0, `arm_evals_spent`,
  `arm_log_prior`, `arm_move_uci`, `arm_backed_value`). Identical work too:
  209 rows / 20 games, **324 waves, 428,016 evaluations** on both arms.
  Artifacts: `artifacts/rollouts/{pre,post}_native_seeded.parquet`.
- **Speed gate: PASS.** Native **3.17 us/node** via the production entry point
  vs **12.18** for the reconstructed retired path -- **3.85x** against a
  `<= 5.56` requirement, reproducible across four runs (3.16-3.34).
- Projection differential: **exact** on 206 positions against both a full
  static vocabulary and a restricted 1-in-7 slice with non-matching ids.
- Suites: full **755 passed**, native package **105 passed**. `cargo fmt`,
  `clippy -D warnings` clean. Ruff at the 56-error baseline exactly.
- `decode_project` fell **13.5s -> 9.1s** on identical work = 10.3 us/node,
  against 9.0 predicted from the isolated benchmark. `search_gpu` was flat
  (18.7s both arms), which is what makes that delta readable.

**Do not claim an end-to-end speedup.** Total wall time moved 54.5s -> 50.9s
(1.07x), below this machine's noise floor. The bucket delta and the isolated
benchmark are the evidence; wall clock is not.

### The label gate, and the baseline that had to be thrown away

The first pre-cutover baseline was **invalid** and was deleted. It predated the
change threading `shuffle_train_month_files_on_start` /
`train_month_shuffle_seed` / `train_shuffle_buffer_size` into generation --
previously hardcoded to the LichessDataset defaults -- so it streamed a
different set of games: **169 rows against 209** for the identical command
afterwards. Comparing it post-cutover would have shown differences that had
nothing to do with the projector.

The gate that actually ran generated both arms on a common stream: the
pre-cutover arm from merge commit `91e1200` in a temporary worktree with
`cozy-chess-py` reinstalled, the post-cutover arm from the cutover commit,
identical flags. **If you ever need to repeat this, that is the shape** -- an
arm built from a commit, not from a saved parquet, because a saved parquet
silently ages out the moment the stream changes.

### Optional follow-on work

- ~~A read-only review~~ **DONE.** One finding, fixed in `15dcbc7`: the
  geometric castling normalization could collide a plain king step and a
  castle onto one vocabulary token on Chess960 boards whose king does not
  start on the e-file, silently returning a duplicate id/UCI pair.
  `project()` now refuses instead. No other correctness findings; the
  reviewer independently traced every `project_legal_moves` call site for the
  3- to 4-tuple signature change and confirmed sort-stability parity with the
  Python oracle.
- `scripts/bench_move_id_micro.py` benchmarked only the deleted per-move
  lookup, in both arms. It now exits non-zero with a pointer instead of
  appearing to run, and is a reasonable candidate for deletion.
- **Next perf target: `search_bookkeeping`**, now the largest Python bucket at
  15.9s / 31.3%. Ceiling is ~1.27x for the tree half alone (section 8). Keep
  an independent Python oracle for it, exactly as the projection port did.

### Resolved: Python version policy

`requires-python` is now bounded `>=3.10,<3.14` in both the root and native
manifests, because the crate builds a version-specific extension (no abi3) and
PyO3 0.23 supports through 3.13. This workstation's default interpreter is
3.14.7, so an unbounded declaration meant a fresh `uv sync` picked it and hit a
compiler error instead of a resolution error. uv now refuses up front:

```text
error: The requested interpreter resolved to Python 3.15.0rc1, which is
incompatible with the project's Python requirement: `>=3.10, <3.14`
```

Raise the bound only together with a PyO3 upgrade or a stable-ABI feature.

### Resolved: the generator hung after every successful run

Fixed in `590838a`. Both `generate_search_rollouts.py` and
`eval_vs_stockfish.py` escaped CPython's shutdown via `os._exit` on the
exception path only, so a run that completed normally still went through
ordinary finalization. A real 20-game rollout finished, wrote its parquet and
sidecar, printed its summary, then held 4 GB of GPU for 11 minutes.

**The cause was not the AsyncCompile pool named in the original writeup** --
that path is no longer reached from here (see `cd32640`). A faulthandler dump
of the stuck shutdown showed:

```text
Fatal Python error: PyGILState_Release: thread state 0x... must be current when releasing
Python runtime state: finalizing
```

A native thread touching the GIL during finalization. Depending on the race it
either parks forever or aborts -- meaning a *successful* run could also exit
non-zero and be misread as a failure by the nightly runner. There is no single
joinable thread to daemonize, so both wrappers now take the same unconditional
`os._exit` on success. Regression tests drive it through a real subprocess
boundary in `tests/test_rollout_script_hard_exit.py` and its
`eval_vs_stockfish` mirror.

Operationally: run that script with `PYTHONUNBUFFERED=1` and redirect to a
file. Piping it to `tail` hides all progress until exit.

### Corrections to the written plan (do not re-derive these)

- **The plan's Chess960 fixture is unreachable as written.** It puts the king
  on b1, where `b1c1` is *both* a normal king step and the normalized long
  castle, so one token covers two legal moves and the asserted output can
  never hold. The committed test uses rooks on b1/h1 with the king on e1,
  whose raw castle `e1b1` is absent from the old four-entry castle table and
  therefore still exercises the board-derived rule -- without the collision.
  This edge is harmless in standard chess, where a king can never legally step
  from e1 to c1 or g1 -- but `project()` no longer *tolerates* it either: since
  `15dcbc7` a token covering two legal moves raises rather than returning a
  duplicate. The lesson worth keeping: a from-to token space cannot express
  Chess960 castling unambiguously, so normalizing into one is a standard-chess
  convention, not a general one.
- **`tests.test_cozy_bridge._random_boards(n_games, ...)` takes a game count,
  not a position count.** Generate 200 games and stride the resulting position
  list to 200 well-spread positions.
- **Post-cutover, an actor-worker test comparing against
  `_project_legal_logits_cozy` is vacuous** -- both sides route through the
  same bridge call. It now compares against
  `tests.test_native_projection._python_project`, an independent oracle that
  re-derives castling from square geometry. Keep that independence: the Rust
  implementation must never become its own only reference.
- **`scripts/bench_python_opts_paired.sh` can no longer reconstruct its A
  arm** and now fails loudly rather than silently reporting the `encode_cozy`
  half as if it covered both changes.
- **Deleting a production helper silently breaks `scripts/`.** Nothing in the
  test suite imports those scripts, so a fully green suite said nothing about
  three of them (`bench_project_decompose.py`, `bench_move_id_micro.py`,
  `probe_project_phases.py`) being broken by the removal of
  `_cozy_move_id_and_uci`. After any similar deletion, import-check every
  script, not just the suite.

### Environment commands

`cozy-chess-py` is uninstalled; do not add it back except in the temporary
`91e1200` worktree used for the pre-cutover arm. PyO3 0.23 supports through
Python 3.13, so keep pinning 3.12.12:

```bash
cd /home/vigi99/CodeDir/imba-chess/.worktrees/imba-chess-native
uv sync --extra dev \
  --python /home/vigi99/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
```

`uv sync` will **not** rebuild the native extension after a Rust edit -- it
caches the path dependency by version. Force it:

```bash
uv sync --extra dev --python "$PWD/.venv/bin/python" \
  --reinstall-package imba-chess-native
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --manifest-path native/imba_chess_native/Cargo.toml -- -D warnings
```

The native package's 105 tests are **not** collected by a bare `pytest -q`;
run `native/imba_chess_native/tests/` explicitly.

### Latest strength gate

The post-arena 200-game SF2200/budget-2048 run completed 200/200 games at
**92 W / 54 D / 54 L, score 0.5950**, with 100% legal-move coverage. This is
consistent with the previous 0.5850 and 0.5975 baselines; no strength
regression was observed. Output:
`artifacts/eval/best_hr10_checkpoint_23_hr10-0.9564_sf2200_value_search_halving_post_arena200.json`.
