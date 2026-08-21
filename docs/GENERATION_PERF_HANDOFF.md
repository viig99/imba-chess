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
| `src/imba_chess/eval/cozy_bridge.py` | python-chess <-> cozy-chess conversion, native terminal detection |
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
| `f957f0c` | memoized cozy move -> (vocab id, UCI) lookup | 311 -> 154 ns/move (**2.02x**) |
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
   - `decode_project`'s movegen + vocab mapping (~26-31% bucket): the aim is
     **one FFI crossing per node instead of ~26**, returning all legal moves
     with their vocab ids in one call. Today there are ~97 cozy-Board crossings
     and ~494 C-level calls per node.
   - `search_bookkeeping` (~20% bucket): the tree/arena port. Note the older
     "1.86x" estimate for this was an artifact of the untimed-bucket bug;
     **true ceiling is ~1.27x** for the tree half alone.
   - A byte-identical oracle exists *now* (the Python implementation), and it
     decays once the fused loop lands — this is the argument for doing it early.
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

## 12. Resume checkpoint: owned native binding (2026-08-21)

Work is intentionally paused after the first independently testable task.
No `MoveProjector`, application import migration, forcing port, or search port
has started.

### Repository state

- Feature branch: `feat/imba-chess-native`
- Worktree:
  `/home/vigi99/CodeDir/imba-chess/.worktrees/imba-chess-native`
- Task-1 checkpoint commit: `40ecabc` (`feat: add owned native chess binding`)
- Design:
  `docs/superpowers/specs/2026-08-21-imba-chess-native-design.md`
- Executable plan:
  `docs/superpowers/plans/2026-08-21-imba-chess-native.md`
- Main baseline commit `eb8df12` contains the user-approved Phase-1b training
  probes needed to restore the committed test suite to green.
- The main checkout still has unrelated, deliberately uncommitted generation
  work in `scripts/generate_search_rollouts.py`,
  `scripts/rollout_nightly_start.sh`, and `scripts/bench_decode_wave.py`.
  Do not stage or overwrite it from the feature worktree.

### Completed

1. Vendored the MIT-licensed `cosy-chess-py` 0.1.1 thin wrapper from exact
   commit `60f663ed4f4f8f95276453245bbc121314ad533f`.
2. Renamed the owned distribution/import module to `imba_chess_native`.
3. Added it as a side-by-side maturin path dependency; production still imports
   `cozy_chess` and therefore remains unchanged.
4. Ported the complete wrapper API/stub/tests rather than creating a partial
   second Board convention.
5. Added an old/new differential over six edge FENs plus 200 positions spread
   across 200 random reachable games.

Verification at the checkpoint:

- Clean rebased project baseline: **336 passed**
- Native wrapper + cross-binding parity: **302 passed**
- `cargo fmt --check`: pass
- `cargo check`: pass when `PYO3_PYTHON` points at the worktree Python

### Environment commands

Do not let `uv` choose the workstation's default Python 3.14 during the
side-by-side stage. `cozy-chess-py` and PyO3 0.23 support through Python 3.13.
The tested environment is CPython 3.12.12:

```bash
cd /home/vigi99/CodeDir/imba-chess/.worktrees/imba-chess-native
uv sync --extra dev \
  --python /home/vigi99/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12

PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo check --manifest-path native/imba_chess_native/Cargo.toml
```

The final owned package must resolve Python 3.14 policy before cutover: either
enable an appropriate PyO3 stable-ABI feature or declare the real upper Python
bound. Do not silently build against 3.14 with an unsupported interpreter.

### Important test correction

`tests.test_cozy_bridge._random_boards(n_games, ...)` takes a game count, not a
position count. A head slice covered only the first few trajectories; passing
all generated positions to pytest created thousands of redundant cases.
The checkpoint generates 200 games and strides the resulting position list to
exactly 200 well-spread test positions:

```python
boards = _random_boards(200, seed=731)
positions = boards[:: max(1, len(boards) // 200)][:200]
```

Keep the same discipline for Task 3's projector differential.

### Next exact step

Resume at **Task 2: Implement `MoveProjector`** in the plan:

1. Add failing native tests for canonical ordering, restricted mappings,
   independent vocabularies, invalid keys, standard castling, and Chess960
   castling.
2. Verify RED (`MoveProjector` absent).
3. Implement only `native/imba_chess_native/src/move_projector.rs`, register it,
   and update the stub.
4. Rebuild and run the native suite.
5. Continue to the isolated projector differential/benchmark.

The hard gate remains: native move generation + vocab mapping + canonical
sorting must be at most **5.56 us/node** (2x faster than the **11.12 us/node**
Python baseline). If it misses, stop before production cutover.

### Latest strength gate

The post-arena 200-game SF2200/budget-2048 run completed 200/200 games at
**92 W / 54 D / 54 L, score 0.5950**, with 100% legal-move coverage. This is
consistent with the previous 0.5850 and 0.5975 baselines; no strength
regression was observed. Output:
`artifacts/eval/best_hr10_checkpoint_23_hr10-0.9564_sf2200_value_search_halving_post_arena200.json`.
