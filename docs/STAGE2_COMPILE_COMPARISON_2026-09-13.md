# Self-play weight-update compilation

Status: the deferred comparison ran after the continuation finished. Eager
completed; putting mask construction inside the full-size model graph failed
with Inductor `CantSplit` on a symbolic block-grid dimension (20:05 Toronto).
The small combined-graph result did not generalize to real batch sizes.
A follow-up keeps the existing compiled mask builder outside the compiled
model-plus-loss graph. The completed comparison is under `separate-mask-final/`.
Production promotion is withheld because four parameter-update elements exceeded
the predeclared stricter delta tolerance; the tolerance was not relaxed.
No HSTU architecture, precision, learning-rate, replay or collection change is
part of this comparison.

The one-shot user service `imba-stage2-compile-comparison-20260913.service`
waits for PID 991186 to exit and for the GPU to have no compute processes, then
runs eager and compiled sequentially. The current run's deadline is September 13
at 20:19 Toronto time. The benchmark cutoff is 21:45, ahead of scheduled 22:00
self-play; if there is no idle window with at least 20 minutes remaining, it
does not start. Source fingerprints are checked before execution. Logs, reports
and `deferred-status.json` are in
`artifacts/self_play_validation/training_compile_2026-09-13/`.

## Full-size results

RTX 3070 Ti Laptop, PyTorch 2.14.0+cu130, FP32, TF32 disabled, four CPU threads.
Both modes used the same actor-000014, replay, sampled games, 1,024-token budget,
StableAdamW settings and 8,197 exposures / 13 updates per throughput trial.

| Trial | Eager seconds | Compiled model/loss seconds |
|---|---:|---:|
| First throughput pass | 9.148 | 3.542 |
| Warmed 1 | 9.043 | 3.080 |
| Warmed 2 | 9.474 | 3.382 |

The first eager throughput pass captured one additional mask graph, so the two
subsequent passes are the warmed comparison: median 9.259 → 3.231 seconds,
approximately **2.87× faster**. Two warmed trials are a bounded estimate, not a
precise sustained-throughput claim. Compiled throughput passes captured no new
graphs. Peak allocated memory was approximately **2.764 → 1.541 GB**, a **44.2%**
reduction; reserved memory was **3.857 → 2.137 GB**. These measurements include
replay reconstruction and updates, and do not measure collection or chess strength.

Main correctness checks passed at absolute `2e-5`, relative `2e-4`: first clipped
gradients, all final model tensors, optimizer state, losses and gradient norms.
The stricter comparison of three-step changes from initialization found **four
elements** outside absolute `2e-6`, relative `2e-3`. Maximum difference:
`4.448e-6`; aggregate update relative L2 error: **0.0629%**. Outliers were in two
square-encoder QKV matrices, one square-encoder MLP matrix and one temporal output
matrix. This is small numerical drift, but the predefined gate did not pass;
it is not evidence of a chess regression or permission to silently relax the gate.

The final comparison's startup/correctness durations (5.07 s eager, 11.64 s
compiled) include CPU snapshots and reuse disk compiler caches; they are not
clean cold-compilation benchmarks. The earlier separate-mask attempt's first
compiled update took 30.12 s. No clean-cache startup claim is made.

The production trainer retains its eager callable. The compiled candidate and
full diagnostic states are retained for review; no actor was replaced. Numerical
failures are now saved in JSON while throughput measurement continues, so the
performance result cannot be mistaken for a passed correctness gate.

## Compilation boundary

On the installed PyTorch 2.14.0+cu130, a small CUDA probe successfully captured
**BlockMask construction, the entire model, and self-play loss together** with
`torch.compile(..., fullgraph=True, dynamic=True)`, and ran backward. This is
stronger than compiling just the FlexAttention call. AOTAutograd generates the
backward graph; `.backward()` remains the ordinary caller-side API.

The model includes the square encoder (still using SDPA), every temporal HSTU
layer (fused FlexAttention), the policy/value heads, and the loss/metric tensor
computations. Fusing temporal attention does not change its softmax, scale,
relative bias, gate or residual connection.

Replay reconstruction, sampling, finite-loss checks, gradient clipping,
StableAdamW, scheduling, logging and checkpoints remain outside. Leaving the
optimizer outside preserves the current update behavior for this comparison.
`fullgraph=True` rejects graph breaks; it does not mean one giant GPU kernel.

The candidate wraps `_training_loss` in
[trainer.py](../src/imba_chess/self_play/trainer.py), keeping the original model
object shared with inference and keeping checkpoint parameter names unchanged.
There is no new production attention-mode option. Production still uses the
eager callable until the full-size comparison passes.

The within-game positional embedding now supplies the known flattened length
as `repeat_interleave(output_size=...)`, avoiding an unnecessary inference of
that length from tensor data. Parameter shapes and embedding values are unchanged.

## Checks

- The full model-plus-loss CUDA graph with an externally constructed BlockMask
  matched eager losses, all parameter gradients (including relative biases), and
  three consecutive optimizer updates across three batch sizes. Fixed comparison
  tolerances: absolute `2e-5`, relative `2e-4`, FP32, dropout disabled.
- The combined graph including mask construction also passed that same
  three-update, changing-batch-size comparison. The regression lives in
  [test_training_compile.py](../tests/test_training_compile.py).
- Existing default suite: 2,267 passed, 18 extended tests deselected.

These are compiler/correctness checks on a small network, not measurements of
full-size training speed, memory savings or chess strength.

## Bounded full-size comparison

[bench_training_compile.py](../scripts/bench_training_compile.py) loads the same
checkpoint and read-only replay for both modes. It records hashes, batch game
IDs/shapes, precision and compiler counters. First it compares three identical
updates: first-step clipped gradients, all final model tensors, Adam state,
losses and the pre-clipping gradient norm. Numerical tolerances are fixed before
running. Parameter changes from initialization are additionally compared at
absolute `2e-6`, relative `2e-3`, so large initial weights cannot hide a wrong
small update. It then resets weights, optimizer and random seeds for three throughput
trials using the same 1,024-token microbatch budget and 8,192 requested supervised
exposures. Whole-game packing can overshoot that exposure budget.

The retained replay was verified: this seed produces 13 batches totaling 8,197
supervised positions, with 847–1,019 context tokens per batch. The exact game IDs
and shapes are saved in `artifacts/self_play_validation/training_compile_2026-09-13/batch-plan.json`.

Timing uses synchronized wall-clock boundaries and includes replay preparation.
Correctness snapshots and initial compilation are reported separately. The first
throughput trial may encounter additional shapes: graph counts identify this,
and subsequent trials establish warmed behavior. Allocated and reserved peak
CUDA memory are both recorded. No active actor/checkpoint is overwritten.

Run sequentially on an idle GPU from the repository root:

```bash
.venv/bin/python scripts/bench_training_compile.py \
  --config artifacts/self_play_validation/nightly_tuning_2026-09-13/tokens-1024.toml \
  --checkpoint artifacts/self_play/laptop-pilot/actor-000014.pt \
  --replay artifacts/self_play/laptop-pilot/replay \
  --output artifacts/self_play_validation/training_compile_2026-09-13 \
  --mode eager

.venv/bin/python scripts/bench_training_compile.py \
  --config artifacts/self_play_validation/nightly_tuning_2026-09-13/tokens-1024.toml \
  --checkpoint artifacts/self_play/laptop-pilot/actor-000014.pt \
  --replay artifacts/self_play/laptop-pilot/replay \
  --output artifacts/self_play_validation/training_compile_2026-09-13 \
  --mode compiled
```

The old tuning checkpoint `actor-000002.pt` has been removed; this comparison
uses the retained `actor-000014.pt` for both sides. Historical timings are not
substituted for a new baseline.

Promotion requires full-size numerical agreement, acceptable shape/recompile
behavior, and measured throughput/memory. Then CUDA trainers can retain only the
full-graph compiled callable, with eager kept solely as the benchmark reference
and CPU test path. Cold startup should be assessed against repeated learning
phases: updated weight values do not by themselves require recompilation.
