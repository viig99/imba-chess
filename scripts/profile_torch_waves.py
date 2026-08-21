"""torch.profiler instrumentation for the rollout decode loop.

Why this exists: every timing bucket in this project uses a bare
`time.perf_counter()` delta with no `torch.cuda.synchronize()`. Per the PyTorch
benchmarking guide that measures how long the CPU took to *enqueue* CUDA work,
not how long the GPU took to run it. That is why `search_gpu` (which wraps only
forward_decode_grouped) looked like 27-31%, while a further 12.5 us/node of
"CPU" time showed up in `d2h` -- the first real sync -- and why hoisting work
above that sync moved 6.7 us/node out of `d2h` and straight into `kv_path`
instead of saving it. The device time is real; the buckets just cannot see where
it is.

This gives the honest split: CUDA kernel time vs CPU time, per op, with the
launch-overhead question answered directly (cuda time vs cpu time on the same
op), plus a chrome trace.

Run:
  .venv/bin/python scripts/profile_torch_waves.py            # summary tables
  .venv/bin/python scripts/profile_torch_waves.py --trace    # also write trace
"""

from __future__ import annotations

import runpy
import sys

from torch.profiler import ProfilerActivity, profile, record_function, schedule

from imba_chess.eval import merged_executors as me

WANT_TRACE = "--trace" in sys.argv
GAMES = "3"

_PROF: object | None = None


def _wrap_executors() -> None:
    """Tag the two GPU-bearing executors and step the profiler per decode wave."""
    orig_decode = me._make_decode_wave_executor
    orig_root = me._make_root_eval_executor

    def decode_factory(**kwargs):
        inner = orig_decode(**kwargs)

        def wrapped(payloads):
            with record_function("decode_wave"):
                out = inner(payloads)
            if _PROF is not None:
                _PROF.step()
            return out

        return wrapped

    def root_factory(**kwargs):
        inner = orig_root(**kwargs)

        def wrapped(payloads):
            with record_function("root_eval"):
                return inner(payloads)

        return wrapped

    me._make_decode_wave_executor = decode_factory
    me._make_root_eval_executor = root_factory


def main() -> None:
    global _PROF
    _wrap_executors()

    # skip_first swallows CUDA context init and the first autotune waves, which
    # the profiling guide warns not to read as workload cost.
    sched = schedule(skip_first=24, wait=0, warmup=2, active=12, repeat=1)
    tables: list[str] = []

    def on_ready(p):
        tables.append(
            "=== by CUDA time (device work) ===\n"
            + p.key_averages().table(sort_by="cuda_time_total", row_limit=18)
        )
        tables.append(
            "=== by self CPU time (host work, leaves) ===\n"
            + p.key_averages().table(sort_by="self_cpu_time_total", row_limit=18)
        )
        if WANT_TRACE:
            p.export_chrome_trace("/tmp/rollout_trace.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=on_ready,
        record_shapes=True,
    ) as prof:
        _PROF = prof
        sys.argv = [
            "generate_search_rollouts.py",
            "--config", "config/imba_chess_exit_seeded_rollout.toml",
            "--checkpoint",
            "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
            "--output-path", "/tmp/torchprof.parquet",
            "--max-games", GAMES, "--search-budget", "2048",
            "--concurrent-games", "6",
            "--dtype", "float32", "--sample-seed", "42",
        ]
        try:
            runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
        except SystemExit:
            pass

    for t in tables:
        print(t)
        print()

    if WANT_TRACE:
        print("chrome trace: /tmp/rollout_trace.json")


if __name__ == "__main__":
    main()
