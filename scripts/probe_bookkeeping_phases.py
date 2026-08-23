"""THROWAWAY: where does search_bookkeeping's 23.4 us/node actually go?

`scripts/bench_search_bookkeeping_decompose.py` measures 3.17 us/node across
forcing+push+term+order+heap+backup, but the real bucket is 10.0s / 428,016
node evals = 23.36 us/node. The synthetic harness therefore explains only
13.6% of the thing it decomposes, and porting all of it to Rust for free
would buy 1.044x -- not the 1.27x the handoff claims.

Two hypotheses, distinguished by this probe:
  (a) the generator's WAVE LOOP (frontier pops, `remaining[id(arm)]` dict
      churn, wave/EvalRequest tuple construction, per-node value assignment)
      is the missing ~20 us/node; or
  (b) the harness UNDER-measures the phases it does cover -- its own output
      reports `mean history len: 0.10`, but production runs to max_depth=8 so
      push_and_classify's repetition scan should see ~2-4, and it sampled only
      160 opening positions.

Method: wrap the real functions during a real rollout (same approach as
probe_project_phases.py -- in situ, never synthetic, because that is the only
thing that made the decode_project numbers trustworthy). `_push_children`
covers order+forcing+push+term+heappush; `_backed_stm` covers backup. Whatever
`search_bookkeeping` has left after subtracting those is the wave loop.

Shares within one run are drift-immune (handoff section 6), so this is valid
even on a contended machine. Do NOT quote its absolute seconds.

Run: .venv/bin/python scripts/probe_bookkeeping_phases.py
"""

from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imba_chess.eval import search  # noqa: E402

T = {"bookkeeping": 0.0, "push_children": 0.0, "backed_stm": 0.0, "prior_order": 0.0}
N = {"push_children": 0, "backed_stm": 0, "prior_order": 0, "waves": 0}


def _wrap(name, module, attr):
    original = getattr(module, attr)

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            T[name] += time.perf_counter() - t0
            N[name] += 1

    setattr(module, attr, timed)


def _timed_search(original):
    """Wrap _halving_stepwise so every generator advance is timed.

    That total IS the search_bookkeeping bucket by construction -- the script's
    own _timed_advance measures exactly the same thing -- so this probe does not
    depend on reaching the generator's private stats object.
    """

    def patched(**kwargs):
        inner = original(**kwargs)

        def proxy():
            t0 = time.perf_counter()
            try:
                request = inner.send(None)
            except StopIteration as stop:
                T["bookkeeping"] += time.perf_counter() - t0
                return stop.value
            T["bookkeeping"] += time.perf_counter() - t0
            while True:
                sent = yield request
                N["waves"] += 1
                t0 = time.perf_counter()
                try:
                    request = inner.send(sent)
                except StopIteration as stop:
                    T["bookkeeping"] += time.perf_counter() - t0
                    return stop.value
                T["bookkeeping"] += time.perf_counter() - t0

        return proxy()

    return patched


def _install() -> None:
    _wrap("push_children", search, "_push_children")
    _wrap("backed_stm", search, "_backed_stm")
    _wrap("prior_order", search, "_prior_order")
    search._halving_stepwise = _timed_search(search._halving_stepwise)
    # The generator hard-exits via os._exit on SUCCESS (590838a), which would
    # skip this probe's report entirely. Convert it to a normal unwind.
    os._exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))


def _report() -> None:
    bucket = T["bookkeeping"]
    nodes = N["push_children"]
    print("\n" + "=" * 70)
    print("search_bookkeeping in-situ decomposition")
    print("=" * 70)
    if not nodes or not bucket:
        print(f"nothing recorded (nodes={nodes}, bucket={bucket})")
        return
    acc = T["push_children"] + T["backed_stm"]
    print(f"nodes expanded : {nodes:,}   waves : {N['waves']:,}")
    print(f"\n  search_bookkeeping (all generator advances) : {bucket:8.3f}s  "
          f"{bucket / nodes * 1e6:7.2f} us/node")
    print(f"  _push_children  (order+forcing+push+heappush) : {T['push_children']:8.3f}s  "
          f"{T['push_children'] / nodes * 1e6:7.2f} us/node  {T['push_children'] / bucket:6.1%}")
    print(f"  _backed_stm     (negamax backup)              : {T['backed_stm']:8.3f}s  "
          f"{T['backed_stm'] / nodes * 1e6:7.2f} us/node  {T['backed_stm'] / bucket:6.1%}")
    print(f"    of which _prior_order (inside push_children): {T['prior_order']:8.3f}s  "
          f"{T['prior_order'] / nodes * 1e6:7.2f} us/node  (not additive)")
    print(f"  WAVE LOOP remainder                           : {bucket - acc:8.3f}s  "
          f"{(bucket - acc) / nodes * 1e6:7.2f} us/node  {(bucket - acc) / bucket:6.1%}")
    print("\n  Synthetic harness measured 3.17 us/node across these phases.")
    print("  VERDICT: (a) WAVE LOOP dominates -- port/optimise the loop, not the phases"
          if (bucket - acc) > acc
          else "  VERDICT: (b) harness UNDER-measured -- the phases really are the cost")


def main() -> None:
    _install()
    sys.argv = [
        "generate_search_rollouts.py",
        "--config", "config/imba_chess_exit_seeded_rollout.toml",
        "--checkpoint", "artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt",
        "--output-path", "/tmp/bookkeeping_probe.parquet",
        "--max-games", os.environ.get("PROBE_GAMES", "4"),
        "--search-budget", os.environ.get("PROBE_BUDGET", "2048"),
        "--concurrent-games", os.environ.get("PROBE_CONCURRENT", "8"),
        "--dtype", "float32", "--sample-seed", "42",
        "--profile", "--profile-every-games", os.environ.get("PROBE_GAMES", "4"),
    ]
    try:
        runpy.run_path("scripts/generate_search_rollouts.py", run_name="__main__")
    except SystemExit:
        pass
    finally:
        _report()


if __name__ == "__main__":
    main()
