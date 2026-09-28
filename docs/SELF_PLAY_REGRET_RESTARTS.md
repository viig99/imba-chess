# Observed-trajectory regret restarts

Add this section to a copy of a streaming self-play configuration, then start a
**new run directory** with `scripts/run_self_play.py`:

```toml
[regret]
capacity = 256
temperature = 0.1
ema_alpha = 0.5
```

The configuration must also contain `[streaming]`. Omitting `[regret]` preserves
the historical configuration and sampler identities, consumer-state format,
four-bucket launch order, and reconciliation path. Enabling regret or changing
its settings is incompatible with an existing run. Context and game limits stay
in force; the sampler also checks the runtime context limit on resume.

## Launch allocation

Each deterministic shuffled cycle requests these five buckets once:

| ID | Starting position | Share of new launches |
|---|---|---:|
| 0 | Initial board | 20% |
| 1 | Human prefix, plies 1–30 | 20% |
| 2 | Human prefix, plies 31–70 | 20% |
| 3 | Human prefix, plies 71–120 | 20% |
| 4 | Regret-prioritized full-history restart | 20% |

If bucket 4 has no usable positive-priority entry, it substitutes an ordinary
bucket. A separate persisted shuffled four-bucket cycle balances these fallback
launches. Allocation is by **new game launches**, not positions, inference calls,
or GPU time. Crash retries reuse their original game ID, full prefix, and
exploration seed; they do not advance either curriculum cycle.

## Priority and storage

For every searched position of a completed training continuation, select raw
`qvalues[legal_ids.index(move_id)]`. With the actual terminal outcome expressed
from that position's side to move, compute squared error `(Q - z)^2`. A single
backward pass computes each position's mean error over its remaining continuation.
There is no substitution with normalized Q, raw root value, or `search_wdl`.
Missing, misaligned, nonfinite, or out-of-range selected Q is a protocol error.
Interrupted, limited, and errored continuations supply no regret observation.

An ordinary completion can admit one position: the highest-regret eligible
history not already buffered, with earliest-ply tie breaking. Eligible histories
are nonempty, legal, nonterminal, and leave room under the existing game/context
limits. The buffer grows to capacity; subsequently a candidate must strictly
exceed the lowest stored priority. Oldest admission breaks eviction ties.
Histories, rather than board FENs, determine duplicates. Every entry keeps its
full prefix, original training-source and corpus identities, parent game/ply,
priority, monotonic admission identity, and refresh count.

Sampling uses `P(s) ∝ R(s) ** (1 / temperature)` with stable log-space weights.
The default temperature 0.1 gives **R¹⁰**; zero-priority entries have no mass.
A completed regret restart refreshes only its starting admission:
`priority = (1 - ema_alpha) * old + ema_alpha * new_R_0`. It never admits
descendants. There is no age expiry or time decay. Unusable entries may be
retired. A result from an evicted admission cannot update a later readmission of
the same history.

The buffer lives inside `stream/consumer.json`. Newly durable replay games are
read in launch-sequence order at collection startup, after completion-triggered
flushes, and after the final flush. Observations and removal of acknowledged
pending launches share one atomic save. Unpublished launches retry after a
crash; published but unacknowledged games are recovered from replay; acknowledged
games are not processed again. Buffer histories survive replay shard garbage
collection independently.

Trajectory JSON gains optional `requested_bucket`, `starting_bucket`,
`restart_parent` (history/admission identity and parent game/ply), and
`exploration_seed` fields. Replay shard columns and checkpoint weights do not
change. The existing training dataset masks inherited histories and supervises
only the new continuation. Evaluation and monitoring do not feed this buffer.

## Metrics and verification

The sampler reports cumulative unique-launch counts in `stream_requested` and
`stream_launched`, cumulative `stream_fallbacks`, and buffer size, positive-entry
count, admission/replacement/refresh/stale-refresh/retirement counts, and priority
min/mean/median/p95/max under `regret`.

Collection reports `starting_buckets`: launch attempts (including crash retries),
fallback attempts, completion and limit rates, mean/max continuation lengths,
completed continuation lengths, searched positions, neural evaluations,
simulations, terminal hits, depth cutoffs, and search seconds. Position and neural
evaluation shares show actual compute allocation. `distinct_restarted_positions`
counts distinct complete histories attempted during that collection call.
Per-bucket search seconds sum coroutine search latency, including scheduler wait;
they are not exclusive GPU occupancy. The existing overall runtime and throughput
measurements remain available. On resume, completed-game counts/lengths can be
reconstructed from replay; search work/time describes the current process.

Focused verification:

```bash
.venv/bin/python -m pytest -q \
  tests/test_self_play_regret.py tests/test_self_play_streaming.py \
  tests/test_self_play.py tests/test_self_play_schedule.py \
  tests/test_search_config_migration.py tests/test_policy_surprise.py
```

Tests cover hand-computed regret, identities and limits, sampling, eviction,
fallback cycles, replay publication/acknowledgement crash boundaries, garbage
collection, and ordinary completion → admission → restart → training → refresh
→ exact trainer/sampler resume. Identical completed trajectories retain identical
targets, masks, and losses.

This implements observed-position prioritization inspired by
[RGSC §3.1](https://arxiv.org/html/2602.20809v1#S3.SS1). It does not implement the
paper's learned regret/ranking networks or selection of unplayed search-tree
nodes. It adds no neural inference, networks, losses, search changes, or Stockfish
supervision. Sustained training and chess-strength comparisons remain separate
experiments.
