# Reusing the training input stream for self-play

The existing loader has ample measured input capacity for self-play. Reuse its
streaming, filters, PGN parser, and background-worker patterns. The integration
needs controlled starting-position selection and resumable prefetching, rather
than a new high-throughput parser.

## Measurement

`scripts/bench_self_play_input.py` calls the existing
`LichessDataset.filtered_shuffled_rows()` and `stream_from_rows()` paths. It
retains the v4 training month window, column projection, 8,192-row Parquet batch,
10,000-row shuffle buffer, seed 42, and the production Elo/time-control/BOT
filters. Stockfish comment parsing is disabled because human evals are not
self-play targets. Parsing still includes full-game board-state encoding.

Single-process measurements on this host, without GPU work:

| Input | First filtered row | Full-game parsing |
| --- | ---: | ---: |
| Existing local training corpus, 1,024 rows | 0.037 seconds | 545–566 games/s over three passes |
| Live monthly stream, 1,024 rows | 23.04 seconds | 500–530 games/s over three passes |
| Live monthly stream, 16,384 rows | 19.13 seconds | 440 games/s over one pass |

Both samples produced 1,023 accepted games and 80,411 plies. Parsing passes reuse
the captured rows, so their times exclude network fetching. These are capacity
measurements, not a concurrent collector benchmark or a completed seed sampler.
Timers start after Python imports. The remote tests reuse normal caches, so
these are not guaranteed cold-cache network measurements.

The larger run crossed multiple input batches: it fetched 16,384 distinct source
rows in 33.15 seconds including startup, then parsed 16,369 accepted games
(1,265,428 plies) in 37.19 seconds. Combined sequential fetch-plus-parse capacity
was 233 accepted games/s including startup. Of these games, 8,467 extended past
ply 71; this is length coverage, not a complete late-bucket eligibility check.
Peak process RSS was 1.27 GiB including imported libraries, remote buffers and
the benchmark's retained 16,384 rows, not the size of a proposed prefix queue.

The recent collection metrics (94 records at iterations 111 and later) report a
median 1,649 completed games/hour and maximum 1,903/hour, roughly 0.46–0.53/s.
This comparison precedes rejection of monitoring sources and final seed
eligibility checks, but leaves substantial input capacity headroom.

Raw results are in `artifacts/self_play_validation/input_stream_2026-09-17/`.
The initial sandbox network attempt failed DNS; the live measurements ran with
network access. No production training or evaluation configuration was changed.
Both remote commands were bounded with `timeout 180`. Both wrote
all completed measurements but remained alive until that timeout (exit 124).
Thus completed throughput measurements are not evidence of clean remote-stream
shutdown. A persistent producer avoids repeated setup/teardown during training,
but the integration must also test bounded shutdown with outstanding stream work.

## Reuse decisions

- Use `LichessDataset` for month selection, file-level worker sharding, column
  projection, filtering before parsing, buffered row shuffling, and PGN parsing.
  Full parsing is already fast enough; specialize prefix parsing only if an
  integrated measurement shows a need.
- Keep one producer alive across collection/training cycles. Reuse the
  persistent-worker and bounded-prefetch pattern from the training DataLoader.
  Start with one producer; the training setting of 12 workers serves a much
  higher consumption rate and need not be copied mechanically.
- Put a bounded queue of prepared prefixes between that producer and the
  collector. Warm the queue before launching games. A target of 512 ready starts
  would cover about 15–19 minutes at the observed total game-consumption rate,
  assuming all those entries are eligible for the requested mixture. Maintain
  availability per bucket so one missing bucket cannot be hidden by aggregate
  queue depth. This queue size is a proposed starting point, not a tuned result.
- Apply the explicit initial-board/human-prefix mixture after shared game
  filtering. Keep human results and Stockfish annotations out of self-play
  targets; preserve source exclusions for monitoring and validation/test.
- Persist source progress, shuffle/prefetch state, bucket queues and sampling
  RNG together with consumed starts. A normal DataLoader alone does not guarantee
  exact mid-stream restart, especially with prefetched but unconsumed work.
- If an on-disk cushion is needed, reuse the existing filtered-Parquet
  materialization pattern, with replenished chunks and recorded provenance.
  Do not turn that cushion into another permanently fixed seed manifest.

Before enabling the integration, measure collector queue wait and per-bucket
occupancy alongside games/hour, positions/hour and optimizer steps/hour. Verify
resume without source rewind or lost prefetched work. The measurements here do
not establish long-duration network reliability or CPU-contention effects during
GPU collection; those belong to that integrated check.

Earlier evidence agrees: `docs/superpowers/notes/2026-08-20-rl-throughput-bottleneck.md`
recorded 288 streamed games/s after a 30.5-second first-game delay. The existing
materializer also documents occasional long remote stalls, which is why input
prefetch should stay outside the collector's synchronous launch path.
