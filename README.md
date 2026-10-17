# imba-chess

Chess sequence modeling, Gumbel self-play learning, and model evaluation against Stockfish. The HSTU model consumes complete board/move histories; native chess primitives handle legal moves, board encoding and terminal rules.

## Current workflows

| Task | Entrypoint |
|---|---|
| Supervised pretraining / fine-tuning | `scripts/train.py` |
| Established halving evaluation against Stockfish | `scripts/eval_vs_stockfish.py` |
| Halving model A versus model B | `scripts/match_two_checkpoints.py` |
| Prepare the held-out monitor-opening manifest | `scripts/materialize_corpus.py`, `scripts/prepare_self_play_seeds.py` |
| Train from stage-2 replay | `scripts/train_self_play.py` |
| Alternate streamed collection, training and paired evaluation | `scripts/run_self_play.py` |
| Gumbel model pairs or Gumbel versus Stockfish | `scripts/eval_self_play.py` |
| Live TensorBoard metrics | `scripts/monitor_self_play.py` |

The offline halving-target generator and its storage format have been retired. Both the halving search algorithm and the Stockfish evaluation pipeline remain supported.

## Model and learning

Games use a static 1,970-token UCI vocabulary, placement-aware board encoding, BOS, and pre-move board/previous-move events. Complete histories are packed into jagged batches. The shared trunk has policy, WDL value and moves-left heads.

New models use a flattened square readout: two square-attention blocks, per-square
normalization, flatten 64×64 features, and a 4096→model-width projection. There is
one model architecture, with no pooling switch or runtime checkpoint conversion.
For a training warm start use `--init-weights`; `--resume` requires a flattened checkpoint
with compatible optimizer state. See [continuation and promotion](docs/FLATTEN_BOARD_CONTINUATION.md).

**The experiment baseline remains the verified flattened copy of ckpt34 until a trained
candidate passes the matched SF2400 halving comparison.** Architecture adoption
does not itself promote a checkpoint.

Stage 1 learns human moves with full-vocabulary cross entropy and configured Elo weighting. Stockfish annotations supervise the value head where present, using the fixed win-percent transform; moves-left supervision remains available. See [event alignment](TRAINING_EVENT_SCHEMA.md), [board encoding](FEN_TO_BOARD_STATE.md) and [value targets](docs/VALUE_TARGET_WINPERCENT_HANDOFF.md).

Stage 2 starts from a checkpoint and human-game prefixes streamed from the training corpus. One frozen actor controls both colors through each continuation. Completed games provide noise-free improved Gumbel policies and actual outcome WDL targets; human prefixes provide unsupervised context. Loss is soft legal-policy cross entropy plus outcome WDL cross entropy, with no Elo weighting or moves-left loss. Unfinished games receive no labels.

Collection and training alternate on one GPU. Optimizer steps occur per training batch; the collection actor advances after a completed training phase. Immutable replay, atomic checkpoints, RNG/sampler state and phase progress support resume. Search uses one pending neural leaf per game, exact terminal values, full repetition history and explicit context limits.

Runs can opt into [regret-guided restarts](docs/SELF_PLAY_REGRET_RESTARTS.md),
which allocate a fifth starting-position bucket to high-error observed histories.
Enable `[regret]` in a new run; existing configurations keep their four-bucket curriculum.

## Setup and commands

Install project dependencies and native bindings with `uv sync --extra dev`. Dataset/checkpoint files are local artifacts and are not committed. Use each command's `--help` for its required inputs.

```bash
# Stage 1: use the config matching the intended architecture.
.venv/bin/python scripts/train.py --config config/imba_chess_v4.toml

# Stage 2: see "Recommended stage-2 run" below.
```

## Recommended stage-2 run

`scripts/run_self_play.py` defaults to the recipe of the 2026-10 long run on a 16 GB RTX 5070 Ti,
so starting self-play after stage-1 pretraining needs only an initialization and an output directory:

```bash
# 1. Add the zeroed auxiliary value heads the recipe trains; stage-1 checkpoints lack them.
.venv/bin/python scripts/add_auxiliary_value_head.py \
  --checkpoint artifacts/<stage-1 run>/<checkpoint>.pt --output artifacts/self_play/initial-aux3.pt

# 2. Start the run. It is continuous: it stops only on RUN/STOP or SIGTERM.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/run_self_play.py \
  --initialize artifacts/self_play/initial-aux3.pt --output artifacts/self_play/new-run

# 3. Every later start goes through the watchdog, which resumes after a crash (at most 3 an hour).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/self_play_watchdog.sh artifacts/self_play/new-run

# Stop cleanly (the current full state is saved): touch the STOP file, then SIGTERM the run.
touch artifacts/self_play/new-run/STOP   # remove it before restarting
```

| Default | Value | Notes |
|---|---|---|
| `--config` | `config/self_play.toml` | Plain Gumbel 256, raw Q at value scale 0.5, bf16 collection, fast play-out of decided games, carry-over of in-flight games, 500k replay at reuse 1, constant LR 2e-4, EMA 0.995. See the [config guide](docs/CONFIG_GUIDE.md). |
| `--seeds` | `artifacts/corpus/v4_self_play_seeds_4096.json` | Held-out monitor openings for screens; build once with `scripts/materialize_corpus.py` and `scripts/prepare_self_play_seeds.py`. Training starts stream from the corpus. |
| `--collect-workers`, slots | 5 workers x 32 games (config `concurrent_games = 160`) | About 11 GB peak on 16 GB. 6 workers never fit, and 5 x 40 ran out of memory beside a Stockfish eval. |
| `--cpu-threads` | 1 | Idle OpenMP threads slow the single search thread of each worker. |
| `--continuous` | on | `--no-continuous` stops after the config's `[run] hours`; `--until` sets a deadline. |
| `--observe-only-screen` | on | Screens record strength against the initial actor without rollback or promotion; `--no-observe-only-screen` gates on them. |
| `--screen-games` | 100,000 | About 7 hours. |
| `--save-every` | 5 | The actor (a full trainer state, EMA included) is saved every 5th iteration (~12 min with 640-game phases), at screens and at graceful stops; weights stay in memory between saves. Screens also write the EMA they evaluate. |

Run the workers under the CUDA MPS daemon (`nvidia-cuda-mps-control -d`, or a user systemd unit),
so the 5 processes share the GPU concurrently. On the 5070 Ti, at commit 8e757d6, the recipe ran at
about 82k neural evaluations/s and 69 s per iteration: ~14.8k games/hour, ~355k a day, 10-11.5 GB peak.
On a larger GPU, raise `--collect-workers` and `--concurrent-games`. The 32 GB 5090 ran
12 workers x 28 games in bf16. Both flags are execution-only and keep the resume identity.

Screens only compare the run against itself. Track strength with Stockfish about twice a day
(100 games, 512 simulations):
`scripts/extract_ema.py --checkpoint RUN/actor-NNNNNN.pt --output <dir>/ema-NNNNNN.pt`, then
`scripts/run_stockfish_eval.py --checkpoint <dir>/ema-NNNNNN.pt --config config/imba_chess_v4_aux3.toml --out <dir> -- --gumbel-simulations 512`.
Use the first saved actor at or after the iteration you want, and extract its EMA promptly: the run prunes old checkpoint files. The 2026-10 lineage sat at about 44%
against SF2600 (ema-100 of the LR 2e-4 run).

Collection is limited by its single search thread, not the GPU. `--collect-workers N` (default 5) splits `--concurrent-games` across N processes sharing the GPU. The main process keeps replay, streamed starts, regret state and training, and workers load its weights at every collect phase. On the 8 GB laptop, 2 workers × 32 games collected 24% faster than one process with 64 games; 4 × 16 saturated the GPU. `--cpu-threads` (default 1) stops idle OpenMP workers from spinning on every core. Both flags are execution-only and keep the resume config identity.

Collection aborts on gameplay errors, preserving unfinished launch IDs for retry without labels. Workers have execution-only startup, progress and shutdown timeouts: `--worker-startup-timeout` (900 seconds), `--worker-progress-timeout` (300 seconds), and `--worker-shutdown-timeout` (5 seconds). Progress time excludes coordinator work such as waiting for streamed starts. Stalled workers are killed; Linux workers also die if their coordinator exits. Increase the progress timeout if first-use compilation or a large search cohort needs longer.

Worker deadlines also cover incomplete pipe messages. Stop requests prevent new launches while active games drain or become unlabeled interruptions.

`--initialize-optimizer` carries the supervised optimizer and remaining OneCycleLR schedule into self-play. After that schedule ends, learning continues at each parameter group's terminal minimum rate. This behavior and the schedule clock survive stage-2 checkpoints.

Search inference uses CUDA FP32 with TF32 disabled. Gumbel uses the compiled decoder and reusable workspace (maximum depth 32); halving uses the grouped cached decoder with its configured depth. Runtime choices follow the selected algorithm. Ordinary model-component SDPA remains unchanged. Compilation adds first-use latency. Historical measurements remain in the [readiness report](docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md).

Use [the config guide](docs/CONFIG_GUIDE.md) before changing an existing run. The 5090 recipe reproduces the running tactical fork of the vmix run (auxiliary-value base config); it is not a measured performance promise.

## Evaluation and monitoring

`eval_vs_stockfish.py` and `match_two_checkpoints.py` expose `--model-move-policy value_search_halving` (default) or `--model-move-policy gumbel`. One algorithm applies to the entire match. Halving uses `--search-budget` neural evaluations and `--search-lambda`; Gumbel uses `--gumbel-simulations` and explicitly supplies zero exploration noise for standalone evaluation. These budgets are different units. Both paths use shared inference and scheduling; Stockfish calls remain concurrent. Halving retains its existing game concurrency (6 in the config; recent Stockfish runs use 4). PGN/HTML traces remain supported. `match_two_checkpoints.py` compares models directly without Stockfish.

The old standalone Gumbel evaluation commands are retired. Select Gumbel explicitly on the common commands. Retired policies and inference optimization/dtype/compile switches are rejected; supervised-training controls remain separate.

Stage-2 screens compare matched held-out prefixes with colors swapped. A run screens at the first phase boundary after `--screen-games` completed games (default 100,000, about 7 hours on the 5070 Ti; the count persists across resumes), or after `--screen-seconds` of wall time instead. A completed 500-game confirmation with paired 95% confidence above 50% is required for best-checkpoint promotion. Incomplete evaluations do not establish a score. Keep full-strength fixed-node Stockfish results separate from historical Elo-limited SF2400 results.

Track usable completed positions/hour, completion/discard rates, training positions/second, policy CE/entropy/KL, WDL CE/Brier, predicted versus observed draws, gradient norms, replay age/reuse, and paired strength scores. Lower training loss alone does not prove stronger chess.

## Validation

```bash
.venv/bin/python -m pytest -q                         # fast default
.venv/bin/python -m pytest -q -m extended             # compiler/device + broad sweeps
.venv/bin/python -m pytest -q -m ''                   # all root checks
.venv/bin/python -m pytest -q native/imba_chess_native/tests
```

The extended suite should run on decoder, attention or native-rule changes. Retained tests cover loss/gradient behavior, replay recovery, worker cleanup, chess edge cases, and full-forward → cached → grouped → optimized decoder parity. See [test maintenance](https://github.com/viig99/imba-chess/blob/0997951a18984ebdf16a5e1097a9d28382075e00/docs/TEST_MAINTENANCE_AUDIT_2026-09-12.md).

## Status and retained history

- [Self-play implementation, measurements and remaining gates](docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md)
- [Current roadmap](PLAN.md)
- [Configuration compatibility](docs/CONFIG_GUIDE.md)
- [Historical results, retired tools and retired reports](docs/EXPERIMENT_HISTORY.md)

Inspiration: [grpo_chess](https://github.com/noamdwc/grpo_chess) and [searchless_chess](https://github.com/google-deepmind/searchless_chess).
