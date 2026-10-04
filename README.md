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

# New stage-2 run (requires a prepared monitor seed manifest).
.venv/bin/python scripts/run_self_play.py \
  --config config/self_play_5090.toml \
  --initialize artifacts/flatten-board-ckpt34/initial.pt \
  --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play/new-run --device cuda

# Resume the same run/config with its optimizer and replay state.
.venv/bin/python scripts/run_self_play.py \
  --config config/self_play_5090.toml \
  --resume --seeds artifacts/corpus/v4_self_play_seeds_4096.json \
  --output artifacts/self_play/new-run --device cuda
```

Collection is limited by its single search thread, not the GPU. `--collect-workers N` splits `--concurrent-games` across N processes sharing the GPU. The main process keeps replay, streamed starts, regret state and training, and workers load its weights at every collect phase. On the 8 GB laptop, 2 workers × 32 games collected 24% faster than one process with 64 games; 4 × 16 saturated the GPU. `--cpu-threads` (default 1) stops idle OpenMP workers from spinning on every core. Both flags are execution-only and keep the resume config identity.

`--initialize-optimizer` carries the supervised optimizer and remaining OneCycleLR schedule into self-play. After that schedule ends, learning continues at each parameter group's terminal minimum rate. This behavior and the schedule clock survive stage-2 checkpoints.

Search inference uses CUDA FP32 with TF32 disabled. Gumbel uses the compiled decoder and reusable workspace (maximum depth 32); halving uses the grouped cached decoder with its configured depth. Runtime choices follow the selected algorithm. Ordinary model-component SDPA remains unchanged. Compilation adds first-use latency. Historical measurements remain in the [readiness report](docs/SELF_PLAY_READINESS_REVIEW_2026-09-11.md).

Use [the config guide](docs/CONFIG_GUIDE.md) before changing an existing run. The 5090 recipe reproduces the running tactical fork of the vmix run (auxiliary-value base config); it is not a measured performance promise.

## Evaluation and monitoring

`eval_vs_stockfish.py` and `match_two_checkpoints.py` expose `--model-move-policy value_search_halving` (default) or `--model-move-policy gumbel`. One algorithm applies to the entire match. Halving uses `--search-budget` neural evaluations and `--search-lambda`; Gumbel uses `--gumbel-simulations` and explicitly supplies zero exploration noise for standalone evaluation. These budgets are different units. Both paths use shared inference and scheduling; Stockfish calls remain concurrent. Halving retains its existing game concurrency (6 in the config; recent Stockfish runs use 4). PGN/HTML traces remain supported. `match_two_checkpoints.py` compares models directly without Stockfish.

The old standalone Gumbel evaluation commands are retired. Select Gumbel explicitly on the common commands. Retired policies and inference optimization/dtype/compile switches are rejected; supervised-training controls remain separate.

Stage-2 screens compare matched held-out prefixes with colors swapped. A run screens at the first phase boundary after `--screen-games` completed games (default 1,500, about 3 hours on the 5090 tactical recipe; the count persists across resumes), or after `--screen-seconds` of wall time instead. A completed 500-game confirmation with paired 95% confidence above 50% is required for best-checkpoint promotion. Incomplete evaluations do not establish a score. Keep full-strength fixed-node Stockfish results separate from historical Elo-limited SF2400 results.

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
