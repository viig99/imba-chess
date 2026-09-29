# Configuration guide

| File | Use |
|---|---|
| `imba_chess_v4.toml` | Base architecture/data config and the ckpt34 SF2400 evaluation protocol. Default `--config` for training, Stockfish/model-pair evaluation and corpus scripts. |
| `imba_chess_v4_laptop.toml` | Laptop supervised continuation from flattened ckpt34 at its saved learning rate, in a separate checkpoint directory. |
| `eval_flatten_sf2400.toml` | Frozen SF2400 halving recipe for flattened checkpoints (ckpt34 r4/b2048 confirmation, seed 1042). |
| `self_play_streaming.toml` | Laptop stage-2 run: 24 collection slots, 128 simulations, depth 32. |
| `self_play_eval.toml` | 512-simulation stage-2 settings used for paired Gumbel evaluations. |
| `self_play_5090.toml` | Planned 32 GB pilot: 32 slots and larger training/replay settings. Never soaked. |

Every stage-2 run streams its starting positions from the training corpus through `[streaming]`. The seed manifest passed with `--seeds` supplies only the held-out monitor openings used by strength screens.

A model checkpoint needs compatible architecture, input encoding, vocabulary and head shapes; only flattened checkpoints load. Stage-2 resume verifies a SHA-256 of every setting plus the base-config bytes, so any edit to either, including a comment, is a new run identity. Paired evaluation progress is keyed on the same identity. Runs and in-progress evaluations created before 2026-09-29 cannot be resumed by current code; runs launched from a source snapshot resume with that snapshot.

Execution controls can change without replacing training state: CUDA inference uses FP32 with TF32 disabled, and decoder choices follow the algorithm internally. `run_self_play.py --concurrent-games` overrides collection slots without altering search targets or the stored config identity, and records the override in run metrics. Evaluation concurrency remains controlled by its config. None of these change training token batch size, LR, reuse or collection thresholds; changing learning settings is a separate experiment.
