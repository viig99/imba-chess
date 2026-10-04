# Configuration guide

| File | Use |
|---|---|
| `imba_chess_v4.toml` | Base architecture/data config and the ckpt34 SF2400 evaluation protocol. Default `--config` for training, Stockfish/model-pair evaluation and corpus scripts. |
| `imba_chess_v4_laptop.toml` | Laptop supervised continuation from flattened ckpt34 at its saved learning rate, in a separate checkpoint directory. |
| `imba_chess_v4_aux3.toml` | v4 plus three auxiliary WDL heads; base for stage-2 runs with auxiliary value learning. |
| `eval_flatten_sf2400.toml` | Frozen SF2400 halving recipe for flattened checkpoints (ckpt34 r4/b2048 confirmation, seed 1042). |
| `self_play_5090.toml` | 32 GB recipe of the tactical fork of the vmix run: 256 slots, tactical Gumbel 512 (root forcing, forcing floor, minimax weight 0.5), 1M-position replay, LR 2e-4 with 4-step accumulation, 0.25 search-WDL value mix, 3-horizon auxiliary value (0.15), regret restarts. |

Every stage-2 run streams its starting positions from the training corpus through `[streaming]`. The seed manifest passed with `--seeds` supplies only the held-out monitor openings used by strength screens.

A model checkpoint needs compatible architecture, input encoding, vocabulary and head shapes; only flattened checkpoints load. Stage-2 resume verifies a SHA-256 of every setting plus the base-config bytes, so any edit to either, including a comment, is a new run identity. Paired evaluation progress is keyed on the same identity. Runs and in-progress evaluations created before 2026-09-29 cannot be resumed by current code; runs launched from a source snapshot resume with that snapshot.

Execution controls can change without replacing training state: CUDA inference uses FP32 with TF32 disabled, and decoder choices follow the algorithm internally. `run_self_play.py --concurrent-games` overrides collection slots without altering search targets or the stored config identity, and records the override in run metrics. Evaluation concurrency remains controlled by its config. None of these change training token batch size, LR, reuse or collection thresholds; changing learning settings is a separate experiment.

The shared inference runtime revision `shared-search-v2` preserves root capture/check/promotion flags and backs up pure negamax values before applying `minimax_weight` once per edge. Existing self-play weights and optimizer state can resume with these corrections; future games use the corrected search. An in-progress evaluation saved with the previous runtime revision requires a new output file or, for `run_stockfish_eval.py`, a new output directory. The batched SF runner checks both its manifest and each completed batch so old results cannot be mixed with the corrected search protocol.
