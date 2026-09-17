# Configuration guide

All eight TOMLs are retained. None was modified during the 2026-09-13 cleanup.

| File | Use |
|---|---|
| `imba_chess_v4.toml` | Base architecture/data and historical ckpt34 evaluation settings. Use with v4 checkpoints. |
| `self_play_laptop_fast.toml` | New laptop runs: 24 collection slots, 128 simulations, depth 32; compiled CUDA decoding is selected by runtime. |
| `self_play_5090.toml` | Planned 32 GB pilot: 32 slots and larger training/replay settings. Remote soak remains unverified. |
| `self_play.toml` | Original generic stage-2 recipe; retained for comparisons and compatible resumes. |
| `self_play_laptop_pilot.toml` | Original two-hour laptop run; retained to resume its optimizer, replay and actor history. |
| `imba_chess.toml` | Default stage-1/evaluation recipe; preserve for checkpoints and commands using it. |
| `imba_chess_v3.toml` | Older architecture/training recipe; use only with the matching checkpoint. |
| `imba_chess_sf_finetune_low_lr.toml` | Supervised low-LR fine-tuning recipe; this is separate from stage-2 outcome learning. |

A model checkpoint needs compatible architecture, input encoding, vocabulary and head shapes. Stage-2 resume additionally verifies the settings and the SHA-256 of the base-config bytes. Reformatting or commenting the base config can change that identity even without changing numerical settings. Do not rewrite old recipes to deduplicate them while associated runs must remain resumable.

Execution controls can change without replacing training state: CUDA inference uses FP32 with TF32 disabled, and decoder choices follow the algorithm internally. The iteration/overnight runner's `--concurrent-games` overrides collection slots without altering search targets or the stored config identity, and records the override in run metrics/supervisor identity. Existing evaluation concurrency remains controlled by its config. It does not change training token batch size, LR, reuse or collection thresholds.

The old pilot can therefore resume with 24 slots while preserving its optimizer and original config. A new run can use the fast laptop recipe directly. Changing learning settings is a separate experiment and must preserve explicit provenance rather than bypass resume validation.

Search consolidation removes evaluation dtype/compile fields and renames `value_rerank_lambda` to `search_lambda` without changing its value. The exact shipped config migrations preserve existing self-play config identities; modifying other config bytes still invalidates resume. Checkpoint, replay, and optimizer serialization formats are unchanged. Evaluation resumes additionally require matching algorithm, budget, exploration, precision, and runtime revision. Historical reports and source snapshots retain their original options as records.
