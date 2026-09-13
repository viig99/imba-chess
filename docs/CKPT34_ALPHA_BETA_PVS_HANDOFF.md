# ckpt34 alpha-beta / PVS experiment — retired

Alpha-beta and PVS were removed from the active codebase on 2026-09-11 at the
user's request. Halving remains the default search. This document and the
validation artifacts are historical evidence, not instructions for the current CLI.

## Recorded results

All runs used ckpt34, SF2400, a 40,000-node/5-second Stockfish limit, fp32,
compilation, budget 2048, seed 42, and four configured actor slots. The two-game
smokes could have only two active games. Halving used depth 8, top-m 16, own
expansion 3, replies 4, lambda .05, coverage off, and Q=0. The new backends used
root-relative depth 9 with iterative deepening. Cache and LMR were off in the
measured alpha-beta and PVS runs.

| Backend | Completed games | W / D / L | Score | Mean move selection |
|---|---:|---|---:|---:|
| Halving screen | 100 | 60 / 24 / 16 | 72.0% | 0.58 s |
| Alpha-beta screen, interrupted | 28 reported | 2 / 7 / 19 | 19.6% | No final aggregate |
| Alpha-beta smoke | 2 | 0 / 0 / 2 | 0.0% | 14.8 s |
| PVS diagnostic | 2 | 0 / 2 / 0 | 50.0% | 19.4 s |

The alpha-beta screen was stopped by the user; its counts come from the last
consistent progress-log aggregate. The remaining 72 planned games have no final
results and must not be represented as completed games. The PVS diagnostic finished
within its 30-minute cap, in about 19 minutes. No cache/LMR screens ran, and the
rest of the campaign was cancelled.

PVS's two draws do not establish playing strength. The implementations usually
completed depth 3–4 under budget and issued almost exclusively single-position GPU
batches. The recorded latency and alpha-beta's poor partial results did not justify
keeping the backends active. These conclusions concern these implementations with
ckpt34, not alpha-beta or PVS as general search algorithms.

## Removed and retained

Removed the torch-free alpha-beta/PVS controller, context-score cache, guarded LMR,
evaluation selectors and dispatch, iterative-deepening/cache/LMR settings, dedicated
tests, per-move reports with no remaining producer, and the campaign runner.

Retained halving, greedy, value reranking, depth-two search, native terminal handling,
KV infrastructure, and shared game-completion/inference-batch diagnostics. Training
and rollout policies were not changed. Existing unrelated working-tree edits were
preserved.

## Evidence and recovery

Local run artifacts:

- `artifacts/eval/ckpt34-ab-pvs-screen-20260908`: control, interrupted alpha-beta,
  source/config snapshots, logs, and cancellation manifest.
- `artifacts/eval/ckpt34-pvs-diagnostic-20260908`: final PVS result and diagnostic log.
- `docs/validation/ckpt34-ab-pvs`: historical implementation validation, CPU smoke,
  and prepared campaign manifest/commands. The prepared commands require the old
  revision and must not be run against the current CLI.

Checkpoint: `artifacts/checkpoints_v4/best_hr10_checkpoint_34_hr10=0.9677.pt`,
using `config/imba_chess_v4.toml`.

- Checkpoint SHA256: `5844b09fdde268f5fd2aba363603c43e9c1020d776c2d5294a17c2f912962826`
- Stockfish SHA256: `2d7cae60c9233a7ff1bd89d74cfa08fa17bbbc6b87b4be3f9297f4ca63b18292`
- Evaluation revision: `9a9f32e45b813efd4c693c6d5909563291ccb499`, plus the working-tree
  patch archived with the campaign.

The original stage commits remain recoverable in Git: `26aa5c1` (alpha-beta),
`73ae249` (PVS), `d6009ed` (context cache), and `063ae18` (LMR). Their tests passed
before screening; correctness tests did not establish strength or practical speed.

The earlier CUDA-unavailable diagnosis was a sandbox-access issue. GPU access worked
outside the sandbox on the RTX 3070 Ti Laptop GPU, and the evaluations above ran there.
