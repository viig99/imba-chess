# Evaluation Specification

This document defines how `imba-chess` should evaluate next-move prediction during pretraining, and what to defer to RL/game-play evaluation.

## Scope

- Phase 1 (now): offline next-move prediction evals on held-out data.
- Phase 2 (later): engine/self-play match evals after RL fine-tuning.
- Runtime framework: use `pytorch-ignite` engines for eval/train/validation orchestration.

## Dataset Sources

- In-domain source: `Lichess/standard-chess-games`.
- Out-of-domain source: `Lichess/tournament-chess-games` (if schema-compatible or adapted via a mapper).

Notes:
- `Lichess/standard-chess-games` is hive-partitioned by `year` and `month`.
- Use partition-aware filtering (`year/month`) for deterministic temporal splits.

## Split Policy (Default: Temporal Month Partitions)

Default split should be chronological by `year/month` partitions, not random/hash split.

Baseline profile (current plan):

- `train`: `2018-01` through `2025-07`
- `val`: `2025-08` (single month)
- `test_in_domain`: `2025-09` (single month)

Rules:

- Split by partition path (`year=YYYY/month=MM`).
- No future leakage: `train` must end strictly before `val/test`.
- Keep `val` and `test_in_domain` frozen for model comparison.
- Pin Hugging Face dataset revision (commit hash) for reproducibility.

## Additional Test Set

Define `test_out_of_domain` from `Lichess/tournament-chess-games`.

Policy:
- Do not use this set for model selection/hyperparameter tuning.
- Use it as a final generalization check only.
- If fields differ from `standard-chess-games`, implement a small schema adapter in the dataset reader and document differences.

## Primary Pretrain Metrics

Track on `val` every eval interval and on `test_in_domain` at checkpoint milestones:

- `loss_ce`: cross-entropy on target move IDs (ignore BOS positions via ignore index).
- `ppl`: perplexity (`exp(loss_ce)`).
- `top1_acc`: fraction where argmax move equals target.
- `top3_acc`
- `top5_acc`
- `top10_acc`
- `mrr`: mean reciprocal rank of the ground-truth move in model ranking.

Recommendation:
- Use these as main model-selection signals: `loss_ce`, `top1_acc`, `mrr`.

## Legal-Move-Aware Diagnostics

These are diagnostics, not replacement objectives:

- `legal_top1`: top-1 prediction is legal in current board state.
- `legal_topk_mass` (k=5,10): fraction of top-k predictions that are legal.

These help detect invalid-move behavior when training without explicit legality masking.

## Slice Metrics (Required)

Report the same core metrics by:

- Game phase: `opening` (ply 1-20), `middlegame` (ply 21-60), `endgame` (ply 61+).
- Elo buckets: `2000-2199`, `2200-2399`, `2400+`.

Purpose:
- Catch regressions hidden by global averages.

## Metrics Not Prioritized for Pretraining

- `NDCG@k`: not a primary fit for single-label next-token prediction.
- `HitRate@k`: largely overlaps with top-k accuracy in this setup.

If needed, these can be added later for external comparison, but they are not required in phase 1.

## Eval Cadence and Size

- `val`: run frequently (for example every N training steps).
- `test_in_domain`: run only at checkpoint milestones.
- `test_out_of_domain`: run at major milestones only.

Use fixed token budgets for eval loops so runs are comparable:

- `val_tokens_budget`
- `test_tokens_budget`

Both budgets must be constant across experiments in the same comparison table.

## Model Selection Policy

Select checkpoints by `val`:

- Primary: lowest `loss_ce`.
- Tie-breaker 1: highest `top1_acc`.
- Tie-breaker 2: highest `mrr`.

Never select by `test_in_domain` or `test_out_of_domain`.

## RL-Phase Evaluation Boundary

Defer engine match play to RL/post-SFT phase:

- Stockfish matches
- Leela matches
- self-play Elo ladders

Reason:
- Offline policy quality should be stabilized first.
- Engine play is slower, noisier, and better used for late-stage policy validation.

## Reproducibility Requirements

- Fixed split month ranges and pinned dataset revision.
- Fixed eval token budgets.
- Fixed random seed for any sampling inside eval loops.
- Log config snapshot with every eval report.

## Minimum Deliverables for Phase 1

- Deterministic temporal `train/val/test` split in streaming pipeline (year/month windows).
- Eval runner for core metrics (`loss_ce`, `ppl`, `top-k`, `mrr`).
- Slice reporting (phase + Elo buckets).
- Frozen in-domain and out-of-domain test reports per milestone checkpoint.
