# Stockfish Eval Plan (Step 2)

This document captures the next-phase plan after the minimal `scripts/eval_vs_stockfish.py` baseline.

## Current Step-1 Baseline

- Play fixed-count games (e.g. 1000) model vs `/usr/bin/stockfish`.
- Alternate colors per game.
- Use python-chess UCI engine API.
- Report total `wins/draws/losses` and side split.

## Why Step-2 Is Needed

Single-opponent-single-setting results can be noisy and not diagnostic. We need segmented evaluation for:

- strength curve (how model score changes vs stronger opponent settings),
- opening robustness (not overfitting one opening family),
- color robustness (white/black asymmetry),
- reproducibility across seeds.

## Stockfish Strength Semantics

- `UCI_LimitStrength=false`: Stockfish runs full strength.
- `UCI_LimitStrength=true` + `UCI_Elo=<x>`: approximate Elo-limited mode.
- There is no post-game API that returns "actual played Elo". Elo must be inferred from match outcomes and opponent settings.

## Step-2 Ladder Evaluation

Run multiple segments and aggregate:

- Segment A: Elo-limited ladder (`UCI_LimitStrength=true`)
- Candidate ladder: `1600, 1800, 2000, 2200, 2400, 2600, 2800`
- Segment B: Full-strength baseline (`UCI_LimitStrength=false`)

Per segment:

- fixed number of games (suggest `200-400`),
- strict color balance (50/50),
- identical time control/nodes/depth policy,
- fixed seed logged.

## Opening Protocol

To reduce opening bias:

- Use a curated opening suite (FEN list or PGN opening book).
- Pair games per opening with side swap.
- Report per-opening-family scores (ECO buckets if available).

## Reporting Additions

For each segment:

- `games, wins, draws, losses, score_rate`,
- side split (`as_white`, `as_black`),
- incomplete/terminated-by-cap count.

Across segments:

- score-vs-ladder curve,
- optional logistic Elo estimate relative to configured opponent Elo,
- confidence interval (bootstrap or normal approximation).

## Reproducibility and Runtime Controls

- Log complete engine options (`Threads`, `Hash`, `UCI_LimitStrength`, `UCI_Elo`).
- Log search limit (`time`, `nodes`, `depth`).
- Log model checkpoint hash/path and git commit hash.
- Save JSON report for each segment and one merged summary artifact.

## Potential Extensions

- Head-to-head against additional engines (Lc0, Maia) for style/strength diversity.
- Puzzle/tactical suite for non-match tactical signal.
- Time-control sweep (bullet/rapid/classical proxies via move time or nodes).
- Auto-regression checks in CI with tiny game counts (smoke only).
