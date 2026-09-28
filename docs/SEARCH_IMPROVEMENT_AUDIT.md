# Frozen-model search diagnostics

Run `python -m scripts.audit_search_improvement match` for paired search-versus-greedy games, or `target` for complete-game training-target scoring. Neither command updates weights or participates in promotion. The production promotion evaluator remains Gumbel-only.

The September 20 campaign is in `artifacts/eval/search-improvement-2026-09-20/`. Its `run.py` waits for the entire existing actor 54 campaign and then for an idle GPU. It runs a one-pair smoke, both 25-pair match budgets, a one-position target smoke, and the remaining target scoring. Read `status.json`, the per-stage logs, and eventually `REPORT.md`. The runner checks frozen source hashes before starting; a source change requires inspection and an explicit new freeze. Remote replay connection refusal is recorded in `remote-replay-attempt.json`.

Example standalone match:

```sh
.venv/bin/python -m scripts.audit_search_improvement match \
  --config artifacts/eval/search-improvement-2026-09-20/config.toml \
  --checkpoint artifacts/eval/search-improvement-2026-09-20/ckpt34.pt \
  --seeds artifacts/eval/search-improvement-2026-09-20/monitor-seeds.json \
  --output artifacts/eval/search-improvement-2026-09-20/match
```

The defaults run budgets 32 and 200, 25 distinct openings each with colors swapped. Search uses scale 0.1, top-m 16, depth 32, no Gumbel noise, FP32 and no TF32. Greedy uses the same runtime, legal projection, full history and model weights, reading only policy logits. Routing identities include checkpoint hash, algorithm and simulation budget. Completed games are saved individually within each budget's JSON state and skipped on restart. Administrative stops remain unlabeled. Paired bootstrap intervals use complete opening pairs, and any missing games are reported explicitly.

Example target audit with a read-only local snapshot of remote replay:

```sh
.venv/bin/python -m scripts.audit_search_improvement target \
  --config artifacts/eval/search-improvement-2026-09-20/config.toml \
  --checkpoint artifacts/eval/search-improvement-2026-09-20/actor54.pt \
  --seeds artifacts/eval/search-improvement-2026-09-20/monitor-seeds.json \
  --replay /path/to/read-only/replay \
  --output /path/to/new/target-audit
```

Without `--replay`, specify `--remote-failure` documenting the failed read-only connection. Fallback generation uses the frozen actor checkpoint with training Gumbel noise and openings 26–45, making at most 20 attempts to obtain ten qualifying completed games. Attempts and selected complete games survive restarts. No optimizer is loaded. The campaign archives the original generating training configuration separately; its diagnostic config carries the same surprise fraction 0.5 and cap 3.0, while reducing inference concurrency to 12. Detaching value features is a training-only operation and is not needed for inference.

Selection shuffles replay game IDs with seed 42 and freezes the first ten qualifying games. Within each game, it samples one eligible position from each of ten consecutive segments of the eligible trajectory, with a reproducible game-specific seed derived from 42. Whole-game surprise weights are calculated before this sampling. Full histories, legal vocabulary alignment, played-move alignment and terminal outcomes are validated. Stored actor priors are never recomputed.

Stockfish runs unrestricted with one thread, 64 MiB hash and 100,000 nodes per legal move, clearing hash before each analysis. All scores use the original side to move. Per-move caching makes scoring restartable. The report includes full-distribution WDL expectation gains, weighted gains, fractions, game-clustered bootstrap intervals, per-game results, and harmful positions. Played-minus-actor-greedy scores are separate because played moves include exploration. CP distribution gains are omitted for any position with a mate-scored legal move; raw mate scores remain in the artifacts.

`--stop-after-positions 1` scores a target smoke position after freezing the complete sample. Rerun the same command without this option to finish. Incompatible checkpoint, configuration, opening or engine identities are rejected. The full target record is `target.json`; compact statistics and harmful corrections are in `target-summary.json`.

These are small-sample engine proxies and paired match measurements for different frozen actors. They do not measure transfer of search improvement through training. Review both measurements before proposing another training change.
