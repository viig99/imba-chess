**Implementation update — 2026-09-13.** Applied the high-value reductions below. The original audit follows as historical evidence.

- Replaced the 800-game circular terminal oracle with ten explicit live/terminal/repetition/claim scenarios and both color perspectives, using `python-chess.Board.outcome(claim_draw=True)` independently. The phantom-en-passant replay uses the same independent oracle. An in-memory mutation returning draw for every board now fails.
- Replaced two duplicated hard-exit suites and production wrappers with a dependency-light `imba_chess.process` helper. Three real subprocess cases retain lingering-thread success/crash and SystemExit behavior; one wiring test checks both scripts resolve their current main through that helper. No mock substitutes for the real shutdown tests.
- Kept all 1,820 native parity cases. Their exact 470 selected random FENs are stored in `tests/fixtures/native_positions.json`, with original seeds and a regeneration script. Collection no longer generates/retains 30,092 snapshots or 1.2 million history entries. Full-history repetition tests remain separate.
- Removed the Elo test that calculated only its own formula; the actual weighted-loss test now includes below-minimum and above-maximum ratings. Removed nine opaque legacy scheduling hashes and recipe widths/budgets/parameter-count assertions; config parsing, unknown-key handling and cross-layer defaults remain.
- Reduced wrapper/CLI routing matrices to representative cases and replaced 24 random starvation seeds with two explicit adverse candidate orders, including exclusion of the global best prior. Reduced the OneCycle resume setup from 230,000 steps to 23 in a 100-step schedule and retained persistent LR/initial_lr checks.
- Registered `extended` and explicitly excluded it from the default suite. Compiler/device integration and three 200-game random sweeps stay runnable; small curated/random chess checks remain in the default suite. The decoder equivalence chain is preserved. Timing/sleep cleanup and small calibration-helper consolidation are deferred: they do not justify expanding this change.

**Measured validation:** `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q --durations=15` passed **2,280 tests**, deselected 17 extended cases, and emitted two warnings in **20.35 seconds**. This is one local run, compared with the audit's same-command 111.80-second baseline. The first cleanup run found two tests relying on an incidental script re-export; they now import the value conversion helper from its owning module and pass. Largest remaining costs are the real actor-process checks (~5.5 seconds combined), block-mask parity (~1.9 seconds) and the hard-deadline process test (~1.9 seconds). The extended CPU run passed four cases (13 device/CPU-compile skips) in 9.86 seconds; the explicit decoder CUDA suite passed 23 cases (two CPU-compile skips) in 9.17 seconds. Native bindings passed 107 tests in 0.03 seconds. Logs are under `artifacts/self_play_validation/maintenance_2026-09-13/`. Changed Python files pass Ruff with the existing importorskip-related E402 convention excluded, and compileall/whitespace checks pass.

Commands:

```bash
.venv/bin/python -m pytest -q                        # fast default
.venv/bin/python -m pytest -q -m extended            # compiler/device + broad sweeps
.venv/bin/python -m pytest -q -m ''                  # all root checks
.venv/bin/python -m pytest -q native/imba_chess_native/tests
.venv/bin/python -m tests.fixtures.generate_native_positions  # explicit corpus regeneration
```

---

Test maintenance audit — 2026-09-12

The suite has worthwhile coverage, but a few tests create a disproportionate amount of work and false confidence. Prioritize the circular terminal oracle, repeated subprocess imports, historical snapshots, and expensive fixture generation. Deleting large numbers of cheap parameterized cases would make the headline count smaller with little practical benefit.

This is an audit and proposed reduction list. No tests or production files were changed. The working tree already contained substantial changes, including deletion of the alpha-beta implementation and its two test modules; those deletions are not new savings from this audit. Findings apply to the working tree, including the new self-play tests.

**Measured baseline**

- Root suite: 46 files, 362 test function definitions, 11,602 lines, 2,348 collected cases.
- CPU run: **2,335 passed, 13 skipped, 5 warnings in 111.80 seconds**. GPU cases were deliberately disabled; two compiled CPU decoder cases explicitly skip in the tests themselves.
- Separate native binding suite: **107 passed in 0.04 seconds**. `testpaths = ["tests"]` excludes this suite from the normal root command.
- Root collection alone: 7.06 seconds. No cold compiler-cache guarantee; these timings are one local run, not a benchmark across hardware.
- `scripts/test_event_dataloader.py` is a manual batch-preview CLI, not a collected test. Renaming or removing it does not speed up pytest.

Commands used:

```bash
.venv/bin/python -m pytest --collect-only -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 360 \
  .venv/bin/python -m pytest -q --durations=35 \
  --junitxml=/tmp/imba-test-audit.xml
.venv/bin/python -m pytest -q native/imba_chess_native/tests --durations=5
```

| Cost center | Measured time | Interpretation |
| --- | ---: | --- |
| Circular terminal replay test | 32.96 s | Highest-priority replacement |
| Both hard-exit suites, ten cases | 26.68 s | Real regression, duplicated expensive execution |
| Entire dense-attention suite | 12.41 s | Includes 6.59 s explicit compilation and 5.65 s block-mask parity |
| Remaining cozy differential tests | 10.55 s | Independent checks worth keeping with bounded sweeps |
| Two real actor-mode integration cases | 6.05 s | Useful process-boundary coverage |
| All three native parity modules, 1,820 cases | ~0.084 s | Rounded JUnit test times; excludes collection cost |
| All Gumbel search tests, 48 cases | 0.12 s | Keep the useful budget matrix |
| All model tests, 24 cases | 0.11 s | Mostly meaningful loss/gradient contracts |

**Prioritized removals and reductions**

| Priority | Concrete target | Recommended action | Coverage to retain / tradeoff |
| --- | --- | --- | --- |
| 1 | `test_cozy_differential.py:161`, `test_terminal_value_native_matches_oracle_on_replayed_games` | Replace the 800-game sweep with a small direct python-chess outcome comparison; move a corrected large sweep to an explicit extended run if desired. | Mate, stalemate, insufficient material, fifty-move claims, repetition claims, both value perspectives, phantom en passant, and history reconstruction. See the proven oracle problem below. |
| 1 | `test_hstu_model.py:179`, `test_elo_normalization_clamps_at_config_bounds` | Delete this test as written. It computes and clamps its own formula without invoking the model's weighting implementation. Put below-minimum and above-maximum Elo inputs into the existing production weighted-loss test. | Production clamp behavior must be checked through actual loss computation; the current local tensor calculation does not protect it. |
| 1 | Both `test_*hard_exit.py` modules: `test_hard_exit_terminates_on_plain_exception_with_no_extra_threads` and `test_hard_exit_wrapper_is_transparent_on_success` | Remove these two cases from each file: **four cases, ~9.87 seconds**. | The same wrappers already have stronger exception-with-lingering-thread and success-with-lingering-thread tests that check the same exit codes/output. Keep `SystemExit` behavior. |
| 1 | Remaining duplicated hard-exit tests and wrappers | Share the test driver now. For the larger saving, extract the duplicated production wrapper into a small dependency-light helper, test its process semantics once, and retain thin wiring checks for both scripts. | A parameterized test over both scripts reduces source duplication but still starts the same number of processes. Keep real subprocess checks for hangs; mocks cannot prove interpreter shutdown. |
| 2 | `test_native_board_state.py`, `test_native_terminal.py`, `test_native_projection.py` import-time random-board setup | Generate/store only the selected boards; move generators and curated FENs into a support module. Avoid storing move stacks where the test deliberately converts the board through FEN and discards history. | First preserve the same seed and selected positions. Keep full histories in repetition/replay tests. Merely delaying generation until a fixture runs moves cost rather than eliminating it. |
| 2 | `test_cozy_differential.py:41,62,72` and `test_cozy_bridge.py:77` broad sweeps | Share a bounded deterministic position corpus; keep curated special-move positions in the normal suite and larger random sweeps in an extended suite. Merge repeated board generation/traversal where assertions protect the same conversion boundary. | `test_move_translation_roundtrips_all_legal_moves` already checks legal sets as part of bidirectional conversion. Preserve reverse push equivalence, check/capture semantics, castling, and promotions. Shrinking random coverage is a real tradeoff and should be explicit. |
| 2 | `test_dense_attn_mask.py:205` and CUDA/compiled cases in `test_tensor_decoder.py` | Separate compiler/device integration tests from the normal CPU feedback loop with registered markers and documented commands. Inspect block-mask parity setup too: a second test costs 5.65 s without an explicit compile in its test body. | Keep eager mask semantics, cross-document isolation, per-layer bias, and a routinely executed compiler/device job. A marker alone does not exclude anything, and GPU hardware should not silently change the default suite's cost. |
| 2 | `test_search_tactical.py:17–51`, nine `test_disabled_search_matches_prechange_trace` cases | Retire historical SHA-256 trace snapshots from routine tests. If exact legacy trace reproduction remains a supported requirement, replace them with a small readable fixture in an explicit compatibility suite. | Current tactical tests independently check move choice, evasions, quiescence and budgets. The hashes additionally freeze private row fields and evaluator scheduling; removing them relinquishes that historical trace contract. |
| 2 | `test_config.py:100,125,145,182` | Remove recipe tuning assertions such as exact search budget, Elo, model width and parameter-count range unless a named compatibility contract requires them. Keep schema loading, override precedence, rejection of unknown keys, and meaningful cross-layer defaults. | Config tuning should not require updating a second copy of the same values in tests. The v4 test constructs ~48M parameters just to count them; use metadata-only construction if the size contract is retained. |
| 3 | `test_search_stepwise.py:46`, twelve generator-versus-wrapper cases | Reduce to one representative nontrivial generator-driving check; optionally retain a zero-work case. | The public wrapper drives the same generator under test, so this checks adaptation, not independent algorithm correctness. Dedicated tactical tests and script/worker tests already cover flag behavior. |
| 3 | `test_search_stepwise.py:92`, 24 starvation seeds | Replace repeated seeds with two explicit adversarial candidate orderings, including one that omits the global best prior. Keep the deterministic inference case. | Preserve the actual bug trigger: first sampled arm differs from highest-prior sampled arm. Reducing to arbitrary seeds can accidentally stop triggering the regression. |
| 3 | `test_eval_vs_stockfish.py:38–99`, sixteen config/CLI/concurrency combinations | Reduce to four deliberately chosen wiring cases: config values through each execution route and CLI enable/disable overrides through each route. | These stub both game runners, so the product is not sixteen distinct end-to-end chess scenarios. Actual worker flag propagation and tactical interactions remain separately covered. |
| 3 | `test_train_lr_override.py:35–62` | Replace 230,000 OneCycle steps with a small schedule advanced to a comparable noninitial phase, then restore it and apply the override. Merge initial-LR checks into this scenario and use a few post-override steps. | Keep the resume ordering, `initial_lr` reset and persistent override. This costs only 0.47 s here, so it is a clarity improvement more than a major speedup. |
| 3 | `test_decode_prep_timing.py:43–66,101–150` | Replace busy loops and positive-wall-time assertions with a controlled clock; parameterize only the timing branches that matter. Move the cache-lifetime case out of this timing module. | Preserve attribution to prep/project/GPU buckets and `stats=None` support. Prefix identity/order, invalidation and weak ownership protect actual cache behavior and should stay. |
| 3 | `test_engine_pool.py:172–199`; `test_actor_worker.py:439–479` | Replace sleep-based synchronization with barriers/events and a child-ready handshake before SIGTERM. | Keep concurrent fan-out, result order and engine cleanup. A 100 ms timing threshold or a one-second assumption about process startup is fragile; deleting concurrency tests would hide real regressions. |
| 4 | `test_calibrate_stockfish_nodes.py`, nineteen helper tests | Consolidate the redundant singleton/identity cases. Keep interpolation, invalid inputs, half-up ties and resulting node-budget recommendations. Consider standard-library helpers where semantics match before retiring their tests. | Entire module costs 0.07 s; broad deletion is not a meaningful speed optimization. Parameterization improves readability but does not itself reduce executed cases. |

**The terminal oracle no longer independently checks terminal correctness**

The replay test imports `search.terminal_value_for_color` as its expected answer. That function now calls `cozy_bridge.terminal_value_native` (`src/imba_chess/eval/search.py:126–143`). The actual answer comes from that same classifier. Different board/history construction still gives some integration coverage, but a shared classification bug is invisible. The dedicated `_root_hash_seed` test already covers much of that reconstruction contract more directly.

I verified the weakness with this in-memory mutation, without editing repository files:

```python
from unittest.mock import patch
from tests.test_cozy_differential import (
    test_terminal_value_native_matches_oracle_on_replayed_games,
)

with patch("imba_chess.eval.cozy_bridge.terminal_value_native", return_value=0.0):
    test_terminal_value_native_matches_oracle_on_replayed_games()
# Passes even though every position is incorrectly classified as a draw.
```

The curated phantom-EP test at line 202 also uses the shared shim for its per-ply expected values. Keep its specific sequence and hash checks, but use direct python-chess outcomes with the intended draw-claim rules. Do not replace the oracle with another project wrapper. The Rust-versus-Python tests in `test_native_terminal.py` are independent implementations of classification, but both still need an external rules oracle to prevent a shared semantic mistake.

This is stronger evidence than a test name or a passing suite: the single slowest test does not reject a deliberately broken terminal classifier.

**Fixture generation is wasteful; most native cases themselves are cheap**

`_random_boards(n_games, seed)` returns every visited position, with copied move histories. Its consumers treat the argument like a requested sample size, then stride the resulting list down to that size.

| Module | Generated snapshots retained in module globals | Random positions selected | Stored move-stack entries |
| --- | ---: | ---: | ---: |
| Native board state | 10,109 | 150 | 425,845 |
| Native terminal | 7,075 | 120 | 274,129 |
| Native projection | 12,908 | 200 | 521,771 |
| Total | 30,092 | 470 | 1,221,745 |

This happens during collection, including when selecting an unrelated test with `-k` after those modules are imported. Extracting reusable helpers also matters: the actor projection test imports `test_native_projection`, which triggers that module's random corpus generation.

The three native modules account for **1,820 / 2,348 cases (77.5%) but only 19 test functions**. Keep their distinct contracts: three en-passant modes, both value perspectives, restricted vocabularies, exact legal-ID/order alignment, and forcing flags. A smaller default random sample is optional after fixing collection; it is not the first speed improvement I would make. If reduced, preserve every curated edge and run the larger corpus on relevant changes or in a scheduled job.

**Coverage worth retaining**

| Tests | Real use cases / suggested organization |
| --- | --- |
| `test_lichess_dataset`, `test_torch_iterable` | Filtering corrupt/bot/overlong games, validation split limits, deterministic splitting and rank/worker/file sharding. Silent duplication or train/validation leakage is costly. |
| `test_event_builder`, `test_collate`, `test_dataloader`, `test_stockfish_evals` | BOS alignment, post-move eval to next-state targets, value-sign changes by side to move, packed offsets, ragged lengths and ignored labels. Merge identical dataloader setup where helpful; retain the boundary checks. |
| `test_board_state`, `test_move_vocab` | Stable model input IDs and vocabulary mapping. Small tests with explicit expected IDs are useful anchors for larger parity tests. |
| `test_hstu_model`, `test_eval_metrics`, `test_train_lr_override` | Actual weighted/masked losses, gradients, tied-parameter optimizer treatment, checkpoint-compatible value-head names, resume semantics and metric aggregation. Keep these despite simple-looking assertions; remove the local-only Elo calculation identified above. |
| `test_search`, `test_search_tactical`, `test_gumbel_search` | Mate/refutation behavior, exact budgets, value sign backup, evasions, noise exclusion from training targets and upstream Gumbel reference fixtures. The 40-case Gumbel budget matrix is cheap and exercises meaningful boundaries. |
| `test_prefix_decode`, `test_grouped_decode`, `test_one_query_decode`, `test_tensor_decoder`, `test_dense_attn_mask` | A useful correctness chain: full forward to cached decode, cached decode to grouped execution, grouped to single-query, single-query to tensor/SDPA/compiled execution. Mixed histories, interleaved groups, stale-prefix replacement, immutable K/V and changing trained weights are distinct failure modes. Consolidate factories rather than deleting this chain. |
| `test_batched_projection`, `test_batched_decode_results`, `test_native_projection`, `test_cozy_move_id_cache` | Ragged legal masks, padding leakage, arena ownership, castling normalization and weak per-vocabulary caching. Old optimization-oriented filenames do not imply dead tests. |
| `test_native_board_state`, `test_native_terminal`, `test_cozy_bridge`, `test_cozy_differential` | Python/Rust convention and rules boundaries. Keep independent oracles and curated cases; fix circular expectations and oversized random generation/sweeps. |
| `test_actor_server`, `test_actor_worker`, `test_engine_pool`, `test_batch_scheduler` | Worker isolation, incremental roots, mixed full/incremental batches, missing or stale state, arena growth/release, no Torch imports in workers, error cleanup and completion ordering. Keep at least one real process success case and one crash case. |
| `test_eval_vs_stockfish`, `test_generate_search_rollouts`, `test_rollout_coroutine`, `test_rollout_store` | CLI wiring, game/result persistence, sequential/batched equivalence, coroutine sequencing, device placement and rollout schema. Reduce repeated scenario setup, not unique persistence/device contracts. |
| `test_self_play`, `test_self_play_overnight`, `test_self_play_runtime_options` | Prefix masks, unfinished-game handling, replay dedup/recovery/eviction, read-only snapshots, exact resume, promotion gates, run locks and interruption recovery. Split the 651-line self-play file by responsibility for maintainability, without presenting a file split as runtime savings. |
| `test_self_play_benchmarks` | Despite its name this is a 0.05 s component smoke test checking output and replay non-mutation, not a prolonged throughput benchmark. Keep. |
| `test_game_animation`, `test_calibrate_stockfish_nodes`, `test_config` | Modest helpers and wiring. Remove duplicated recipe expectations and redundant helper cases selectively; no broad purge needed. |
| `native/imba_chess_native/tests/test_imba_chess_native.py` | Owned Rust/Python binding contracts. Some enum/accessor checks are low-value in isolation, but all 107 execute in 0.04 s and are already outside the root suite. Reduce only alongside narrowing the binding API, not to claim speed savings. |

**Dead code versus historical scaffolding**

- `test_alphabeta.py` and `test_alphabeta_integration.py` are already deleted in the working tree. They are not collected. The remaining CLI rejection tests exercise current rejection behavior and are cheap.
- Rerank, depth-two search, synchronous decode, and grouped decode still have production callers. Their tests cannot be declared dead merely because halving or another decoder is the preferred mode.
- Old Python implementations used as genuine independent oracles have an ongoing testing purpose. Keep one clear oracle chain; the terminal example shows why its independence must be checked after refactors.
- The strongest dead-weight examples are the model test that only checks its own formula and the expensive terminal test whose oracle ceased to be independent. Historical trace hashes and generator-refactor matrices are candidates to retire or shrink, subject to their specific compatibility contract.

**Maintenance refactors to do with the cleanup**

Move shared FENs, model/batch factories, projection oracles and fake evaluators into small `tests/support/` modules. Existing imports connect test modules to each other: grouped/tensor decoder tests import prefix tests; tactical tests import stepwise/search tests; self-play helpers are imported by decode and benchmark tests; actor projection imports native projection tests. Avoid having an unrelated test's import construct thousands of positions or load a training stack.

Keep independent reference implementations separate from production helpers: sharing setup is useful, sharing the implementation that computes the expected answer defeats parity testing. Restore `torch.set_num_threads` after local changes, or configure it once for the suite; several tests currently leave this global setting changed.

Move the scheduler completion-order case out of the self-play module, cache ownership out of the timing module, and the decode merge/device regression out of rollout-script setup into executor tests. Share tiny model factories, but instantiate fresh mutable models for tests that train or alter weights. Do not introduce a giant test framework just to remove a few repeated lines.

**Suggested implementation order and acceptance criteria**

1. Replace the circular terminal sweep with small independent cases and ensure the constant-draw mutation fails. Remove the four weaker hard-exit cases and the local-only Elo formula test, preserving a production clamp assertion.
2. Share the hard-exit implementation and its tests if desired; preserve actual non-daemon-thread subprocess tests and both script entrypoint wiring checks.
3. Fix random-board materialization and cross-test imports while preserving the sampled corpus first. Measure collection separately.
4. Retire legacy trace hashes where exact historical scheduling is not a product contract; shrink wrapper/config matrices and recipe assertions. Keep readable semantic failures.
5. Establish a normal CPU suite and a documented extended/compiler/CUDA run. Ensure the extended run actually gets executed; there is no checked-in `.github` workflow currently enforcing it.

The terminal replacement and four immediately redundant subprocess cases target about **42.8 seconds of the 111.8-second run**, before replacement-test overhead. That is a credible first reduction without touching the 1,820 native cases. Further subprocess consolidation and separating compiler/large-sweep work make an **under-one-minute default CPU suite a reasonable target**, not a verified result yet. Report elapsed time, collection time and retained contracts after each change; a lower case count alone is not success.
