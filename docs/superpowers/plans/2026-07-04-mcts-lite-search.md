# MCTS-lite (Sequential Halving) Search + Strategy Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate all move-selection strategies from `scripts/eval_vs_stockfish.py` into `src/imba_chess/eval/search.py` behind a `PositionEvaluator` interface, and add a new `value_search_halving` strategy (sequential halving at the root, beam-by-prior tree growth, negamax backup) per `docs/superpowers/specs/2026-07-04-mcts-lite-search-design.md`.

**Architecture:** The new module is torch-free: strategies consume a `PositionEvaluator` protocol (opaque `handle` + batched `evaluate`) and plain floats/`chess.Move`s. The eval script keeps all model plumbing (`_SequenceHistory`, jagged batching, chunked forwards) and exposes it as one adapter class. Existing rerank/d2 tests run end-to-end through the script and gate the extraction; the halving algorithm gets direct unit tests with scripted dummy evaluators (no torch model needed).

**Tech Stack:** Python, `python-chess`, `heapq`/`math`/`itertools` (stdlib), `pytest`. Torch only in the script-side adapter.

## Global Constraints

- No new third-party dependencies. `src/imba_chess/eval/search.py` must NOT import torch.
- Existing tests in `tests/test_eval_vs_stockfish.py` must pass **unchanged** — they are the regression gate for moving rerank/d2 out of the script.
- Debug-row key names preserved verbatim: value_rerank rows `{move_uci, policy_logit, policy_log_prob, value_next, terminal, rerank_score}`; d2 rows `{move_uci, policy_logit, policy_log_prob, worst_reply_value, best_reply_uci, search_score}`. (`policy_logit` values become log-priors — no test or trace formatting depends on raw-logit values.)
- Config defaults verbatim from the spec: `search_budget = 256`, `search_top_m = 16`, `halving_rounds = 0` (0 = auto `ceil(log2(num_arms))`), `search_refutation_top_r = 2`, `search_expand_top = 3`, `search_max_depth = 4`. Halving reuses `value_rerank_lambda` as its λ.
- Determinism: no sampling anywhere; ties break by insertion order (`itertools.count` heap tiebreaker).
- Value never selects within the tree: frontier ordering is by cumulative policy log-prior only; value enters only at backup/arm scoring.
- Python: `.venv/bin/python` (all test commands below use it).

---

### Task 1: `search.py` foundation — protocol, config, shared helpers, greedy

**Files:**
- Create: `src/imba_chess/eval/search.py`
- Test: `tests/test_search.py` (new)

**Interfaces:**
- Consumes: nothing (new module; only `chess` + stdlib).
- Produces (used by Tasks 2–4):
  - `PositionEval(value_stm: float, legal_moves: list[chess.Move], legal_log_priors: list[float])` NamedTuple
  - `PositionEvaluator` Protocol with `extend(handle, board_before, move) -> Any` and `evaluate(batch: list[tuple[Any, chess.Board]]) -> list[PositionEval]`
  - `HalvingConfig(budget=256, top_m=16, rounds=0, refutation_top_r=2, expand_top=3, max_depth=4, lam=0.05)` frozen dataclass
  - `terminal_value_for_color(board, *, color) -> float | None`
  - `select_greedy(legal_log_priors: list[float]) -> int`
  - `_auto_rounds(num_arms: int) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_search.py`:

```python
from __future__ import annotations

import chess

from imba_chess.eval.search import (
    HalvingConfig,
    _auto_rounds,
    select_greedy,
    terminal_value_for_color,
)


def test_select_greedy_returns_argmax_index_first_on_ties():
    assert select_greedy([-2.0, -0.5, -1.0]) == 1
    assert select_greedy([-1.0, -1.0]) == 0


def test_terminal_value_for_color():
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/7K b - - 0 1")  # black is mated
    assert terminal_value_for_color(mated, color=chess.WHITE) == 1.0
    assert terminal_value_for_color(mated, color=chess.BLACK) == -1.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # black stalemated
    assert terminal_value_for_color(stalemate, color=chess.WHITE) == 0.0
    ongoing = chess.Board()
    assert terminal_value_for_color(ongoing, color=chess.WHITE) is None


def test_auto_rounds():
    assert _auto_rounds(16) == 4
    assert _auto_rounds(2) == 1
    assert _auto_rounds(3) == 2
    assert _auto_rounds(1) == 1
    assert _auto_rounds(17) == 5


def test_halving_config_defaults_match_spec():
    config = HalvingConfig()
    assert config.budget == 256
    assert config.top_m == 16
    assert config.rounds == 0
    assert config.refutation_top_r == 2
    assert config.expand_top == 3
    assert config.max_depth == 4
    assert config.lam == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.eval.search'`.

- [ ] **Step 3: Write the module foundation**

Create `src/imba_chess/eval/search.py`:

```python
"""Move-selection strategies for eval play, decoupled from the model.

Strategies consume a PositionEvaluator: `handle` is opaque (the eval script
uses a _SequenceHistory clone; tests use whatever they need), `extend` derives
the handle for the position after a move, and `evaluate` batch-scores
positions, returning the value-head scalar (side-to-move POV) plus the legal
moves that map to the move vocab and their log-softmax policy priors.

This module must stay torch-free so strategy unit tests need no model.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Protocol

import chess


class PositionEval(NamedTuple):
    value_stm: float
    legal_moves: list[chess.Move]
    legal_log_priors: list[float]


class PositionEvaluator(Protocol):
    def extend(
        self, handle: Any, board_before: chess.Board, move: chess.Move
    ) -> Any: ...

    def evaluate(
        self, batch: list[tuple[Any, chess.Board]]
    ) -> list[PositionEval]: ...


@dataclass(frozen=True)
class HalvingConfig:
    budget: int = 256
    top_m: int = 16
    rounds: int = 0  # 0 = auto ceil(log2(num_arms))
    refutation_top_r: int = 2
    expand_top: int = 3
    max_depth: int = 4
    lam: float = 0.05


def _auto_rounds(num_arms: int) -> int:
    return max(1, math.ceil(math.log2(max(2, num_arms))))


def terminal_value_for_color(
    board: chess.Board, *, color: chess.Color
) -> Optional[float]:
    if not board.is_game_over(claim_draw=True):
        return None
    result = board.result(claim_draw=True)
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if color == chess.BLACK else -1.0
    return 0.0


def select_greedy(legal_log_priors: list[float]) -> int:
    return max(range(len(legal_log_priors)), key=legal_log_priors.__getitem__)


def _is_forcing(board: chess.Board, move: chess.Move) -> bool:
    return (
        move.promotion is not None
        or board.is_capture(move)
        or board.gives_check(move)
    )


def _prior_order(legal_log_priors: list[float]) -> list[int]:
    return sorted(
        range(len(legal_log_priors)),
        key=legal_log_priors.__getitem__,
        reverse=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/search.py tests/test_search.py
git commit -m "feat: add search strategy module foundation (evaluator protocol, config, greedy)"
```

---

### Task 2: Move value_rerank + value_search_d2 into the module; script adapter + dispatch

**Files:**
- Modify: `src/imba_chess/eval/search.py` (append strategies)
- Modify: `scripts/eval_vs_stockfish.py` (adapter, dispatch, delete moved code)
- Test: `tests/test_eval_vs_stockfish.py` (NO changes — it is the gate)

**Interfaces:**
- Consumes (Task 1): `PositionEval`, `PositionEvaluator`, `terminal_value_for_color`, `select_greedy`, `_prior_order`, `_is_forcing`.
- Produces:
  - `select_value_rerank(*, evaluator, root_handle, board, legal_moves, legal_log_priors, top_k: int, lam: float) -> tuple[int, list[dict]]`
  - `select_value_search_d2(*, evaluator, root_handle, board, legal_moves, legal_log_priors, top_k: int, lam: float) -> tuple[int, list[dict]]`
  - Script-side `_HistoryPositionEvaluator` (adapter; used again in Task 4's dispatch of halving).

The moved strategies are behavior-preserving rewrites against `PositionEvaluator`: batching structure is identical (rerank: root + 1 candidate batch; d2: root + 1 board1 batch + 1 board2 batch), so the dummy-model `forward_calls` assertions in the existing tests (2, 3, 2, 1) still hold.

- [ ] **Step 1: Confirm the existing tests pass before touching anything (baseline)**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py -v`
Expected: PASS (all 11 tests). Record the count.

- [ ] **Step 2: Append the two strategies to `src/imba_chess/eval/search.py`**

```python
def select_value_rerank(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
    lam: float,
) -> tuple[int, list[dict[str, Any]]]:
    root_color = board.turn
    local_indices = _prior_order(legal_log_priors)[: min(top_k, len(legal_moves))]

    candidates: list[dict[str, Any]] = []
    batch: list[tuple[Any, chess.Board]] = []
    batch_to_candidate: list[int] = []
    for idx in local_indices:
        candidate_move = legal_moves[idx]
        # Keep the move stack so claimable draws (repetition/50-move) count as terminal.
        next_board = board.copy()
        next_board.push(candidate_move)
        # Terminal boards never appear as training tokens; use the exact game
        # result instead of the value head there.
        terminal_value = terminal_value_for_color(next_board, color=root_color)
        candidates.append(
            {
                "local_idx": idx,
                "move": candidate_move,
                "terminal": terminal_value is not None,
                "value_root": terminal_value,
            }
        )
        if terminal_value is not None:
            continue
        batch.append((evaluator.extend(root_handle, board, candidate_move), next_board))
        batch_to_candidate.append(len(candidates) - 1)

    if batch:
        for cand_idx, position_eval in zip(batch_to_candidate, evaluator.evaluate(batch)):
            # Side-to-move at next_board is the opponent; negate to root POV.
            candidates[cand_idx]["value_root"] = -position_eval.value_stm

    chosen_index = local_indices[0]
    best_score = float("-inf")
    rerank_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        local_idx = int(candidate["local_idx"])
        value_root = float(candidate["value_root"])
        log_prior = float(legal_log_priors[local_idx])
        # Value-dominant score with a small log-prob policy prior as tiebreak.
        rerank_score = value_root + (lam * log_prior)
        rerank_rows.append(
            {
                "move_uci": candidate["move"].uci(),
                "policy_logit": log_prior,
                "policy_log_prob": log_prior,
                "value_next": value_root,
                "terminal": bool(candidate["terminal"]),
                "rerank_score": rerank_score,
            }
        )
        if rerank_score > best_score:
            best_score = rerank_score
            chosen_index = local_idx

    return chosen_index, rerank_rows


def select_value_search_d2(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    top_k: int,
    lam: float,
) -> tuple[int, list[dict[str, Any]]]:
    root_color = board.turn
    local_indices = _prior_order(legal_log_priors)[: min(top_k, len(legal_moves))]

    root_candidates: list[dict[str, Any]] = []
    board1_batch: list[tuple[Any, chess.Board]] = []
    board1_batch_to_root: list[int] = []
    for local_idx in local_indices:
        move = legal_moves[local_idx]
        board1 = board.copy()
        board1.push(move)
        terminal_value = terminal_value_for_color(board1, color=root_color)
        if terminal_value is not None and terminal_value >= 1.0:
            # Immediate win (checkmate delivered): no other move can score higher.
            return local_idx, [
                {
                    "move_uci": move.uci(),
                    "policy_logit": float(legal_log_priors[local_idx]),
                    "policy_log_prob": float(legal_log_priors[local_idx]),
                    "worst_reply_value": 1.0,
                    "best_reply_uci": None,
                    "search_score": 1.0,
                }
            ]
        root_candidate: dict[str, Any] = {
            "local_idx": local_idx,
            "move": move,
            "board1": board1,
            "log_prior": float(legal_log_priors[local_idx]),
            "terminal_value": terminal_value,
            "board1_eval": None,
            "handle1": None,
            "reply_candidates": [],
        }
        root_candidates.append(root_candidate)
        if terminal_value is not None:
            continue
        handle1 = evaluator.extend(root_handle, board, move)
        root_candidate["handle1"] = handle1
        board1_batch.append((handle1, board1))
        board1_batch_to_root.append(len(root_candidates) - 1)

    if board1_batch:
        for root_idx, position_eval in zip(
            board1_batch_to_root, evaluator.evaluate(board1_batch)
        ):
            root_candidates[root_idx]["board1_eval"] = position_eval

    board2_batch: list[tuple[Any, chess.Board]] = []
    board2_meta: list[tuple[int, int]] = []
    for root_idx, root_candidate in enumerate(root_candidates):
        if root_candidate["terminal_value"] is not None:
            continue
        board1_eval: Optional[PositionEval] = root_candidate["board1_eval"]
        if board1_eval is None:
            continue
        board1 = root_candidate["board1"]
        if not board1_eval.legal_moves:
            root_candidate["worst_reply_value"] = -float(board1_eval.value_stm)
            continue

        opp_indices = _prior_order(board1_eval.legal_log_priors)[
            : min(top_k, len(board1_eval.legal_moves))
        ]
        # Always consider forcing replies (captures/checks/promotions): the
        # tactical refutation is often a low-probability move under a
        # human-imitation policy, so policy top-k alone misses it.
        opp_seen = set(opp_indices)
        for opp_idx, opp_move in enumerate(board1_eval.legal_moves):
            if opp_idx in opp_seen:
                continue
            if _is_forcing(board1, opp_move):
                opp_indices.append(opp_idx)
                opp_seen.add(opp_idx)

        reply_rows: list[dict[str, Any]] = []
        for opp_local_idx in opp_indices:
            opp_move = board1_eval.legal_moves[opp_local_idx]
            board2 = board1.copy()
            board2.push(opp_move)
            terminal_value = terminal_value_for_color(board2, color=root_color)
            reply_rows.append(
                {
                    "move_uci": opp_move.uci(),
                    "opp_policy_logit": float(
                        board1_eval.legal_log_priors[opp_local_idx]
                    ),
                    "value_after_reply": terminal_value,
                    "terminal": terminal_value is not None,
                }
            )
            if terminal_value is not None:
                continue
            board2_batch.append(
                (evaluator.extend(root_candidate["handle1"], board1, opp_move), board2)
            )
            board2_meta.append((root_idx, len(reply_rows) - 1))
        root_candidate["reply_candidates"] = reply_rows

    if board2_batch:
        for (root_idx, reply_idx), position_eval in zip(
            board2_meta, evaluator.evaluate(board2_batch)
        ):
            # Side-to-move at board2 is the root color again: POV matches root.
            root_candidates[root_idx]["reply_candidates"][reply_idx][
                "value_after_reply"
            ] = float(position_eval.value_stm)

    chosen_index = local_indices[0]
    best_score = float("-inf")
    search_rows: list[dict[str, Any]] = []
    for root_candidate in root_candidates:
        if root_candidate["terminal_value"] is not None:
            worst_reply_value = float(root_candidate["terminal_value"])
            best_reply_uci = None
        elif "worst_reply_value" in root_candidate:
            worst_reply_value = float(root_candidate["worst_reply_value"])
            best_reply_uci = None
        else:
            reply_rows = [
                row
                for row in root_candidate["reply_candidates"]
                if row.get("value_after_reply") is not None
            ]
            if not reply_rows:
                board1_eval = root_candidate.get("board1_eval")
                worst_reply_value = (
                    -float(board1_eval.value_stm) if board1_eval is not None else 0.0
                )
                best_reply_uci = None
            else:
                best_reply = min(
                    reply_rows, key=lambda row: float(row["value_after_reply"])
                )
                worst_reply_value = float(best_reply["value_after_reply"])
                best_reply_uci = str(best_reply["move_uci"])

        log_prior = float(root_candidate["log_prior"])
        # Value-dominant score with a small log-prob policy prior as tiebreak.
        search_score = worst_reply_value + (lam * log_prior)
        search_rows.append(
            {
                "move_uci": root_candidate["move"].uci(),
                "policy_logit": log_prior,
                "policy_log_prob": log_prior,
                "worst_reply_value": worst_reply_value,
                "best_reply_uci": best_reply_uci,
                "search_score": search_score,
            }
        )
        if search_score > best_score:
            best_score = search_score
            chosen_index = int(root_candidate["local_idx"])

    return chosen_index, search_rows
```

- [ ] **Step 3: Add the adapter to `scripts/eval_vs_stockfish.py` and rewire dispatch**

3a. Add the import (after the existing `from imba_chess.eval.game_animation import render_game_html` line):

```python
from imba_chess.eval.search import (
    PositionEval,
    select_greedy,
    select_value_rerank,
    select_value_search_d2,
)
```

3b. Delete these three functions from the script entirely: `_value_scalar_from_logits` stays; delete `_terminal_value_for_color`, `_select_value_rerank_index`, `_select_value_search_d2_index`.

3c. Insert the adapter class immediately after `_forward_last_token_outputs`:

```python
class _HistoryPositionEvaluator:
    """PositionEvaluator over _SequenceHistory handles and the chunked forward."""

    def __init__(self, *, model, move_vocab, device, dtype, policy_name: str) -> None:
        self._model = model
        self._move_vocab = move_vocab
        self._device = device
        self._dtype = dtype
        self._policy_name = policy_name

    def extend(self, handle, board_before: chess.Board, move: chess.Move):
        new_handle = handle.clone()
        new_handle.append_observed_position(board_before)
        new_handle.record_played_move(move.uci())
        return new_handle

    def evaluate(self, batch):
        batches = [
            handle.build_batch_for_current_position(board) for handle, board in batch
        ]
        policy_rows, value_rows = _forward_last_token_outputs(
            model=self._model,
            batches=batches,
            device=self._device,
            dtype=self._dtype,
            policy_name=self._policy_name,
        )
        results = []
        for row_idx, (_, board) in enumerate(batch):
            value_stm = _value_scalar_from_logits(value_rows[row_idx])
            try:
                legal_logits, legal_moves, _, _ = _project_legal_logits(
                    logits=policy_rows[row_idx],
                    board=board,
                    move_vocab=self._move_vocab,
                )
                log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
            except RuntimeError:
                # No legal move maps to the vocab: value-only leaf.
                legal_moves, log_priors = [], []
            results.append(
                PositionEval(
                    value_stm=value_stm,
                    legal_moves=legal_moves,
                    legal_log_priors=log_priors,
                )
            )
        return results
```

3d. Rewrite the body of `_select_model_move` between the `_project_legal_logits` call and the `debug` dict construction. The current code:

```python
    rerank_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    if policy == "greedy":
        chosen_index = int(torch.argmax(legal_logits).item())
    elif policy == "value_rerank":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_rerank requires a checkpoint with value head enabled."
            )
        chosen_index, rerank_rows = _select_value_rerank_index(
            model=model,
            history=history,
            board=board,
            legal_logits=legal_logits,
            legal_moves_with_ids=legal_moves_with_ids,
            device=device,
            dtype=dtype,
            value_rerank_top_k=value_rerank_top_k,
            value_rerank_lambda=value_rerank_lambda,
        )
    elif policy == "value_search_d2":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_d2 requires a checkpoint with value head enabled."
            )
        chosen_index, search_rows = _select_value_search_d2_index(
            model=model,
            history=history,
            board=board,
            legal_logits=legal_logits,
            legal_moves_with_ids=legal_moves_with_ids,
            move_vocab=move_vocab,
            device=device,
            dtype=dtype,
            value_rerank_top_k=value_rerank_top_k,
            value_rerank_lambda=value_rerank_lambda,
        )
    else:
        raise ValueError(f"Unknown model move policy: {policy}")
```

becomes:

```python
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    evaluator = _HistoryPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        device=device,
        dtype=dtype,
        policy_name=policy,
    )
    rerank_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    if policy == "greedy":
        chosen_index = select_greedy(legal_log_priors)
    elif policy == "value_rerank":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_rerank requires a checkpoint with value head enabled."
            )
        chosen_index, rerank_rows = select_value_rerank(
            evaluator=evaluator,
            root_handle=history,
            board=board,
            legal_moves=legal_moves_with_ids,
            legal_log_priors=legal_log_priors,
            top_k=value_rerank_top_k,
            lam=value_rerank_lambda,
        )
    elif policy == "value_search_d2":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_d2 requires a checkpoint with value head enabled."
            )
        chosen_index, search_rows = select_value_search_d2(
            evaluator=evaluator,
            root_handle=history,
            board=board,
            legal_moves=legal_moves_with_ids,
            legal_log_priors=legal_log_priors,
            top_k=value_rerank_top_k,
            lam=value_rerank_lambda,
        )
    else:
        raise ValueError(f"Unknown model move policy: {policy}")
```

Also in the `debug_topk` block below it, replace `f"{entry['logit']:.3f}"` sourcing: change `top_values, top_indices = torch.topk(legal_logits, k=k, largest=True)` usage to keep working as-is (it may stay on raw logits — do not change that block).

- [ ] **Step 4: Run the gate — existing tests must pass unchanged**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py tests/test_search.py -v`
Expected: PASS, same test count as Step 1 plus Task 1's 4. If any `forward_calls` assertion fails, the adapter is batching differently from the original — fix the adapter, not the test.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/eval/search.py scripts/eval_vs_stockfish.py
git commit -m "refactor: move rerank/d2 strategies behind PositionEvaluator in eval.search"
```

---

### Task 3: `select_value_search_halving` + unit tests

**Files:**
- Modify: `src/imba_chess/eval/search.py` (append)
- Test: `tests/test_search.py` (append)

**Interfaces:**
- Consumes (Task 1): `HalvingConfig`, `PositionEval`, `terminal_value_for_color`, `_auto_rounds`, `_is_forcing`, `_prior_order`.
- Produces: `select_value_search_halving(*, evaluator, root_handle, board, legal_moves, legal_log_priors, config: HalvingConfig) -> tuple[int, list[dict]]`. Debug rows have keys `{move_uci, policy_log_prob, evals_spent, max_depth, backed_value, search_score, eliminated_round}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
import math

from imba_chess.eval.search import (
    PositionEval,
    select_value_search_halving,
)

_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_stm(board: chess.Board) -> float:
    """Material balance from the side-to-move's POV, scaled to roughly [-1, 1]."""
    total = 0
    for piece in board.piece_map().values():
        sign = 1 if piece.color == board.turn else -1
        total += sign * _PIECE_VALUES[piece.piece_type]
    return max(-1.0, min(1.0, total / 10.0))


class _ArmValueEvaluator:
    """Value depends only on which root move started the line (handle[0]).

    value_stm flips sign with ply parity so negamax backup returns exactly
    the arm's root-POV value at any depth. Priors are uniform.
    """

    def __init__(self, arm_values_root_pov: dict[str, float]) -> None:
        self.arm_values = arm_values_root_pov
        self.eval_calls = 0
        self.positions_evaluated = 0

    def extend(self, handle, board_before, move):
        return (handle or ()) + (move,)

    def evaluate(self, batch):
        self.eval_calls += 1
        self.positions_evaluated += len(batch)
        results = []
        for handle, board in batch:
            value_root_pov = self.arm_values[handle[0].uci()]
            stm_is_root_side = len(handle) % 2 == 0
            value_stm = value_root_pov if stm_is_root_side else -value_root_pov
            moves = list(board.legal_moves)
            log_prior = math.log(1.0 / len(moves)) if moves else 0.0
            results.append(PositionEval(value_stm, moves, [log_prior] * len(moves)))
        return results


class _MaterialEvaluator:
    """Material-count value; priors rank captures/checks/promotions LAST."""

    def __init__(self) -> None:
        self.eval_calls = 0
        self.positions_evaluated = 0

    def extend(self, handle, board_before, move):
        return (handle or ()) + (move,)

    def evaluate(self, batch):
        self.eval_calls += 1
        self.positions_evaluated += len(batch)
        results = []
        for handle, board in batch:
            moves = list(board.legal_moves)
            priors = [
                -5.0
                if (m.promotion is not None or board.is_capture(m) or board.gives_check(m))
                else -0.1
                for m in moves
            ]
            results.append(PositionEval(_material_stm(board), moves, priors))
        return results


def test_halving_eliminates_low_value_arm_and_spends_exact_budget():
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    legal_log_priors = [-0.5, -0.6]
    evaluator = _ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6})
    config = HalvingConfig(budget=8, top_m=2, rounds=2, lam=0.05)

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )

    assert legal_moves[chosen].uci() == "e2e4"
    assert evaluator.positions_evaluated == 8  # exact budget
    by_move = {row["move_uci"]: row for row in rows}
    assert by_move["d2d4"]["eliminated_round"] == 0
    assert by_move["e2e4"]["eliminated_round"] is None
    # Round-2 budget flowed to the survivor.
    assert by_move["e2e4"]["evals_spent"] > by_move["d2d4"]["evals_spent"]
    assert by_move["e2e4"]["backed_value"] > by_move["d2d4"]["backed_value"]


def test_rounds_one_is_pure_beam_no_elimination():
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    evaluator = _ArmValueEvaluator({"e2e4": 0.6, "d2d4": -0.6})
    config = HalvingConfig(budget=8, top_m=2, rounds=1, lam=0.05)

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=[-0.5, -0.6],
        config=config,
    )

    assert legal_moves[chosen].uci() == "e2e4"
    assert all(row["eliminated_round"] is None for row in rows)
    assert all(row["evals_spent"] > 0 for row in rows)
    assert evaluator.positions_evaluated <= 8


def test_mate_in_one_short_circuits_with_zero_evals():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    legal_moves = [chess.Move.from_uci("a1b1"), chess.Move.from_uci("a1a8")]
    legal_log_priors = [-0.1, -3.0]  # mate is LOW prior
    evaluator = _ArmValueEvaluator({"a1b1": 0.0, "a1a8": 0.0})
    config = HalvingConfig(budget=16, top_m=2, rounds=2)

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )

    assert legal_moves[chosen].uci() == "a1a8"
    assert evaluator.eval_calls == 0
    assert len(rows) == 1 and rows[0]["search_score"] == 1.0


def test_refutation_floor_catches_low_prior_forcing_refutation():
    # White Qd2, black Nb4. Qd3?? hangs the queen to Nxd3 — a capture the
    # priors rank last. Qh6 is safe. Only the forcing-reply floor finds Nxd3.
    board = chess.Board("k7/8/8/8/1n6/8/3Q4/K7 w - - 0 1")
    legal_moves = [chess.Move.from_uci("d2h6"), chess.Move.from_uci("d2d3")]
    legal_log_priors = [-0.7, -0.7]
    evaluator = _MaterialEvaluator()
    config = HalvingConfig(
        budget=8, top_m=2, rounds=1, refutation_top_r=1, expand_top=2, max_depth=2
    )

    chosen, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=(),
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=config,
    )

    assert legal_moves[chosen].uci() == "d2h6"
    by_move = {row["move_uci"]: row for row in rows}
    assert by_move["d2d3"]["backed_value"] < by_move["d2h6"]["backed_value"]
    assert evaluator.positions_evaluated <= 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: the 4 new tests FAIL with `ImportError: cannot import name 'select_value_search_halving'`.

- [ ] **Step 3: Implement the algorithm**

Append to `src/imba_chess/eval/search.py`:

```python
@dataclass
class _TreeNode:
    board: chess.Board
    handle: Any
    depth: int  # plies below the arm root (arm root = 0)
    path_log_prior: float
    value_stm: Optional[float] = None  # set when evaluated by the value head
    terminal_value_stm: Optional[float] = None  # exact, side-to-move POV
    children: list["_TreeNode"] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.value_stm is not None or self.terminal_value_stm is not None


@dataclass
class _Arm:
    local_idx: int
    move: chess.Move
    root_log_prior: float
    root_node: Optional[_TreeNode]
    terminal_value_root: Optional[float]
    frontier: list = field(default_factory=list)
    evals_spent: int = 0
    max_depth_reached: int = 0
    eliminated_round: Optional[int] = None
    backed_value: Optional[float] = None
    score: float = float("-inf")


def _backed_stm(node: _TreeNode) -> float:
    """Negamax over the realized (partially scored) tree, side-to-move POV."""
    if node.terminal_value_stm is not None:
        return node.terminal_value_stm
    child_values = [-_backed_stm(child) for child in node.children if child.scored]
    if child_values:
        return max(child_values)
    assert node.value_stm is not None
    return node.value_stm


def _score_arm(arm: _Arm, lam: float) -> None:
    if arm.terminal_value_root is not None:
        backed_root = float(arm.terminal_value_root)
    elif arm.root_node is not None and arm.root_node.scored:
        backed_root = -_backed_stm(arm.root_node)
    else:
        arm.backed_value = None
        arm.score = float("-inf")
        return
    arm.backed_value = backed_root
    arm.score = backed_root + lam * arm.root_log_prior


def _push_children(
    arm: _Arm,
    node: _TreeNode,
    position_eval: PositionEval,
    evaluator: PositionEvaluator,
    config: HalvingConfig,
    counter: "itertools.count",
    root_color: chess.Color,
) -> None:
    if node.depth >= config.max_depth or not position_eval.legal_moves:
        return
    opponent_to_move = node.board.turn != root_color
    order = _prior_order(position_eval.legal_log_priors)
    if opponent_to_move:
        # Refutation floor: top-r replies by prior plus ALL forcing replies.
        picks = list(order[: config.refutation_top_r])
        seen = set(picks)
        for idx, move in enumerate(position_eval.legal_moves):
            if idx not in seen and _is_forcing(node.board, move):
                picks.append(idx)
                seen.add(idx)
    else:
        picks = list(order[: config.expand_top])

    for idx in picks:
        move = position_eval.legal_moves[idx]
        child_board = node.board.copy()
        child_board.push(move)
        # Forcing replies inherit the parent's priority (no decay for their
        # own low prior): a refutation must compete at the plausibility of
        # the line it refutes, not of the reply itself.
        floor_pick = opponent_to_move and _is_forcing(node.board, move)
        child_prior = node.path_log_prior + (
            0.0 if floor_pick else position_eval.legal_log_priors[idx]
        )
        child = _TreeNode(
            board=child_board,
            handle=None,
            depth=node.depth + 1,
            path_log_prior=child_prior,
        )
        terminal_stm = terminal_value_for_color(child_board, color=child_board.turn)
        if terminal_stm is not None:
            child.terminal_value_stm = terminal_stm
            node.children.append(child)
            continue
        child.handle = evaluator.extend(node.handle, node.board, move)
        node.children.append(child)
        heapq.heappush(arm.frontier, (-child.path_log_prior, next(counter), child))


def select_value_search_halving(
    *,
    evaluator: PositionEvaluator,
    root_handle: Any,
    board: chess.Board,
    legal_moves: list[chess.Move],
    legal_log_priors: list[float],
    config: HalvingConfig,
) -> tuple[int, list[dict[str, Any]]]:
    root_color = board.turn
    order = _prior_order(legal_log_priors)
    picks = list(order[: min(config.top_m, len(order))])
    seen = set(picks)
    for idx, move in enumerate(legal_moves):
        if idx not in seen and _is_forcing(board, move):
            picks.append(idx)
            seen.add(idx)

    counter = itertools.count()
    arms: list[_Arm] = []
    for idx in picks:
        move = legal_moves[idx]
        board1 = board.copy()
        board1.push(move)
        terminal_root = terminal_value_for_color(board1, color=root_color)
        if terminal_root is not None and terminal_root >= 1.0:
            # Immediate win (checkmate delivered): no other move can score higher.
            return idx, [
                {
                    "move_uci": move.uci(),
                    "policy_log_prob": float(legal_log_priors[idx]),
                    "evals_spent": 0,
                    "max_depth": 0,
                    "backed_value": 1.0,
                    "search_score": 1.0,
                    "eliminated_round": None,
                }
            ]
        arm = _Arm(
            local_idx=idx,
            move=move,
            root_log_prior=float(legal_log_priors[idx]),
            root_node=None,
            terminal_value_root=terminal_root,
        )
        if terminal_root is None:
            node = _TreeNode(
                board=board1,
                handle=evaluator.extend(root_handle, board, move),
                depth=0,
                path_log_prior=float(legal_log_priors[idx]),
            )
            arm.root_node = node
            heapq.heappush(arm.frontier, (-node.path_log_prior, next(counter), node))
        arms.append(arm)

    rounds = config.rounds if config.rounds > 0 else _auto_rounds(len(arms))
    spent = 0
    survivors = list(arms)
    for round_idx in range(rounds):
        active = [arm for arm in survivors if arm.frontier]
        if not active or spent >= config.budget:
            break
        per_arm = max(
            1, (config.budget - spent) // ((rounds - round_idx) * len(active))
        )
        remaining = {id(arm): per_arm for arm in active}
        # Waves: pop -> batched evaluate -> expand, until the round budget is
        # spent or frontiers empty. One batched evaluate per wave (per level).
        while spent < config.budget:
            wave: list[tuple[_Arm, _TreeNode]] = []
            for arm in active:
                take = min(
                    remaining[id(arm)],
                    len(arm.frontier),
                    config.budget - spent - len(wave),
                )
                for _ in range(max(0, take)):
                    _, _, node = heapq.heappop(arm.frontier)
                    wave.append((arm, node))
                    remaining[id(arm)] -= 1
            if not wave:
                break
            evals = evaluator.evaluate([(node.handle, node.board) for _, node in wave])
            spent += len(wave)
            for (arm, node), position_eval in zip(wave, evals):
                node.value_stm = float(position_eval.value_stm)
                arm.evals_spent += 1
                arm.max_depth_reached = max(arm.max_depth_reached, node.depth)
                _push_children(
                    arm, node, position_eval, evaluator, config, counter, root_color
                )
        for arm in survivors:
            _score_arm(arm, config.lam)
        if round_idx < rounds - 1 and len(survivors) > 1:
            survivors.sort(key=lambda arm: arm.score, reverse=True)
            keep = math.ceil(len(survivors) / 2)
            for arm in survivors[keep:]:
                arm.eliminated_round = round_idx
            survivors = survivors[:keep]

    for arm in arms:
        _score_arm(arm, config.lam)
        # Preserve elimination bookkeeping; _score_arm only sets score/backed.

    best = max(survivors, key=lambda arm: arm.score)
    if best.score == float("-inf"):
        # Budget starvation: fall back to the highest-prior candidate.
        best = arms[0]

    rows = [
        {
            "move_uci": arm.move.uci(),
            "policy_log_prob": arm.root_log_prior,
            "evals_spent": arm.evals_spent,
            "max_depth": arm.max_depth_reached,
            "backed_value": arm.backed_value
            if arm.terminal_value_root is None
            else float(arm.terminal_value_root),
            "search_score": None if arm.score == float("-inf") else arm.score,
            "eliminated_round": arm.eliminated_round,
        }
        for arm in arms
    ]
    return best.local_idx, rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS (8 tests). If `test_halving_eliminates_low_value_arm_and_spends_exact_budget` fails on the exact-8 assertion, check the wave loop's budget cap — pops must stop exactly at `config.budget`.

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/eval/search.py tests/test_search.py
git commit -m "feat: add value_search_halving (sequential halving / beam) strategy"
```

---

### Task 4: Config, CLI, dispatch wiring, integration test

**Files:**
- Modify: `src/imba_chess/config.py` (EvalVsStockfishConfig)
- Modify: `config/imba_chess.toml` ([eval_vs_stockfish])
- Modify: `scripts/eval_vs_stockfish.py` (CLI args, resolution, validation, dispatch, debug trace)
- Test: `tests/test_eval_vs_stockfish.py` (append integration test), `tests/test_config.py` (append defaults test)

**Interfaces:**
- Consumes: `select_value_search_halving`, `HalvingConfig` (Task 3); `_HistoryPositionEvaluator` (Task 2).
- Produces: policy name `"value_search_halving"` usable via config/CLI end-to-end.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_eval_vs_stockfish_search_knob_defaults():
    config = EvalVsStockfishConfig()
    assert config.search_budget == 256
    assert config.search_top_m == 16
    assert config.halving_rounds == 0
    assert config.search_refutation_top_r == 2
    assert config.search_expand_top == 3
    assert config.search_max_depth == 4
```

Append to `tests/test_eval_vs_stockfish.py`:

```python
class _DummyHalvingModel(torch.nn.Module):
    """Root policy prefers e2e4; value head says the d2d4 subtree is winning.

    Value is read from the side-to-move POV, so the dummy uses the last
    token's turn_id to keep the signal consistent at every depth.
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        seq_offsets = batch["seq_offsets"]
        prev_move_id = batch["prev_move_id"]
        turn_id = batch["turn_id"]
        d2d4_id = self.move_vocab.token_to_id["d2d4"]
        for game_idx in range(int(batch["num_games"])):
            start = int(seq_offsets[game_idx].item())
            end = int(seq_offsets[game_idx + 1].item())
            last = end - 1
            logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
            logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
            contains_d2d4 = bool((prev_move_id[start:end] == d2d4_id).any().item())
            good_for_white = contains_d2d4
            stm_is_white = int(turn_id[last].item()) == 0
            if good_for_white == stm_is_white:
                value_logits[last] = torch.tensor([0.0, 0.0, 3.0])  # stm winning
            else:
                value_logits[last] = torch.tensor([3.0, 0.0, 0.0])  # stm losing
        return {"logits": logits, "value_logits": value_logits}


def test_value_search_halving_end_to_end_picks_value_backed_move():
    module = _load_eval_script_module()
    from imba_chess.eval.search import HalvingConfig

    move_vocab = _mini_vocab()
    model = _DummyHalvingModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)

    move, debug = module._select_model_move(
        model=model,
        batch=batch,
        history=history,
        board=board,
        move_vocab=move_vocab,
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_search_halving",
        value_rerank_top_k=2,
        value_rerank_lambda=0.05,
        debug_topk=0,
        halving_config=HalvingConfig(budget=6, top_m=2, rounds=2, lam=0.05),
    )

    assert move.uci() == "d2d4"  # higher root logit is e2e4; value flips it
    assert debug["policy"] == "value_search_halving"
    rows = debug["value_search_halving_candidates"]
    assert {row["move_uci"] for row in rows} == {"e2e4", "d2d4"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_eval_vs_stockfish.py -v -k "search_knob or halving"`
Expected: config test FAILS with `AttributeError: ... no attribute 'search_budget'`; integration test FAILS with `TypeError` (unexpected keyword `halving_config`) or `ValueError: Unknown model move policy`.

- [ ] **Step 3: Add config fields**

In `src/imba_chess/config.py`, `EvalVsStockfishConfig`, after `value_rerank_lambda: float = 0.35`:

```python
    search_budget: int = 256
    search_top_m: int = 16
    halving_rounds: int = 0
    search_refutation_top_r: int = 2
    search_expand_top: int = 3
    search_max_depth: int = 4
```

In `config/imba_chess.toml`, `[eval_vs_stockfish]`, after the `value_rerank_lambda` line:

```toml
# value_search_halving knobs (sequential halving; halving_rounds = 1 is pure beam).
search_budget = 256
search_top_m = 16
halving_rounds = 0
search_refutation_top_r = 2
search_expand_top = 3
search_max_depth = 4
```

- [ ] **Step 4: Wire the script**

4a. Extend the search-module import (from Task 2) to:

```python
from imba_chess.eval.search import (
    HalvingConfig,
    PositionEval,
    select_greedy,
    select_value_rerank,
    select_value_search_d2,
    select_value_search_halving,
)
```

4b. `_parse_args`: change `--model-move-policy` choices to
`["greedy", "value_rerank", "value_search_d2", "value_search_halving"]`, and add after the `--value-rerank-lambda` argument:

```python
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--search-top-m", type=int, default=None)
    parser.add_argument("--halving-rounds", type=int, default=None)
    parser.add_argument("--search-refutation-top-r", type=int, default=None)
    parser.add_argument("--search-expand-top", type=int, default=None)
    parser.add_argument("--search-max-depth", type=int, default=None)
```

4c. In `main()`, after the `args.value_rerank_lambda` resolution block:

```python
    args.search_budget = int(
        eval_cfg.search_budget if args.search_budget is None else args.search_budget
    )
    args.search_top_m = int(
        eval_cfg.search_top_m if args.search_top_m is None else args.search_top_m
    )
    args.halving_rounds = int(
        eval_cfg.halving_rounds if args.halving_rounds is None else args.halving_rounds
    )
    args.search_refutation_top_r = int(
        eval_cfg.search_refutation_top_r
        if args.search_refutation_top_r is None
        else args.search_refutation_top_r
    )
    args.search_expand_top = int(
        eval_cfg.search_expand_top
        if args.search_expand_top is None
        else args.search_expand_top
    )
    args.search_max_depth = int(
        eval_cfg.search_max_depth
        if args.search_max_depth is None
        else args.search_max_depth
    )
```

4d. In `main()`'s validation block, update the policy set:

```python
    if args.model_move_policy not in {
        "greedy",
        "value_rerank",
        "value_search_d2",
        "value_search_halving",
    }:
        raise ValueError(
            "--model-move-policy must be one of: greedy, value_rerank, "
            "value_search_d2, value_search_halving"
        )
    if args.search_budget < 1:
        raise ValueError("--search-budget must be >= 1")
    if args.search_top_m < 1:
        raise ValueError("--search-top-m must be >= 1")
    if args.halving_rounds < 0:
        raise ValueError("--halving-rounds must be >= 0")
    if args.search_refutation_top_r < 1:
        raise ValueError("--search-refutation-top-r must be >= 1")
    if args.search_expand_top < 1:
        raise ValueError("--search-expand-top must be >= 1")
    if args.search_max_depth < 1:
        raise ValueError("--search-max-depth must be >= 1")
```

Also update `require_value_head` in the `_load_model` call:

```python
        require_value_head=(
            str(args.model_move_policy)
            in {"value_rerank", "value_search_d2", "value_search_halving"}
        ),
```

4e. `_select_model_move`: add parameter `halving_config: Optional[HalvingConfig] = None` (after `debug_topk`), and add the dispatch branch before the final `else`:

```python
    elif policy == "value_search_halving":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_halving requires a checkpoint with value head enabled."
            )
        if halving_config is None:
            raise ValueError("policy=value_search_halving requires halving_config")
        chosen_index, halving_rows = select_value_search_halving(
            evaluator=evaluator,
            root_handle=history,
            board=board,
            legal_moves=legal_moves_with_ids,
            legal_log_priors=legal_log_priors,
            config=halving_config,
        )
```

Initialize `halving_rows: list[dict[str, Any]] = []` next to `search_rows`, and add to the debug dict construction:

```python
    if policy == "value_search_halving":
        debug["search_budget"] = int(halving_config.budget)
        debug["value_search_halving_candidates"] = halving_rows
```

4f. `_run_segment`: add parameter `halving_config: "HalvingConfig | None" = None` and pass it through to `_select_model_move(... halving_config=halving_config)`. In the debug-trace block after the `value_search_d2` printing, add:

```python
                        halving_rows = debug_info.get("value_search_halving_candidates")
                        if isinstance(halving_rows, list) and halving_rows:
                            halving_str = ", ".join(
                                f"{entry['move_uci']}:evals={entry['evals_spent']}"
                                f"|backed={entry['backed_value']}"
                                f"|score={entry['search_score']}"
                                f"|out_r={entry['eliminated_round']}"
                                for entry in halving_rows
                            )
                            tqdm.write(
                                f"[debug][{segment_name}]   value_search_halving={halving_str}"
                            )
```

4g. In `main()`'s `_run_segment(...)` call site, add:

```python
                halving_config=HalvingConfig(
                    budget=int(args.search_budget),
                    top_m=int(args.search_top_m),
                    rounds=int(args.halving_rounds),
                    refutation_top_r=int(args.search_refutation_top_r),
                    expand_top=int(args.search_expand_top),
                    max_depth=int(args.search_max_depth),
                    lam=float(args.value_rerank_lambda),
                ),
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_eval_vs_stockfish.py tests/test_search.py -v`
Expected: PASS (all, including the new integration + config tests).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/imba_chess/config.py config/imba_chess.toml scripts/eval_vs_stockfish.py tests/test_config.py tests/test_eval_vs_stockfish.py
git commit -m "feat: wire value_search_halving policy through config/CLI/eval loop"
```

---

## Post-implementation (manual, not part of the plan)

A/B per the spec: `POLICIES="value_search_halving" ./eval_best_checkpoint.sh` on
`best_hr10_checkpoint_6_hr10=0.9131.pt` (100 games vs SF1400, seed 42).
Decision rule: ≥ 0.39 → depth pays (build prefix caching next); ≤ 0.34 → pivot
to value distillation. Extra sweep point: `--halving-rounds 1` (pure beam).
