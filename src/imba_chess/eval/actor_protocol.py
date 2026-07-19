"""Wire protocol between torch-free eval actor workers and the GPU inference
server (`docs/superpowers/specs/2026-07-19-multiprocess-eval-actors-design.md`).

Plain dataclasses only: every field is a plain Python type (int/float/str/
bool/None/list/dict), so instances are picklable across a
`multiprocessing.Pipe()` without either endpoint needing torch or a chess
library to unpickle them (this module itself imports neither). Message
direction is documented per class; `run_eval_worker`
(`src/imba_chess/eval/actor_worker.py`) is the only producer/consumer of
worker-side messages, `ActorInferenceServer` (Task 2) the server side.

Node identity: `WaveRow.node_id`/`parent_id` are worker-minted integers,
scoped to one `(worker_id, turn_id)` search tree -- mirrors
`position_evaluator._CachedNode`'s parent-link semantics (see
`actor_worker._WorkerSearchNode`), just flattened to plain ints instead of
Python object references so they survive the pipe. `parent_id is None`
means "this node's decode suffix is empty: it hangs directly off the turn's
root prefix KV" (the same case `_CachedNode.__init__(parent=None, ...)`
encodes for the root-adjacent evaluator handle in
`CachedPositionEvaluator.extend`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RootEvalRequest:
    """worker -> server: one game-turn's root forward.

    `batch_arrays` is the torch-free twin of `_SequenceHistory`'s
    `_build_single_batch()` dict (`position_evaluator.py`): every tensor
    field there becomes a plain nested list here, scalar ints stay ints. See
    `actor_worker._PlainSequenceHistory.build_batch_for_current_position`.
    """

    worker_id: int
    turn_id: int
    batch_arrays: dict


@dataclass
class RootEvalResponse:
    """server -> worker: root forward result, already vocab-projected.

    `legal_ucis`/`legal_log_priors` are index-aligned and UCI-sorted (the
    same canonical order `position_evaluator._project_legal_logits_cozy`
    produces) -- the server owns projection since it owns the logits.
    """

    turn_id: int
    value_stm: float
    legal_ucis: list[str]
    legal_log_priors: list[float]


@dataclass
class WaveRow:
    """One search-tree node's decode request, worker-minted.

    `board_state` is `vars(BoardState)` (`data/models.py`): plain
    ints/lists, chess-free -- the server reconstructs whatever it needs
    (a cozy board, for legal-move projection) from these fields rather than
    receiving a board object across the pipe.
    """

    node_id: int
    parent_id: int | None  # None = child of the turn's root prefix
    prev_move_vocab_id: int
    board_state: dict


@dataclass
class WaveRequest:
    """worker -> server: one decode wave (one batched `EvalRequest` from a
    `search.py` `*_stepwise` generator), rows in request order."""

    worker_id: int
    turn_id: int
    rows: list[WaveRow]


@dataclass
class WaveResponse:
    """server -> worker: one `(value_stm, legal_ucis, legal_log_priors)`
    tuple per `WaveRow`, in the same order as `WaveRequest.rows`."""

    rows: list[tuple[float, list[str], list[float]]]


@dataclass
class GameDone:
    """worker -> server: one just-finished game's summary-counter fragment.

    `summary_fragment` is `dataclasses.asdict()` of the worker's own
    torch-free `_EvalSummaryFragment` (`actor_worker.py`) -- field-for-field
    the same counters as `scripts/eval_vs_stockfish.py`'s `EvalSummary`, so
    the orchestrator can fold it into a running total the same way
    `_accumulate_summary` does there.
    """

    worker_id: int
    game_idx: int
    summary_fragment: dict


@dataclass
class WorkerFinished:
    """worker -> server: sent once after every assigned game_idx is done,
    immediately before the worker's engine is closed and the process exits."""

    worker_id: int
