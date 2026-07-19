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

    `legal_vocab_ids` (profile-driven thin-down, see
    `docs/superpowers/sdd/thin-report.md`): the worker -- which holds the
    real python-chess board for the current position -- computes the
    UCI-sorted, vocab-mapped legal-move projection itself
    (`actor_worker._legal_vocab_projection`, converting to a cozy board
    first) and sends only the resulting vocab ids. The server never
    reconstructs a board or runs movegen at all anymore: it just gathers raw
    logits at these ids, in this order, and returns them unprojected (see
    `RootEvalResponse`). The worker keeps its own parallel
    (`cc.Move`, uci) lists locally so it never needs move/uci strings back
    on the wire.
    """

    worker_id: int
    turn_id: int
    batch_arrays: dict
    legal_vocab_ids: list[int]


@dataclass
class RootEvalResponse:
    """server -> worker: root forward result, UNPROJECTED.

    `legal_logits` is the raw (pre-softmax) logit gathered at each of the
    request's own `legal_vocab_ids`, in that same order -- the server does a
    single batched index op per call (no per-request Python movegen/sort
    loop), and the worker computes log-softmax itself
    (`actor_worker._log_softmax_f32`) since it already knows which
    moves/ucis those ids correspond to.
    """

    turn_id: int
    value_stm: float
    legal_logits: list[float]


@dataclass
class WaveRow:
    """One search-tree node's decode request, worker-minted.

    `board_state` is `vars(BoardState)` (`data/models.py`): plain
    ints/lists, chess-free -- used ONLY to tensorize this node's one new
    decode token (piece placement / turn / castle / ep / clock buckets);
    the server no longer reconstructs a board object from it at all (no
    movegen happens server-side anymore).

    `legal_vocab_ids` is this node's own UCI-sorted, vocab-mapped legal-move
    projection, computed worker-side from the REAL cozy board for this node
    (`actor_worker._legal_vocab_projection`) -- see `RootEvalRequest`'s
    docstring for the full rationale; identical division of labor, just
    per-node instead of per-turn.
    """

    node_id: int
    parent_id: int | None  # None = child of the turn's root prefix
    prev_move_vocab_id: int
    board_state: dict
    legal_vocab_ids: list[int]


@dataclass
class WaveRequest:
    """worker -> server: one decode wave (one batched `EvalRequest` from a
    `search.py` `*_stepwise` generator), rows in request order."""

    worker_id: int
    turn_id: int
    rows: list[WaveRow]


@dataclass
class WaveResponse:
    """server -> worker: one `(value_stm, legal_logits)` pair per `WaveRow`,
    in the same order as `WaveRequest.rows`. `legal_logits` is the raw
    logits gathered at that row's own `legal_vocab_ids` (one padded batched
    `torch.gather` per wave server-side, not a per-row Python loop) -- the
    worker computes log-softmax itself, same division of labor as
    `RootEvalResponse`."""

    rows: list[tuple[float, list[float]]]


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
