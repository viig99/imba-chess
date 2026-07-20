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


@dataclass(kw_only=True)
class RootEvalRequest:
    """worker -> server: one game-turn's root forward, in one of two forms
    (incremental-root-KV optimization; see
    `docs/superpowers/sdd/increm-report.md`):

    FULL form (`batch_arrays` set, `incremental_tokens` None) -- the
    game's FIRST root request: `batch_arrays` is the torch-free twin of
    `_SequenceHistory`'s `_build_single_batch()` dict
    (`position_evaluator.py`): every tensor field there becomes a plain
    nested list here, scalar ints stay ints. See `actor_worker.
    _PlainSequenceHistory.build_batch_for_current_position`. The server
    runs a full-sequence forward and PERSISTS the resulting prefix KV,
    keyed by `worker_id` alone (a worker plays exactly one game at a time,
    so `worker_id` unambiguously identifies "this worker's current game")
    -- overwriting any prefix already persisted for that worker (a fresh
    game always starts fresh, whether or not the previous game's prefix
    was explicitly released first).

    INCREMENTAL form (`incremental_tokens` set, `batch_arrays` None) --
    every SUBSEQUENT root request of the same game: `incremental_tokens`
    carries only the NEW board-state tokens committed to the worker's
    history since its previous root request (normally 2 -- this worker's
    own prior move settling into history, then the opponent's reply -- but
    ALWAYS derived from the actual history-length delta, never assumed,
    since opening plies / this worker's model playing black / etc. can
    change the count; see `actor_worker._PlainSequenceHistory.
    build_incremental_tokens_for_current_position`). Same field set as one
    wave's `new_token_batch` (`WaveRow.board_state`'s encoded fields, here
    as parallel lists instead of one dict per row): `piece_ids`,
    `seq_token_id`, `turn_id`, `castle_id`, `ep_file_id`,
    `halfmove_bucket_id`, `fullmove_bucket_id`, `prev_move_id`, each a
    list of length k (k = number of new tokens), in sequence order -- the
    server extends its persisted per-worker prefix KV by running
    `hstu_model.HSTUChessModel.forward_decode` (single-prefix decode) once
    per new token, sequentially, folding each token's own returned KV
    directly into the persisted prefix (see `actor_server.
    _service_incremental_root`); the LAST token's logits/value_logits are
    the root eval output, exactly as the full forward's last-token output
    would be. `prefix_len_before` is this worker's own record of what the
    server's persisted prefix length should be BEFORE this extension (from
    `_PlainSequenceHistory.server_prefix_len`) -- a fail-fast desync check,
    not itself load-bearing for the extension math (the server tracks its
    own authoritative prefix length).

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

    Construction is KEYWORD-ONLY (`@dataclass(kw_only=True)`): `batch_arrays`
    used to be the 3rd required positional field, before `incremental_tokens`/
    `prefix_len_before` existed; every real caller already constructs this
    with keywords (worker/server/tests), but a positional call written from
    muscle memory of the old signature would otherwise silently bind the
    wrong argument to the wrong field instead of raising -- `__post_init__`
    below only checks `is not None`, which a swapped dict/list argument can
    still satisfy. Keyword-only construction turns that into an immediate
    `TypeError` instead.
    """

    worker_id: int
    turn_id: int
    legal_vocab_ids: list[int]
    batch_arrays: dict | None = None
    incremental_tokens: dict | None = None
    prefix_len_before: int | None = None

    def __post_init__(self) -> None:
        has_full = self.batch_arrays is not None
        has_incremental = self.incremental_tokens is not None
        if has_full == has_incremental:  # both set, or neither
            raise ValueError(
                "RootEvalRequest must set exactly ONE of batch_arrays (full "
                "forward -- this game's first root request) or "
                "incremental_tokens (incremental root extension -- every "
                f"subsequent request), got batch_arrays="
                f"{'<set>' if has_full else None}, incremental_tokens="
                f"{'<set>' if has_incremental else None}."
            )
        if has_incremental and self.prefix_len_before is None:
            raise ValueError(
                "RootEvalRequest.incremental_tokens requires "
                "prefix_len_before (the worker's own record of the "
                "server's persisted prefix length before this extension) "
                "-- needed for the server's fail-fast desync check."
            )


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
