"""GPU inference server for the multiprocess eval actors design
(`docs/superpowers/specs/2026-07-19-multiprocess-eval-actors-design.md`,
profile-driven thin-down: `docs/superpowers/sdd/thin-report.md`).

`ActorInferenceServer` is the main-process counterpart of the torch-free
`run_eval_worker` (`actor_worker.py`): it owns the model and ALL KV state,
keyed by `(worker_id, turn_id)` for root prefixes and per-node arena rows
for decode-wave nodes (`actor_protocol.py`'s wire messages mint these as
plain ints precisely so they survive the pipe without a board or a
`_CachedNode` object ever crossing it).

Profile-driven thin-down (cProfile evidence in the report above; the
short version): a 20-game/543s server-side cProfile showed three
per-ROW Python bottlenecks that this rewrite removes:
  1. `_project_legal_logits_cozy` run server-side per row (107s cum) --
     ELIMINATED. The worker now computes the UCI-sorted, vocab-mapped
     legal-move projection itself (it holds the real board; see
     `actor_worker._legal_vocab_projection`) and sends only
     `legal_vocab_ids`; the server does one padded batched
     `torch.gather` per wave/root call instead (`_gather_legal_logits`).
  2. Reconstructing a cozy board from the wire `BoardState`, then
     RE-ENCODING it via `encode_cozy` inside `CachedPositionEvaluator.
     build_decode_request` (58s cum combined) -- ELIMINATED. The wire
     `BoardState` fields ARE the encoded fields; `_tensorize_wave_rows`
     feeds them directly into the decode token batch. No board is ever
     reconstructed server-side anymore -- `cozy_chess` is not even
     imported by this module.
  3. Per-node KV-chain `torch.cat`s inside `consume_decode_result`
     (4.2M calls, 19s self) -- ELIMINATED. `_KVArena` preallocates one
     growable `[L, H, capacity, d]` tensor per (worker_id, turn_id); each
     node's own decode-token KV is written ONCE at its arena row, and a
     wave's suffix (each row's full root->parent ancestor chain) is built
     via ONE indexed gather (`_KVArena.gather_suffix`) from cheap
     append-only Python `list[int]` ancestor-index chains
     (`_TurnState.node_chains`) instead of a tensor concatenation per node
     per depth.
  4. `_value_scalar_from_logits` called per row in Python (2M calls, 13s
     self) -- ELIMINATED. `_batched_value_stm` runs one softmax over the
     whole wave's/root batch's value_logits and `.tolist()`s once.

Composition vs. reuse: the ragged root-batch merge/split
(`merged_executors._merge_root_batches`/`_split_root_output`) and the
model-forward helpers (`position_evaluator._forward_model`/
`_autocast_context`) are imported and used UNMODIFIED -- zero edits to
either `merged_executors.py` or `position_evaluator.py`; both stay
byte-identical for their existing rollout-shared callers.
`CachedPositionEvaluator`/`_CachedNode`/`_project_legal_logits_cozy` are no
longer used by this module at all (the composition they used to provide --
per-node KV chains, board reconstruction + projection -- is exactly what
`_KVArena`/the worker-side projection replace), so this module no longer
imports them, `cozy_chess`, or `BoardStateEncoder` either.

`merged_executors._merge_decode_requests`/`_split_decode_output` (the
cross-worker decode-wave merge/pad math) ARE reused unmodified too, via a
small duck-typed `_ArenaDecodeRequest` -- those two functions only ever
read `.nodes` (for its length), `.new_token_batch`, `.positions`,
`.suffix_kv`/`.suffix_positions`/`.suffix_mask`, `.prefix_kv`, `.prefix_len`
off whatever object they're given (no `isinstance` check), so this module's
own arena-backed request object satisfies that interface without either
module needing to know about the other's node representation.

Incremental-root-KV optimization (`docs/superpowers/sdd/increm-report.md`):
wall-clock server profiling showed root evals -- a FULL-sequence forward
over the whole game so far, EVERY model turn -- as the #1 GPU cost.
`RootEvalRequest` now has two forms (see its own docstring): FULL (this
game's first turn; unchanged full-forward path, `_service_full_roots`,
still cross-worker batched via `_merge_root_batches`) and INCREMENTAL
(every later turn; `_service_incremental_root`, one sequential single-
prefix `forward_decode` call per new token, folded directly into a NEW
per-worker persisted prefix, `_GameState`, keyed by `worker_id` alone and
released per-GAME by the orchestrator on `GameDone`/`WorkerFinished` --
distinct from `_TurnState`, still released per-TURN as before for the
decode-wave search tree).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import time

import torch

from imba_chess.data.event_builder import EVENT_TOKEN_ID
from imba_chess.eval.actor_protocol import (
    RootEvalRequest,
    RootEvalResponse,
    WaveRequest,
    WaveResponse,
)
from imba_chess.eval.merged_executors import (
    _merge_decode_requests,
    _merge_root_batches,
    _split_decode_output,
    _split_root_output,
)
from imba_chess.eval.position_evaluator import _autocast_context, _forward_model

# The plain tensor-batch fields RootEvalRequest.batch_arrays carries -- same
# field set _SequenceHistory._build_single_batch() (position_evaluator.py)
# and _PlainSequenceHistory._build_single_batch() (actor_worker.py) both
# produce, just torch-tensorized here on receipt.
_ROOT_BATCH_INT_LIST_FIELDS = (
    "seq_lens",
    "seq_offsets",
    "piece_ids",
    "seq_token_id",
    "game_result_white",
    "turn_id",
    "castle_id",
    "ep_file_id",
    "halfmove_bucket_id",
    "fullmove_bucket_id",
    "prev_move_id",
    "target_move_id",
    "played_by_elo",
)

def _tensorize_root_batch(batch_arrays: dict) -> dict[str, Any]:
    """`RootEvalRequest.batch_arrays` (plain lists/ints, torch-free on the
    wire) -> the torch tensor dict `_merge_root_batches`/`_forward_model`
    expect -- the same shape `_SequenceHistory._build_single_batch()`
    produces directly, just built from lists instead of appended to
    incrementally."""
    out: dict[str, Any] = {
        "game_id": list(batch_arrays["game_id"]),
        "num_games": int(batch_arrays["num_games"]),
        "total_tokens": int(batch_arrays["total_tokens"]),
    }
    for key in _ROOT_BATCH_INT_LIST_FIELDS:
        out[key] = torch.tensor(batch_arrays[key], dtype=torch.long)
    return out


def _ensure_value_logits_placeholder(output: dict[str, Any]) -> dict[str, Any]:
    """When the server's model has no value head (`ActorInferenceServer(
    require_value_head=False, ...)` -- see that constructor's own
    docstring), `output` (from `_forward_model`/`forward_decode_grouped`)
    never has a `"value_logits"` key at all -- the model only sets it `if
    self.value_head is not None` (`model/hstu_model.py`). But Task 1's wire
    protocol makes `RootEvalResponse.value_stm`/every `WaveResponse` row's
    value_stm UNCONDITIONAL fields, and the REUSED tensor-math helpers this
    server calls (`_split_root_output`, `_merge_decode_requests`/
    `_split_decode_output`) both assume a same-shaped tensor is available at
    that key wherever it's threaded through -- rather than editing those
    shared, rollout-consumed functions to make that key optional, this
    injects an explicit all-zero placeholder of the right shape in place.
    softmax(zeros) is the uniform distribution, so `_batched_value_stm`
    (`probs[:, 2] - probs[:, 0]`) evaluates to exactly `0.0` for every row: a
    deliberate, documented "no opinion" placeholder, not a real value
    estimate. `require_value_head=False` is only valid for the `greedy`
    policy (the caller's own gate -- see
    `scripts/eval_vs_stockfish.py`'s `_run_segment_actor_mode`), which never
    reads `value_stm` at all, so this placeholder is never actually
    consumed by anything downstream; it exists purely so the reused
    merge/split machinery always has a tensor of the shape it expects.
    No-op (returns `output` unchanged) when a real `"value_logits"` key is
    already present."""
    if "value_logits" not in output:
        logits = output["logits"]
        output["value_logits"] = torch.zeros(
            (logits.shape[0], 3), device=logits.device, dtype=torch.float32
        )
    return output


def _batched_value_stm(value_logits: torch.Tensor) -> list[float]:
    """One softmax + one `.tolist()` over a WHOLE wave's/root batch's
    `value_logits` -- replaces `position_evaluator._value_scalar_from_logits`
    called per row in Python (profile motivation #4: 2M calls, 13s self)."""
    probs = torch.softmax(value_logits.float(), dim=-1)
    return (probs[:, 2] - probs[:, 0]).tolist()


def _gather_legal_logits(
    logits: torch.Tensor, legal_vocab_ids_per_row: list[list[int]]
) -> list[list[float]]:
    """One padded batched `torch.gather` for a whole wave/root call --
    replaces the per-row `_project_legal_logits_cozy` server-side movegen +
    `index_select` this design deletes entirely (profile motivation #1: 107s
    cum across 50M calls). `logits` is `[B, V]`; row i's own
    `legal_vocab_ids_per_row[i]` may have a different length than other
    rows' (a different number of legal moves per position), so ids are
    padded to this call's own max count (padding value 0 -- always a valid
    vocab index, and the padded output columns are sliced off per row
    below, so the padding value's actual logit is never read by anyone)."""
    max_k = max((len(ids) for ids in legal_vocab_ids_per_row), default=0)
    if max_k == 0:
        return [[] for _ in legal_vocab_ids_per_row]
    ids_tensor = torch.zeros(
        (len(legal_vocab_ids_per_row), max_k), dtype=torch.long, device=logits.device
    )
    for row, ids in enumerate(legal_vocab_ids_per_row):
        if ids:
            ids_tensor[row, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=logits.device
            )
    gathered = torch.gather(logits, 1, ids_tensor)
    return [
        gathered[row, : len(ids)].tolist()
        for row, ids in enumerate(legal_vocab_ids_per_row)
    ]


class _KVArena:
    """Growable per-(worker_id, turn_id) decode-token KV store:
    `k`/`v` each `[L, H, capacity, d]`. Each search-tree node's own
    one-token decode KV is written ONCE, at an arena row minted by
    `append`; a node's descendants retrieve their suffix (the full
    root->parent ancestor chain) via `gather_suffix`, one indexed gather for
    a whole wave instead of a `torch.cat` per node per depth
    (`_TurnState.node_chains` holds the cheap append-only Python
    `list[int]` chains `gather_suffix`'s caller builds the index matrix
    from).

    Capacity grows by doubling (`_ensure_capacity`), amortizing the
    reallocation cost across a whole turn's worth of decode waves; lazily
    created on the turn's FIRST decode wave (`_get_or_create_arena`) from
    that wave's own KV output shape, since the server never knows a turn's
    eventual node count in advance.
    """

    __slots__ = ("k", "v", "size")

    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v
        self.size = 0

    def _ensure_capacity(self, extra: int) -> None:
        needed = self.size + extra
        capacity = self.k.shape[2]
        if needed <= capacity:
            return
        new_capacity = max(capacity, 1)
        while new_capacity < needed:
            new_capacity *= 2
        new_k = self.k.new_zeros((self.k.shape[0], self.k.shape[1], new_capacity, self.k.shape[3]))
        new_v = self.v.new_zeros((self.v.shape[0], self.v.shape[1], new_capacity, self.v.shape[3]))
        new_k[:, :, : self.size, :] = self.k[:, :, : self.size, :]
        new_v[:, :, : self.size, :] = self.v[:, :, : self.size, :]
        self.k = new_k
        self.v = new_v

    def append(self, k_rows: torch.Tensor, v_rows: torch.Tensor) -> list[int]:
        """`k_rows`/`v_rows`: `[L, H, n, d]`, one row per new node, IN WAVE
        ORDER. Returns the n arena row indices assigned, same order."""
        n = k_rows.shape[2]
        self._ensure_capacity(n)
        start = self.size
        self.k[:, :, start : start + n, :] = k_rows
        self.v[:, :, start : start + n, :] = v_rows
        self.size += n
        return list(range(start, start + n))

    def gather_suffix(
        self, idx: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """`idx`: `[B, S]` long tensor of arena rows (padding positions may
        hold any in-bounds row -- caller's own `suffix_mask` is what makes
        those positions inert to the model, not the value gathered here).
        Returns the per-layer `[(k, v), ...]` list `forward_decode_grouped`
        expects, each `[B, H, S, d]` -- ONE indexed gather plus one permute
        for the whole wave, not a per-node `torch.cat` chain."""
        gathered_k = self.k[:, :, idx, :].permute(0, 2, 1, 3, 4)  # [L, B, H, S, d]
        gathered_v = self.v[:, :, idx, :].permute(0, 2, 1, 3, 4)
        return list(zip(gathered_k.unbind(0), gathered_v.unbind(0)))


def _get_or_create_arena(
    arena: _KVArena | None, k_rows: torch.Tensor, v_rows: torch.Tensor
) -> _KVArena:
    if arena is not None:
        return arena
    capacity = max(int(k_rows.shape[2]), 16)
    return _KVArena(
        k_rows.new_zeros((k_rows.shape[0], k_rows.shape[1], capacity, k_rows.shape[3])),
        v_rows.new_zeros((v_rows.shape[0], v_rows.shape[1], capacity, v_rows.shape[3])),
    )


@dataclass
class _TurnState:
    """Everything the server keeps for one live `(worker_id, turn_id)`:
    the root forward's per-layer prefix KV (`[H, T, d]` each, from
    `_forward_model(..., return_kv=True)`), that prefix's token length
    (decode positions are `prefix_len + depth`), the turn's lazily-created
    `_KVArena`, and each already-evaluated node's own ancestor-index chain
    (`node_chains[node_id]` = arena rows from the shallowest wave-decode
    node through this node itself, root->self order -- exactly the token
    span a CHILD of this node needs as its decode suffix).

    `prefix_kv`/`prefix_len` here are a SNAPSHOT of the worker's `_GameState`
    at this turn's root-request time (same tensor objects, not a copy --
    see `_GameState`'s own docstring for why sharing is safe): a later
    turn's incremental extension rebinds `_GameState.prefix_kv` to a NEW
    concatenated list rather than mutating the old one in place, so an
    already-created `_TurnState`'s snapshot is never retroactively changed
    by a later turn's growth."""

    prefix_kv: Any
    prefix_len: int
    arena: _KVArena | None = None
    node_chains: dict[int, list[int]] = field(default_factory=dict)


@dataclass
class _GameState:
    """Persistent per-worker prefix KV, spanning a whole game ACROSS model
    turns (incremental-root-KV optimization; see
    `docs/superpowers/sdd/increm-report.md`) -- distinct from `_TurnState`,
    which is scoped to one turn's decode-wave search tree and still
    released every turn as before.

    Keyed by `worker_id` ALONE (not `(worker_id, turn_id)`): a worker plays
    exactly one game at a time (`actor_worker._play_one_game` runs games
    sequentially in a loop), so `worker_id` unambiguously identifies "this
    worker's current game" -- `turn_id` keeps incrementing across a
    worker's whole lifetime (many games), so it plays no role in this key.

    `prefix_kv`/`prefix_len` grow monotonically via `_service_incremental_
    root`'s sequential `forward_decode` extension, which REBINDS this
    attribute to a freshly `torch.cat`-built list each time rather than
    mutating the old tensors in place -- so any `_TurnState` that captured
    an earlier snapshot stays valid. A FULL `RootEvalRequest` (this
    worker's next game) unconditionally overwrites the whole entry via
    dict assignment (`ActorInferenceServer._service_full_roots`), the same
    "same key re-registration overwrites" discipline `_TurnState`'s own
    docstring documents for `(worker_id, turn_id)`."""

    prefix_kv: Any
    prefix_len: int


def _snapshot_turn_state(game: _GameState) -> _TurnState:
    """Builds a turn's `_TurnState` from its worker's CURRENT `_GameState`
    -- shared by `_service_full_roots` (this game's first turn) and
    `_service_incremental_root` (every later one), so the "snapshot the
    game's prefix at root-request time" logic lives in exactly one place.
    Shares the tensor objects (no copy) -- see `_TurnState`'s own docstring
    for why that's safe (extension rebinds `_GameState.prefix_kv` rather
    than mutating it in place)."""
    return _TurnState(prefix_kv=game.prefix_kv, prefix_len=game.prefix_len)


@dataclass
class _ArenaDecodeRequest:
    """Duck-typed stand-in for `position_evaluator._DecodeRequest`, built
    from `_KVArena` data instead of `_CachedNode` path_kv chains -- see this
    module's own docstring for why this satisfies
    `merged_executors._merge_decode_requests`/`_split_decode_output`
    unmodified. `nodes` only needs a `len()` (those two functions never read
    its elements), so a plain `range` stands in for the `_CachedNode` list
    those functions were originally written against."""

    nodes: range
    new_token_batch: dict[str, Any]
    positions: torch.Tensor
    suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None
    suffix_positions: torch.Tensor | None
    suffix_mask: torch.Tensor | None
    prefix_kv: Any
    prefix_len: int


def _tensorize_wave_rows(
    rows: list, turn: _TurnState
) -> tuple[dict[str, Any], torch.Tensor, list[list[int]]]:
    """Builds this wave's decode-token batch DIRECTLY from each row's wire
    `board_state` dict (already the encoded fields -- no board is ever
    reconstructed or re-encoded, profile motivation #2) plus its own
    `prev_move_vocab_id`; also returns each row's parent ancestor-index
    chain (empty list = child of the turn's root prefix), looked up from
    `turn.node_chains` -- the CPU-side prep `_KVArena.gather_suffix`'s
    caller needs to build this wave's suffix index matrix."""
    piece_ids, turn_ids, castle_ids, ep_ids = [], [], [], []
    halfmove_ids, fullmove_ids, prev_move_ids = [], [], []
    parent_chains: list[list[int]] = []
    for row in rows:
        bs = row.board_state
        piece_ids.append(bs["piece_ids"])
        turn_ids.append(int(bs["turn_id"]))
        castle_ids.append(int(bs["castle_id"]))
        ep_ids.append(int(bs["ep_file_id"]))
        halfmove_ids.append(int(bs["halfmove_bucket_id"]))
        fullmove_ids.append(int(bs["fullmove_bucket_id"]))
        prev_move_ids.append(int(row.prev_move_vocab_id))
        parent_chains.append(
            turn.node_chains[row.parent_id] if row.parent_id is not None else []
        )
    wave_size = len(rows)
    new_token_batch = {
        "piece_ids": torch.tensor(piece_ids, dtype=torch.long),
        "seq_token_id": torch.full((wave_size,), EVENT_TOKEN_ID, dtype=torch.long),
        "turn_id": torch.tensor(turn_ids, dtype=torch.long),
        "castle_id": torch.tensor(castle_ids, dtype=torch.long),
        "ep_file_id": torch.tensor(ep_ids, dtype=torch.long),
        "halfmove_bucket_id": torch.tensor(halfmove_ids, dtype=torch.long),
        "fullmove_bucket_id": torch.tensor(fullmove_ids, dtype=torch.long),
        "prev_move_id": torch.tensor(prev_move_ids, dtype=torch.long),
    }
    depths = [len(chain) for chain in parent_chains]
    positions = torch.tensor([turn.prefix_len + d for d in depths], dtype=torch.long)
    return new_token_batch, positions, parent_chains


class ActorInferenceServer:
    """Owns the model and the ID-keyed KV store for every worker's in-flight
    game turns.

    `release_turn` is an explicit method, not a wire message -- the
    orchestrator, which lives in the SAME process as this server (no pipe
    needed for this control signal), is expected to call it once it
    observes a turn is over: either a worker's next `RootEvalRequest`
    (implying the previous turn's search finished) or its `WorkerFinished`
    (implying the last turn of the worker's last game finished). This
    module does not infer that on its own -- an un-released turn's KV
    simply persists until either explicitly released or the same
    `(worker_id, turn_id)` key is overwritten by a same-turn
    re-registration (defensive; should not happen given the worker's
    per-worker monotonic turn_id counter) -- so a caller that forgets to
    call it leaks memory across turns, but never silently corrupts a
    DIFFERENT turn's results (every lookup is scoped by the exact key).
    `release_turn` itself is idempotent: releasing an already-released or
    never-registered key is a no-op, not an error.
    """

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        dtype: torch.dtype,
        require_value_head: bool = True,
        profile_sync: bool = False,
    ) -> None:
        # Wall-clock service buckets, always accumulated (perf_counter cost is
        # negligible); `profile_sync` additionally cuda-synchronizes at bucket
        # boundaries so GPU vs pre/post CPU attribution is honest -- enable
        # only for diagnosis (IMBA_ACTOR_PROFILE=1), it serializes the device.
        self.profile_sync = bool(profile_sync)
        self.stats: dict[str, float] = {
            "root_build_s": 0.0, "root_gpu_s": 0.0, "root_post_s": 0.0,
            "wave_build_s": 0.0, "wave_gpu_s": 0.0, "wave_post_s": 0.0,
            "root_calls": 0, "root_reqs": 0,
            "wave_calls": 0, "wave_reqs": 0, "wave_rows": 0,
            # Incremental-root-KV optimization: a DISTINCT bucket from
            # "root_*" above (which stays scoped to the FULL-forward,
            # merged-batch path only) -- see _service_incremental_root's own
            # comment on why mixing the two units would corrupt profiling.
            "incremental_root_gpu_s": 0.0,
            "incremental_root_calls": 0, "incremental_root_reqs": 0,
            "incremental_root_tokens": 0,
        }
        # Validated ONCE here, not per request. RootEvalResponse.value_stm
        # and every WaveResponse row's value_stm are UNCONDITIONAL fields
        # (populated regardless of the worker's own model_move_policy, which
        # the server never even sees), so by default
        # (`require_value_head=True`, matching the G=1 path's own
        # `load_hstu_checkpoint(require_value_head=...)` gate for
        # value-dependent policies) a model with no value head can never
        # serve a single request. `require_value_head=False` -- the
        # orchestrator passes this exactly when `model_move_policy ==
        # "greedy"`, the only policy that never reads value_stm -- instead
        # allows construction with a value-head-less model; every response's
        # value_stm is then a documented `0.0` placeholder (see
        # `_ensure_value_logits_placeholder`), never a real value estimate.
        if require_value_head and getattr(model, "value_head", None) is None:
            raise ValueError(
                "ActorInferenceServer(require_value_head=True) requires a "
                "model with an enabled value head: every RootEvalResponse/"
                "WaveResponse row carries a value_stm field unconditionally, "
                "regardless of which worker's model_move_policy asked for it "
                "(the server never sees that policy). Got a model whose "
                ".value_head is None -- either enable_value_head=False at "
                "construction, or a checkpoint with no value_head.* "
                "parameters (see position_evaluator.load_hstu_checkpoint's "
                "require_value_head for the load-time analogue of this "
                "guard). Pass require_value_head=False instead if this run "
                "only ever uses model_move_policy='greedy' (the only policy "
                "that never reads value_stm) -- every response's value_stm "
                "will then be a documented 0.0 placeholder, not a real "
                "value estimate."
            )
        self._model = model
        self._device = device
        self._dtype = dtype

        # (worker_id, turn_id) -> that turn's state (prefix KV + KV arena +
        # node ancestor-index chains). Replaces the pre-thin-down design's
        # CachedPositionEvaluator + {node_id: _CachedNode} pairing -- see
        # this module's own docstring for the profiling motivation.
        self._turns: dict[tuple[int, int], _TurnState] = {}

        # worker_id -> that worker's CURRENT game's persisted prefix KV
        # (incremental-root-KV optimization; see _GameState's own
        # docstring). Released on GameDone/WorkerFinished by the
        # orchestrator (scripts/eval_vs_stockfish.py's _serve_actor_workers),
        # same explicit-release discipline as self._turns/release_turn.
        self._games: dict[int, _GameState] = {}

    def register_root(
        self, worker_id: int, turn_id: int, batch_arrays: dict, legal_vocab_ids: list[int]
    ) -> RootEvalResponse:
        """Convenience single-request wrapper around `service()`'s root path
        -- still goes through `_service_roots`, so a lone `register_root`
        call is byte-identical to a `service([...])` call with one
        `RootEvalRequest` (the merge path's own `len(payloads) == 1`
        trivial-passthrough case, `merged_executors._merge_root_batches`)."""
        request = RootEvalRequest(
            worker_id=int(worker_id),
            turn_id=int(turn_id),
            batch_arrays=batch_arrays,
            legal_vocab_ids=list(legal_vocab_ids),
        )
        return self._service_roots([request])[0]

    def service(
        self, requests: list[RootEvalRequest | WaveRequest]
    ) -> list[RootEvalResponse | WaveResponse]:
        """Groups `requests` by type, merges each group into ONE model call
        (ragged root merge / one `forward_decode_grouped`), and returns
        responses in the SAME order as `requests` (not grouped order)."""
        responses: list[Any] = [None] * len(requests)
        root_positions = [
            i for i, r in enumerate(requests) if isinstance(r, RootEvalRequest)
        ]
        wave_positions = [
            i for i, r in enumerate(requests) if isinstance(r, WaveRequest)
        ]
        if len(root_positions) + len(wave_positions) != len(requests):
            bad_types = sorted(
                {
                    type(r).__name__
                    for r in requests
                    if not isinstance(r, (RootEvalRequest, WaveRequest))
                }
            )
            raise TypeError(
                f"ActorInferenceServer.service() got unsupported request type(s): {bad_types}"
            )
        if root_positions:
            root_responses = self._service_roots([requests[i] for i in root_positions])
            for i, response in zip(root_positions, root_responses):
                responses[i] = response
        if wave_positions:
            wave_responses = self._service_waves([requests[i] for i in wave_positions])
            for i, response in zip(wave_positions, wave_responses):
                responses[i] = response
        return responses

    def _sync_if_profiling(self) -> None:
        # Honest GPU-vs-CPU wall attribution needs a device sync at bucket
        # boundaries; only paid when profiling (IMBA_ACTOR_PROFILE=1 path).
        if self.profile_sync and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def release_turn(self, worker_id: int, turn_id: int) -> None:
        self._turns.pop((int(worker_id), int(turn_id)), None)

    def release_game(self, worker_id: int) -> None:
        """Frees `worker_id`'s persisted per-game prefix KV
        (`_GameState`) -- the orchestrator calls this on `GameDone` (the
        worker's game just finished) and defensively again on
        `WorkerFinished`, mirroring `release_turn`'s own idempotent
        "no-op if already released or never registered" contract. Distinct
        from `release_turn`: that frees one TURN's decode-wave arena/node
        chains (still per-turn, unaffected by this optimization); this
        frees the whole GAME's growing root prefix."""
        self._games.pop(int(worker_id), None)

    def _service_roots(
        self, requests: list[RootEvalRequest]
    ) -> list[RootEvalResponse]:
        """Dispatches each request to the FULL (`_service_full_roots`,
        cross-worker batched) or INCREMENTAL (`_service_incremental_root`,
        per-request sequential single-prefix decode) path per its own
        `batch_arrays`/`incremental_tokens` field, preserving `requests`'
        own order in the returned list regardless of which path each one
        took -- mirrors `service()`'s own root/wave dispatch discipline."""
        if not requests:
            return []
        full_positions = [
            i for i, r in enumerate(requests) if r.batch_arrays is not None
        ]
        incremental_positions = [
            i for i, r in enumerate(requests) if r.incremental_tokens is not None
        ]
        responses: list[RootEvalResponse | None] = [None] * len(requests)
        if full_positions:
            full_responses = self._service_full_roots(
                [requests[i] for i in full_positions]
            )
            for i, response in zip(full_positions, full_responses):
                responses[i] = response
        for i in incremental_positions:
            responses[i] = self._service_incremental_root(requests[i])
        return responses  # type: ignore[return-value]

    def _service_full_roots(
        self, requests: list[RootEvalRequest]
    ) -> list[RootEvalResponse]:
        if not requests:
            return []
        _t0 = time.perf_counter()
        payloads = [_tensorize_root_batch(r.batch_arrays) for r in requests]
        merged = _merge_root_batches(payloads)
        self._sync_if_profiling()
        _t1 = time.perf_counter()
        output = _forward_model(
            model=self._model,
            batch=merged,
            device=self._device,
            dtype=self._dtype,
            return_kv=True,
        )
        self._sync_if_profiling()
        _t2 = time.perf_counter()
        self.stats["root_build_s"] += _t1 - _t0
        self.stats["root_gpu_s"] += _t2 - _t1
        self.stats["root_calls"] += 1
        self.stats["root_reqs"] += len(requests)
        output = _ensure_value_logits_placeholder(output)
        splits = _split_root_output(output, payloads)

        # One batched softmax over every pending root request's last-token
        # value_logits in this call, instead of one Python call per request.
        last_value_logits = torch.stack(
            [split["value_logits"][-1] for split in splits], dim=0
        )
        value_stm_list = _batched_value_stm(last_value_logits)

        responses: list[RootEvalResponse] = []
        for request, payload, split, value_stm in zip(
            requests, payloads, splits, value_stm_list
        ):
            key = (int(request.worker_id), int(request.turn_id))
            logits_last = split["logits"][-1]
            [legal_logits] = _gather_legal_logits(
                logits_last.unsqueeze(0), [list(request.legal_vocab_ids)]
            )

            # This game's first root request: (re-)establish the persisted
            # per-worker game prefix (incremental-root-KV optimization),
            # unconditionally overwriting any prior entry -- a fresh FULL
            # request always means "start fresh", whether or not the
            # previous game's entry was already explicitly released (see
            # _GameState's own docstring).
            game = _GameState(
                prefix_kv=split["kv_caches"], prefix_len=int(payload["total_tokens"])
            )
            self._games[int(request.worker_id)] = game
            self._turns[key] = _snapshot_turn_state(game)

            responses.append(
                RootEvalResponse(
                    turn_id=request.turn_id,
                    value_stm=value_stm,
                    legal_logits=legal_logits,
                )
            )
        self._sync_if_profiling()
        self.stats["root_post_s"] += time.perf_counter() - _t2
        return responses

    def _service_incremental_root(self, request: RootEvalRequest) -> RootEvalResponse:
        """Extends `worker_id`'s persisted `_GameState` prefix KV by the
        request's `incremental_tokens`, one new token per sequential
        `HSTUChessModel.forward_decode` (single-prefix decode) call --
        folding each step's own returned (k, v) directly into the persisted
        prefix (concatenated along its token dim) rather than accumulating
        a separate suffix, since (unlike a decode-WAVE's many divergent
        search-tree children) there is exactly one path here and the
        prefix must persist and grow for the NEXT turn too. The LAST
        token's logits/value_logits are the root eval output -- the same
        contract a full forward's last-token output satisfies.

        No cross-request batching here (unlike `_service_full_roots`'s
        ragged merge / `_service_waves`'s grouped decode): each incremental
        request has its OWN, differently-shaped, persisted prefix, so
        batching multiple workers' incremental extensions together would
        need the same per-game grouped-decode machinery `_service_waves`
        already uses for wave nodes -- out of this optimization's scope
        (see `docs/superpowers/sdd/increm-report.md`); k is normally 2 so
        each incremental request is already a small, cheap GPU call
        relative to the full-sequence forward it replaces.
        """
        worker_id = int(request.worker_id)
        game = self._games.get(worker_id)
        if game is None:
            raise KeyError(
                f"RootEvalRequest(incremental) for worker={worker_id} has no "
                "persisted game prefix -- missing/out-of-order full "
                "RootEvalRequest, or release_game() already freed this "
                "worker's game."
            )
        # request.prefix_len_before is guaranteed not None here --
        # RootEvalRequest.__post_init__ already enforces that whenever
        # incremental_tokens is set (the only way a request reaches this
        # method, via _service_roots' dispatch) -- so this check is
        # unconditional, not defensive against a None that can't occur.
        if int(request.prefix_len_before) != game.prefix_len:
            raise RuntimeError(
                f"worker={worker_id}: incremental RootEvalRequest expected "
                f"prefix_len_before={request.prefix_len_before}, server has "
                f"{game.prefix_len} -- worker/server prefix desync "
                "(protocol/ordering bug, e.g. a missed release_game or "
                "out-of-order request)."
            )
        tokens = request.incremental_tokens
        n_new = len(tokens["piece_ids"])
        if n_new < 1:
            raise ValueError(
                f"worker={worker_id}: incremental RootEvalRequest carries "
                "zero new tokens."
            )
        max_positions = int(self._model.position_embedding.max_seq_len)
        if game.prefix_len + n_new > max_positions:
            # Fail fast at the same conceptual boundary a full forward would
            # silently clamp position ids at (PositionEmbedding.at_positions
            # torch.clamps rather than raising) -- this persisted prefix
            # grows indefinitely across a whole game, unlike a one-shot full
            # forward, so silently clamping here would silently corrupt
            # every later turn's attention, not just this one's.
            raise RuntimeError(
                f"worker={worker_id}: incremental root extension would grow "
                f"the persisted prefix to {game.prefix_len + n_new} tokens, "
                f"past this model's max_position_embeddings={max_positions}."
            )

        # Batch-tensorize the WHOLE request's fields once (same discipline
        # as _tensorize_wave_rows/_tensorize_root_batch), instead of
        # rebuilding n_new tiny single-row tensors per field inside the
        # decode loop below -- k is small (normally 2) so this is a minor
        # win, but it's free and matches the pattern used everywhere else
        # in this module.
        batched_fields = {
            key: torch.tensor(tokens[key], dtype=torch.long)
            for key in (
                "piece_ids", "seq_token_id", "turn_id", "castle_id",
                "ep_file_id", "halfmove_bucket_id", "fullmove_bucket_id",
                "prev_move_id",
            )
        }

        _t0 = time.perf_counter()
        out: dict[str, Any] | None = None
        with torch.inference_mode(), _autocast_context(self._device, self._dtype):
            for i in range(n_new):
                row_batch = {
                    key: values[i : i + 1] for key, values in batched_fields.items()
                }
                position = torch.tensor([game.prefix_len], dtype=torch.long)
                out = self._model.forward_decode(
                    new_token_batch=row_batch,
                    positions=position,
                    prefix_kv=game.prefix_kv,
                )
                # out["kv"][layer] is (k_new, v_new), each [1, H, 1, d] (the
                # decode call's own batch dim of 1) -- squeeze it and fold
                # directly into the persisted [H, T, d] prefix (concatenated
                # along the token dim) so the NEXT iteration/turn already
                # sees this token as part of the prefix, exactly matching
                # what a full forward's own kv_caches would contain at this
                # length (tests/test_prefix_decode.py's decode-vs-forward
                # equivalence, generalized from "reusable fixed prefix +
                # growing suffix" to "prefix that grows in place" -- same
                # attention math either way, see
                # SequentialTransductionUnitJagged.forward_decode's own
                # position-based relative bias, not prefix/suffix identity).
                game.prefix_kv = [
                    (
                        torch.cat([prefix_k, k_new.squeeze(0)], dim=1),
                        torch.cat([prefix_v, v_new.squeeze(0)], dim=1),
                    )
                    for (prefix_k, prefix_v), (k_new, v_new) in zip(
                        game.prefix_kv, out["kv"]
                    )
                ]
                game.prefix_len += 1
        self._sync_if_profiling()
        # Distinct bucket from the full-forward path's "root_*" stats, NOT
        # folded in: those count one merged multi-request batched call, this
        # counts one request's own k-step SEQUENTIAL decode loop (with
        # per-token CPU build time included in the elapsed span) -- sharing
        # one counter across two different units would silently corrupt any
        # "avg time per root call" analysis built on top of it.
        self.stats["incremental_root_gpu_s"] += time.perf_counter() - _t0
        self.stats["incremental_root_calls"] += 1
        self.stats["incremental_root_reqs"] += 1
        self.stats["incremental_root_tokens"] += n_new

        assert out is not None  # n_new >= 1 guaranteed above
        out = _ensure_value_logits_placeholder(out)
        self._turns[(worker_id, int(request.turn_id))] = _snapshot_turn_state(game)
        logits_last = out["logits"][0]  # [V] (this call's batch dim is 1)
        [legal_logits] = _gather_legal_logits(
            logits_last.unsqueeze(0), [list(request.legal_vocab_ids)]
        )
        value_stm = _batched_value_stm(out["value_logits"])[0]
        return RootEvalResponse(
            turn_id=request.turn_id, value_stm=value_stm, legal_logits=legal_logits
        )

    def _service_waves(self, requests: list[WaveRequest]) -> list[WaveResponse]:
        if not requests:
            return []
        _t0 = time.perf_counter()
        decode_requests: list[_ArenaDecodeRequest] = []
        turns: list[_TurnState] = []
        per_request_chains: list[list[list[int]]] = []
        for request in requests:
            key = (int(request.worker_id), int(request.turn_id))
            turn = self._turns.get(key)
            if turn is None:
                raise KeyError(
                    f"WaveRequest for worker={request.worker_id} "
                    f"turn={request.turn_id} has no registered root eval -- "
                    "missing/out-of-order RootEvalRequest, or release_turn() "
                    "already freed this turn."
                )
            new_token_batch, positions, parent_chains = _tensorize_wave_rows(
                request.rows, turn
            )
            wave_size = len(request.rows)
            max_suffix = max((len(c) for c in parent_chains), default=0)
            suffix_kv = suffix_positions = suffix_mask = None
            if max_suffix > 0:
                # idx/mask must live on the SAME device as the arena
                # (self._device -- e.g. cuda:0 in production): the arena's
                # k/v tensors are device-resident (model output), and
                # advanced indexing (`arena[:, :, idx, :]` in
                # `_KVArena.gather_suffix`) requires the index tensor to
                # match, same rationale as merged_executors._merge_decode_
                # requests's own explicit device= on its fabricated
                # suffix rows (see that module's comment on the same
                # footgun).
                idx = torch.zeros(
                    (wave_size, max_suffix), dtype=torch.long, device=self._device
                )
                mask = torch.zeros(
                    (wave_size, max_suffix), dtype=torch.bool, device=self._device
                )
                for row, chain in enumerate(parent_chains):
                    if chain:
                        idx[row, : len(chain)] = torch.tensor(
                            chain, dtype=torch.long, device=self._device
                        )
                        mask[row, : len(chain)] = True
                suffix_kv = turn.arena.gather_suffix(idx)
                suffix_positions = (
                    torch.arange(max_suffix, device=self._device) + turn.prefix_len
                ).unsqueeze(0).expand(wave_size, -1)
                suffix_mask = mask

            decode_requests.append(
                _ArenaDecodeRequest(
                    nodes=range(wave_size),
                    new_token_batch=new_token_batch,
                    positions=positions,
                    suffix_kv=suffix_kv,
                    suffix_positions=suffix_positions,
                    suffix_mask=suffix_mask,
                    prefix_kv=turn.prefix_kv,
                    prefix_len=turn.prefix_len,
                )
            )
            turns.append(turn)
            per_request_chains.append(parent_chains)

        merged = _merge_decode_requests(decode_requests)
        self._sync_if_profiling()
        _t1 = time.perf_counter()
        with torch.inference_mode(), _autocast_context(self._device, self._dtype):
            out = self._model.forward_decode_grouped(
                new_token_batch=merged.new_token_batch,
                positions=merged.positions,
                group_index=merged.group_index,
                prefix_kv_grouped=merged.prefix_kv_grouped,
                prefix_lens=merged.prefix_lens,
                prefix_lens_list=merged.prefix_lens_list,
                suffix_kv=merged.suffix_kv,
                suffix_positions=merged.suffix_positions,
                suffix_mask=merged.suffix_mask,
            )
        self._sync_if_profiling()
        _t2 = time.perf_counter()
        self.stats["wave_build_s"] += _t1 - _t0
        self.stats["wave_gpu_s"] += _t2 - _t1
        self.stats["wave_calls"] += 1
        self.stats["wave_reqs"] += len(requests)
        self.stats["wave_rows"] += sum(len(r.rows) for r in requests)
        out = _ensure_value_logits_placeholder(out)
        split_outs = _split_decode_output(
            out, [len(dr.nodes) for dr in decode_requests]
        )

        responses: list[WaveResponse] = []
        for request, turn, parent_chains, out_g in zip(
            requests, turns, per_request_chains, split_outs
        ):
            rows = request.rows
            # Write this wave's new node KV into the turn's arena, ONCE per
            # node (profile motivation #3: replaces a torch.cat per node per
            # depth with one write here + one indexed gather on read, above).
            k_stack = torch.stack([k for k, _ in out_g["kv"]], dim=0)  # [L, B, H, 1, d]
            v_stack = torch.stack([v for _, v in out_g["kv"]], dim=0)
            k_rows = k_stack.squeeze(3).permute(0, 2, 1, 3)  # [L, H, B, d]
            v_rows = v_stack.squeeze(3).permute(0, 2, 1, 3)
            turn.arena = _get_or_create_arena(turn.arena, k_rows, v_rows)
            assigned_rows = turn.arena.append(k_rows, v_rows)
            for row, chain, own_row in zip(rows, parent_chains, assigned_rows):
                turn.node_chains[row.node_id] = chain + [own_row]

            legal_logits_per_row = _gather_legal_logits(
                out_g["logits"], [list(row.legal_vocab_ids) for row in rows]
            )
            value_stm_list = _batched_value_stm(out_g["value_logits"])
            responses.append(
                WaveResponse(rows=list(zip(value_stm_list, legal_logits_per_row)))
            )
        self._sync_if_profiling()
        self.stats["wave_post_s"] += time.perf_counter() - _t2
        return responses
