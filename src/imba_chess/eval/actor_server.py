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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    span a CHILD of this node needs as its decode suffix)."""

    prefix_kv: Any
    prefix_len: int
    arena: _KVArena | None = None
    node_chains: dict[int, list[int]] = field(default_factory=dict)


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
    ) -> None:
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

    def release_turn(self, worker_id: int, turn_id: int) -> None:
        self._turns.pop((int(worker_id), int(turn_id)), None)

    def _service_roots(
        self, requests: list[RootEvalRequest]
    ) -> list[RootEvalResponse]:
        if not requests:
            return []
        payloads = [_tensorize_root_batch(r.batch_arrays) for r in requests]
        merged = _merge_root_batches(payloads)
        output = _forward_model(
            model=self._model,
            batch=merged,
            device=self._device,
            dtype=self._dtype,
            return_kv=True,
        )
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

            self._turns[key] = _TurnState(
                prefix_kv=split["kv_caches"], prefix_len=int(payload["total_tokens"])
            )

            responses.append(
                RootEvalResponse(
                    turn_id=request.turn_id,
                    value_stm=value_stm,
                    legal_logits=legal_logits,
                )
            )
        return responses

    def _service_waves(self, requests: list[WaveRequest]) -> list[WaveResponse]:
        if not requests:
            return []
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
        return responses
