"""GPU inference server for the multiprocess eval actors design
(`docs/superpowers/specs/2026-07-19-multiprocess-eval-actors-design.md`).

`ActorInferenceServer` is the main-process counterpart of the torch-free
`run_eval_worker` (`actor_worker.py`, Task 1): it owns the model and ALL KV
state, keyed by `(worker_id, turn_id)` for root prefixes and
`(worker_id, turn_id, node_id)` for decode-wave nodes (Task 1's protocol
mints these as plain ints precisely so they survive the pipe without a
`_CachedNode`/board object ever crossing it -- see `actor_protocol.py`'s
module docstring).

Composition vs. reuse (per the task brief): the actual tensor math --
`_forward_model`, `_project_legal_logits_cozy`, `CachedPositionEvaluator.
build_decode_request`/`consume_decode_result`, `merged_executors.
_merge_root_batches`/`_split_root_output`/`_merge_decode_requests`/
`_split_decode_output` -- is imported and used UNMODIFIED (zero edits to
`position_evaluator.py`/`merged_executors.py`; both stay byte-identical for
their existing rollout-shared callers). What THIS module reimplements is the
composition layer those functions assumed lived in `_CachedNode` object
graphs: translating wire-safe `(worker_id, turn_id, node_id)` integer keys
into `_CachedNode` parent-links (`_service_waves`), and reconstructing a
`cozy_chess.Board` from the plain-int `BoardState` fields that cross the
wire, since (unlike the in-process evaluator, which is always handed a live
board by its caller) the server never receives one -- see
`_reconstruct_cozy_board`.
"""

from __future__ import annotations

from typing import Any

import cozy_chess as cc
import torch

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
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
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _CachedNode,
    _autocast_context,
    _forward_model,
    _project_legal_logits_cozy,
    _value_scalar_from_logits,
)

_PIECE_BY_OFFSET = (
    cc.Piece.Pawn,
    cc.Piece.Knight,
    cc.Piece.Bishop,
    cc.Piece.Rook,
    cc.Piece.Queen,
    cc.Piece.King,
)
_FILE_CHARS = "abcdefgh"

# The plain tensor-batch fields RootEvalRequest.batch_arrays/WaveRow carry --
# same field set _SequenceHistory._build_single_batch() (position_evaluator.py)
# and _PlainSequenceHistory._build_single_batch() (actor_worker.py) both
# produce, just torch-tensorized here on receipt.
_ROOT_BATCH_INT_LIST_FIELDS = (
    "seq_lens",
    "seq_offsets",
    "piece_ids",
    "seq_token_id",
    "turn_id",
    "castle_id",
    "ep_file_id",
    "halfmove_bucket_id",
    "fullmove_bucket_id",
    "prev_move_id",
    "target_move_id",
    "played_by_elo",
    "game_result_white",
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


def _reconstruct_cozy_board(
    *,
    piece_ids: list[int],
    turn_id: int,
    castle_id: int,
    ep_file_id: int,
    halfmove_bucket_id: int,
    fullmove_bucket_id: int,
    board_state_encoder: BoardStateEncoder,
) -> cc.Board:
    """Rebuilds a `cozy_chess.Board` from the plain-int `BoardState` fields
    that cross the wire (a `WaveRow.board_state`, or a `RootEvalRequest`'s
    own current-position token row) -- the server never receives a board
    object, only these fields, so legal-move generation and UCI derivation
    (`_project_legal_logits_cozy`, called on the result) need one rebuilt
    from scratch.

    Placement (`piece_ids`), side to move (`turn_id`), castling rights
    (`castle_id`), and the en-passant target square (`ep_file_id`, whose
    target RANK is derived from `turn_id` exactly the way
    `cozy_bridge._ep_adjacent_capturers_cozy` derives it: rank 6 if White is
    to move -- Black just double-pushed -- else rank 3) are all lossless 1:1
    encodings of the real position (`BoardStateEncoder.encode_cozy`), so
    this reconstruction is EXACT for all four. cozy-chess's legal-move
    generator (`Board.generate_moves()`/`checkers()`/pin detection) only
    ever consults placement, side to move, castling rights, and the
    en-passant square -- verified empirically during development via a
    1200-position differential fuzz check (30 random-playout games x 40
    plies) against real `cozy_bridge.board_to_cozy()` boards: zero legal-move-set
    mismatches.

    `halfmove_bucket_id`/`fullmove_bucket_id` ARE lossy (many real clock
    values collapse to one bucket) and movegen never consults them anyway
    (only `Board.status()`'s fifty-move check and `Board.hash()`/`.fen()`
    do -- none of which this reconstructed board is ever used for here).
    But `CachedPositionEvaluator.build_decode_request` -- reused UNMODIFIED
    by `ActorInferenceServer._service_waves` below -- re-derives its
    `new_token_batch` fields by calling `board_state_encoder.encode_cozy()`
    on whatever board it's handed, so an arbitrary movegen-safe
    representative would silently feed the MODEL a different (wrong)
    halfmove/fullmove bucket than the wire actually carried, breaking both
    real correctness and the fp32-exact-vs-reference test. So the
    representative picked here is the smallest value in each bucket
    (`bucket_id * bucket_size`, floor-clamped to >= 1 for fullmove_number,
    which cozy's `BoardBuilder.build()` rejects at 0 -- verified) --
    `_bucket(value, max_value, bucket_size) = min(value, max_value) //
    bucket_size` (`data/board_state.py`) is exactly invertible this way:
    since the ORIGINAL bucket_id already satisfies `bucket_id * bucket_size
    <= max_value` (it came from flooring some value `<= max_value`), the
    representative is never clamped, and `(bucket_id * bucket_size) //
    bucket_size == bucket_id` exactly. Re-encoding this board therefore
    reproduces the identical `halfmove_bucket_id`/`fullmove_bucket_id` (and,
    trivially, the identical placement/turn/castle/ep fields) the wire
    carried -- PRECONDITION: the server's own `board_state_encoder` must be
    configured with the same `BoardTokenConfig` (bucket sizes/maxima) the
    workers use, same as the model itself already assumes a fixed encoding
    scheme end to end.
    """
    builder = cc.BoardBuilder.empty()
    for square, value in enumerate(piece_ids):
        if value == 0:
            continue
        is_white = value <= 6
        offset = (value - 1) if is_white else (value - 7)
        builder.set_piece(
            cc.Square.from_index(square),
            _PIECE_BY_OFFSET[offset],
            cc.Color.White if is_white else cc.Color.Black,
        )
    if turn_id == 1:
        builder.set_side_to_move(cc.Color.Black)

    short_white = cc.File.H if (castle_id & 1) else None
    long_white = cc.File.A if (castle_id & 2) else None
    if short_white is not None or long_white is not None:
        builder.set_castle_rights(cc.Color.White, short=short_white, long=long_white)
    short_black = cc.File.H if (castle_id & 4) else None
    long_black = cc.File.A if (castle_id & 8) else None
    if short_black is not None or long_black is not None:
        builder.set_castle_rights(cc.Color.Black, short=short_black, long=long_black)

    if ep_file_id:
        file_char = _FILE_CHARS[ep_file_id - 1]
        rank_char = "6" if turn_id == 0 else "3"
        builder.set_en_passant(cc.Square.from_str(f"{file_char}{rank_char}"))

    cfg = board_state_encoder.config
    builder.set_halfmove_clock(int(halfmove_bucket_id) * cfg.halfmove_bucket_size)
    builder.set_fullmove_number(
        max(1, int(fullmove_bucket_id) * cfg.fullmove_bucket_size)
    )
    return builder.build()


def _cozy_board_from_root_batch(
    batch_arrays: dict, board_state_encoder: BoardStateEncoder
) -> cc.Board:
    """The current-position row is the LAST token `_SequenceHistory.
    build_batch_for_current_position` appends (transient; popped again
    worker-side before sending, per `_PlainSequenceHistory`'s own
    docstring) -- index -1 into every per-token field list."""
    return _reconstruct_cozy_board(
        piece_ids=batch_arrays["piece_ids"][-1],
        turn_id=int(batch_arrays["turn_id"][-1]),
        castle_id=int(batch_arrays["castle_id"][-1]),
        ep_file_id=int(batch_arrays["ep_file_id"][-1]),
        halfmove_bucket_id=int(batch_arrays["halfmove_bucket_id"][-1]),
        fullmove_bucket_id=int(batch_arrays["fullmove_bucket_id"][-1]),
        board_state_encoder=board_state_encoder,
    )


def _ensure_value_logits_placeholder(output: dict[str, Any]) -> dict[str, Any]:
    """When the server's model has no value head (`ActorInferenceServer(
    require_value_head=False, ...)` -- see that constructor's own
    docstring), `output` (from `_forward_model`/`forward_decode_grouped`)
    never has a `"value_logits"` key at all -- the model only sets it `if
    self.value_head is not None` (`model/hstu_model.py`). But Task 1's wire
    protocol makes `RootEvalResponse.value_stm`/every `WaveResponse` row's
    value_stm UNCONDITIONAL fields, and the REUSED tensor-math helpers this
    server calls (`_split_root_output`, `CachedPositionEvaluator.
    consume_decode_result`) both unconditionally index `out["value_logits"]`
    -- rather than editing those shared, rollout-consumed functions to make
    that key optional, this injects an explicit all-zero placeholder of the
    right shape in place. softmax(zeros) is the uniform distribution, so
    `_value_scalar_from_logits` (`probs[2] - probs[0]`) evaluates to exactly
    `0.0` for every row: a deliberate, documented "no opinion" placeholder,
    not a real value estimate. `require_value_head=False` is only valid for
    the `greedy` policy (the caller's own gate -- see
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


def _cozy_board_from_wave_row(
    board_state: dict, board_state_encoder: BoardStateEncoder
) -> cc.Board:
    return _reconstruct_cozy_board(
        piece_ids=board_state["piece_ids"],
        turn_id=int(board_state["turn_id"]),
        castle_id=int(board_state["castle_id"]),
        ep_file_id=int(board_state["ep_file_id"]),
        halfmove_bucket_id=int(board_state["halfmove_bucket_id"]),
        fullmove_bucket_id=int(board_state["fullmove_bucket_id"]),
        board_state_encoder=board_state_encoder,
    )


class ActorInferenceServer:
    """Owns the model and the ID-keyed KV store for every worker's in-flight
    game turns.

    `release_turn` is an explicit method, not a wire message (Task 1's
    protocol has no "release" message type) -- Task 3's orchestrator, which
    lives in the SAME process as this server (no pipe needed for this
    control signal), is expected to call it once it observes a turn is over:
    either a worker's next `RootEvalRequest` (implying the previous turn's
    search finished) or its `WorkerFinished` (implying the last turn of the
    worker's last game finished). This module does not infer that on its
    own -- an un-released turn's KV simply persists until either explicitly
    released or the same `(worker_id, turn_id)` key is overwritten by a
    same-turn re-registration (defensive; should not happen given Task 1's
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
        move_vocab: MoveVocab,
        board_state_encoder: BoardStateEncoder,
        device: torch.device,
        dtype: torch.dtype,
        require_value_head: bool = True,
    ) -> None:
        # Validated ONCE here, not per request. RootEvalResponse.value_stm
        # and every WaveResponse row's value_stm are UNCONDITIONAL fields in
        # Task 1's protocol (populated regardless of the worker's own
        # model_move_policy, which the server never even sees), so by
        # default (`require_value_head=True`, matching the G=1 path's own
        # `load_hstu_checkpoint(require_value_head=...)` gate for
        # value-dependent policies) a model with no value head can never
        # serve a single request. `require_value_head=False` -- the
        # orchestrator passes this exactly when `model_move_policy ==
        # "greedy"`, the only policy that never reads value_stm -- instead
        # allows construction with a value-head-less model; every response's
        # value_stm is then a documented `0.0` placeholder (see
        # `_ensure_value_logits_placeholder`, called from `_service_roots`/
        # `_service_waves`), never a real value estimate. This replaces the
        # per-turn RuntimeError scripts/eval_vs_stockfish.py's
        # `_select_model_move` used to raise (moved out of the worker per
        # Task 1's handoff note; the worker no longer has a model to check).
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
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder
        self._device = device
        self._dtype = dtype

        # (worker_id, turn_id) -> that turn's root-prefix evaluator (owns
        # prefix_kv/prefix_len; build_decode_request/consume_decode_result
        # reused unmodified from it for every decode wave in the turn).
        self._evaluators: dict[tuple[int, int], CachedPositionEvaluator] = {}
        # (worker_id, turn_id) -> {node_id: _CachedNode}, the ID-keyed
        # translation of what CachedPositionEvaluator's callers normally
        # hold as live Python object handles -- WaveRow.node_id/parent_id
        # resolve through this dict instead of an object reference, since
        # only ints survive the pipe (see actor_protocol.py's docstring).
        self._node_registry: dict[tuple[int, int], dict[int, _CachedNode]] = {}

    def register_root(
        self, worker_id: int, turn_id: int, batch_arrays: dict
    ) -> RootEvalResponse:
        """Convenience single-request wrapper around `service()`'s root path
        -- still goes through `_service_roots`, so a lone `register_root`
        call is byte-identical to a `service([...])` call with one
        `RootEvalRequest` (the merge path's own `len(payloads) == 1`
        trivial-passthrough case, `merged_executors._merge_root_batches`)."""
        request = RootEvalRequest(
            worker_id=int(worker_id), turn_id=int(turn_id), batch_arrays=batch_arrays
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
        key = (int(worker_id), int(turn_id))
        self._evaluators.pop(key, None)
        self._node_registry.pop(key, None)

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

        responses: list[RootEvalResponse] = []
        for request, payload, split in zip(requests, payloads, splits):
            key = (int(request.worker_id), int(request.turn_id))
            value_stm = _value_scalar_from_logits(split["value_logits"][-1])
            cozy_board = _cozy_board_from_root_batch(
                request.batch_arrays, self._board_state_encoder
            )
            legal_logits, _legal_moves, legal_ucis, _total, _mapped = (
                _project_legal_logits_cozy(
                    logits=split["logits"][-1],
                    cozy_board=cozy_board,
                    move_vocab=self._move_vocab,
                )
            )
            legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()

            self._evaluators[key] = CachedPositionEvaluator(
                model=self._model,
                move_vocab=self._move_vocab,
                board_state_encoder=self._board_state_encoder,
                device=self._device,
                dtype=self._dtype,
                prefix_kv=split["kv_caches"],
                prefix_len=int(payload["total_tokens"]),
            )
            self._node_registry[key] = {}

            responses.append(
                RootEvalResponse(
                    turn_id=request.turn_id,
                    value_stm=value_stm,
                    legal_ucis=legal_ucis,
                    legal_log_priors=legal_log_priors,
                )
            )
        return responses

    def _service_waves(self, requests: list[WaveRequest]) -> list[WaveResponse]:
        if not requests:
            return []
        decode_requests = []
        evaluators = []
        for request in requests:
            key = (int(request.worker_id), int(request.turn_id))
            evaluator = self._evaluators.get(key)
            if evaluator is None:
                raise KeyError(
                    f"WaveRequest for worker={request.worker_id} "
                    f"turn={request.turn_id} has no registered root eval -- "
                    "missing/out-of-order RootEvalRequest, or release_turn() "
                    "already freed this turn."
                )
            registry = self._node_registry.setdefault(key, {})
            handles: list[_CachedNode] = []
            boards: list[cc.Board] = []
            for row in request.rows:
                parent = registry[row.parent_id] if row.parent_id is not None else None
                depth = parent.depth + 1 if parent is not None else 0
                handle = _CachedNode(parent, int(row.prev_move_vocab_id), depth)
                registry[row.node_id] = handle
                handles.append(handle)
                boards.append(
                    _cozy_board_from_wave_row(row.board_state, self._board_state_encoder)
                )
            decode_requests.append(
                evaluator.build_decode_request(list(zip(handles, boards)))
            )
            evaluators.append(evaluator)

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
        for evaluator, decode_request, out_g in zip(
            evaluators, decode_requests, split_outs
        ):
            position_evals = evaluator.consume_decode_result(decode_request, out_g)
            responses.append(
                WaveResponse(
                    rows=[
                        (pe.value_stm, list(pe.legal_ucis), list(pe.legal_log_priors))
                        for pe in position_evals
                    ]
                )
            )
        return responses
