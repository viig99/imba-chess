from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

import chess
import cozy_chess as cc

from imba_chess.config import ModelConfig
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig, all_possible_uci_moves
from imba_chess.eval import cozy_bridge
from imba_chess.eval.actor_protocol import (
    RootEvalRequest,
    RootEvalResponse,
    WaveRequest,
    WaveResponse,
    WaveRow,
)
from imba_chess.eval.actor_server import ActorInferenceServer, _KVArena
from imba_chess.eval.actor_worker import _legal_vocab_projection
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _SequenceHistory,
    _value_scalar_from_logits,
)
from imba_chess.model import HSTUChessModel, build_hstu_chess_config

ATOL = 1e-6
RTOL = 1e-6


def _static_vocab() -> MoveVocab:
    return MoveVocab.build(
        all_possible_uci_moves(), config=MoveVocabConfig(include_unk=False)
    )


def _tiny_model(vocab_size: int, *, enable_value_head: bool = True) -> HSTUChessModel:
    torch.manual_seed(2)
    config = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=8,
            attention_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            max_position_embeddings=64,
            enable_value_head=enable_value_head,
        ),
        move_vocab_size=vocab_size,
    )
    return HSTUChessModel(config).eval()


def _plain_batch_arrays(batch: dict) -> dict:
    """Torch-tensor batch (position_evaluator._SequenceHistory's own shape)
    -> the plain lists/ints RootEvalRequest.batch_arrays carries on the
    wire, exactly what actor_worker._PlainSequenceHistory would build."""
    out = {}
    for key, value in batch.items():
        out[key] = value.tolist() if torch.is_tensor(value) else value
    return out


def _history_batch_for(
    *, moves: list[str], move_vocab: MoveVocab, encoder: BoardStateEncoder
) -> tuple[dict, chess.Board]:
    board = chess.Board()
    history = _SequenceHistory(move_vocab=move_vocab, board_state_encoder=encoder)
    for uci in moves:
        history.append_observed_position(board)
        history.record_played_move(uci)
        board.push_uci(uci)
    batch = history.build_batch_for_current_position(board)
    return batch, board


class _Fixture:
    def __init__(self):
        self.move_vocab = _static_vocab()
        self.model = _tiny_model(vocab_size=len(self.move_vocab))
        self.encoder = BoardStateEncoder()
        self.server = ActorInferenceServer(
            model=self.model, device=torch.device("cpu"), dtype=torch.float32
        )


def _reference_root_response(fixture: _Fixture, moves: list[str]):
    """Single-game reference: the same `_forward_model`-shaped pipeline the
    server reuses, but driven directly off ONE game's own batch (no
    cross-worker merge) and the game's REAL board -- the ground truth the
    server's merged path must match fp32-exactly. Legal-move projection is
    computed the same way `actor_worker._select_model_move` now computes it
    (worker-side, off the real board), not server-side."""
    batch, board = _history_batch_for(
        moves=moves, move_vocab=fixture.move_vocab, encoder=fixture.encoder
    )
    with torch.no_grad():
        output = fixture.model(batch, return_loss=False, return_kv=True)
    value_stm = _value_scalar_from_logits(output["value_logits"][-1])
    cozy_board = cozy_bridge.board_to_cozy(board)
    legal_vocab_ids, legal_moves, legal_ucis = _legal_vocab_projection(
        cozy_board, fixture.move_vocab
    )
    ids_tensor = torch.tensor(legal_vocab_ids, dtype=torch.long)
    legal_logits = output["logits"][-1].index_select(0, ids_tensor).tolist()

    cached = CachedPositionEvaluator(
        model=fixture.model,
        move_vocab=fixture.move_vocab,
        board_state_encoder=fixture.encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=output["kv_caches"],
        prefix_len=int(batch["total_tokens"]),
    )
    return {
        "board": board,
        "value_stm": value_stm,
        "legal_vocab_ids": legal_vocab_ids,
        "legal_moves": legal_moves,
        "legal_ucis": legal_ucis,
        "legal_logits": legal_logits,
        "cached": cached,
        "wire_batch_arrays": _plain_batch_arrays(batch),
    }


def _reference_decode(cached: CachedPositionEvaluator, batch):
    """Runs `CachedPositionEvaluator`'s own decode pipeline manually
    (instead of via `evaluate()`, which internally discards raw logits
    after projecting+log-softmaxing them) so tests can compare the server's
    arena-based decode path against RAW pre-softmax logits, not just the
    post-log-softmax `PositionEval.legal_log_priors`. This matters because
    `log_softmax` is shift-invariant (`log_softmax(x) == log_softmax(x +
    c)` for any per-row constant `c`), so matching log-priors alone would
    NOT catch a uniform per-row additive error in the arena's reconstructed
    decode logits -- exactly the kind of bug a suffix-gather-indexing
    mistake could introduce. Returns `(raw_logits [B, V] float32 cpu,
    position_evals)`; also runs `consume_decode_result` so callers get the
    normal `PositionEval` list AND each node's `path_kv` gets set (needed
    for a subsequent depth+1 wave in the same test)."""
    request = cached.build_decode_request(batch)
    with torch.inference_mode():
        out = cached._model.forward_decode(
            new_token_batch=request.new_token_batch,
            positions=request.positions,
            prefix_kv=request.prefix_kv,
            suffix_kv=request.suffix_kv,
            suffix_positions=request.suffix_positions,
            suffix_mask=request.suffix_mask,
        )
    raw_logits = out["logits"].float().cpu()
    position_evals = cached.consume_decode_result(request, out)
    return raw_logits, position_evals


def _assert_root_matches(
    response: RootEvalResponse, ref: dict, *, turn_id: int, atol: float = ATOL, rtol: float = RTOL
) -> None:
    assert isinstance(response, RootEvalResponse)
    assert response.turn_id == turn_id
    torch.testing.assert_close(
        torch.tensor(response.value_stm), torch.tensor(ref["value_stm"]),
        atol=atol, rtol=rtol,
    )
    torch.testing.assert_close(
        torch.tensor(response.legal_logits),
        torch.tensor(ref["legal_logits"]),
        atol=atol,
        rtol=rtol,
    )


def test_two_worker_root_eval_matches_reference_fp32_exact():
    """Two workers' RootEvalRequests in one service() call -> the ragged
    root merge (_merge_root_batches, pre-existing/rollout-shared, reused
    unmodified) must match each worker's own single-game forward pass, RAW
    logits gathered at each request's own legal_vocab_ids (the server no
    longer runs log_softmax at all -- that moved to the worker).

    Tolerance note: this specific assertion uses 1e-5, not the module's
    1e-6 default -- isolated during development (a standalone script
    calling ONLY _merge_root_batches/_forward_model/_split_root_output,
    zero actor_server.py code involved) to a documented, PRE-EXISTING
    floating-point characteristic of the ragged multi-document merge
    itself: concatenating two different-length token sequences into one
    flex_attention call and slicing the output back apart is logically
    (mask-wise) identical to two isolated calls, but not bit-identical --
    max observed drift ~7.6e-6 in logits for this exact scenario with zero
    server-side involvement. This is exactly what the design spec's
    "Collection policy and determinism" section calls out and accepts. The
    decode-wave merge path below (test_two_worker_wave_eval_...) has NO
    such drift and stays at the strict 1e-6 bar, as does single-request
    register_root -- this loosened tolerance is scoped to exactly the
    multi-request ragged root merge."""
    fixture = _Fixture()
    ref0 = _reference_root_response(fixture, ["e2e4", "e7e5"])
    ref1 = _reference_root_response(
        fixture, ["d2d4", "d7d5", "g1f3", "b8c6", "c1f4"]
    )

    req0 = RootEvalRequest(
        worker_id=0, turn_id=0, batch_arrays=ref0["wire_batch_arrays"],
        legal_vocab_ids=ref0["legal_vocab_ids"],
    )
    req1 = RootEvalRequest(
        worker_id=1, turn_id=0, batch_arrays=ref1["wire_batch_arrays"],
        legal_vocab_ids=ref1["legal_vocab_ids"],
    )
    responses = fixture.server.service([req0, req1])

    assert len(responses) == 2
    _assert_root_matches(responses[0], ref0, turn_id=0, atol=1e-5, rtol=1e-5)
    _assert_root_matches(responses[1], ref1, turn_id=0, atol=1e-5, rtol=1e-5)
    # Root registration must key state by (worker_id, turn_id).
    assert (0, 0) in fixture.server._turns
    assert (1, 0) in fixture.server._turns


def test_register_root_single_worker_matches_reference():
    fixture = _Fixture()
    ref = _reference_root_response(fixture, ["e2e4", "c7c5", "g1f3"])
    response = fixture.server.register_root(
        0, 0, ref["wire_batch_arrays"], ref["legal_vocab_ids"]
    )
    _assert_root_matches(response, ref, turn_id=0)


def _wave_row_for_child(
    *, node_id: int, parent_id: int | None, board_before: chess.Board, move: chess.Move,
    encoder: BoardStateEncoder, move_vocab: MoveVocab,
) -> tuple[WaveRow, chess.Board, list[int]]:
    child = board_before.copy()
    child.push(move)
    cozy_child = cozy_bridge.board_to_cozy(child)
    state = encoder.encode_cozy(cozy_child)
    legal_vocab_ids, _legal_moves, _legal_ucis = _legal_vocab_projection(
        cozy_child, move_vocab
    )
    row = WaveRow(
        node_id=node_id,
        parent_id=parent_id,
        # The move that LED to this node is a real model input token
        # (new_token_batch["prev_move_id"]), mirroring
        # _WorkerSearchNode.move_vocab_id / _CachedNode.move_id -- must be
        # the actual encoded move, not a placeholder.
        prev_move_vocab_id=int(move_vocab.encode(move.uci())),
        board_state=vars(state),
        legal_vocab_ids=legal_vocab_ids,
    )
    return row, child, legal_vocab_ids


def test_two_worker_wave_eval_matches_cached_position_evaluator_fp32_exact():
    """Depth-1 decode waves for two workers merged into one
    forward_decode_grouped call must match each worker's own
    CachedPositionEvaluator.evaluate() results fp32-exactly (RAW logits
    gathered at each row's own legal_vocab_ids -- CachedPositionEvaluator's
    own log-softmaxed legal_log_priors are converted back with the same
    reference-side index_select so both sides compare RAW logits)."""
    fixture = _Fixture()
    ref0 = _reference_root_response(fixture, ["e2e4", "e7e5"])
    ref1 = _reference_root_response(fixture, ["d2d4", "d7d5", "g1f3"])
    fixture.server.register_root(0, 0, ref0["wire_batch_arrays"], ref0["legal_vocab_ids"])
    fixture.server.register_root(1, 0, ref1["wire_batch_arrays"], ref1["legal_vocab_ids"])

    candidates0 = list(ref0["board"].legal_moves)[:2]
    candidates1 = list(ref1["board"].legal_moves)[:1]

    rows0, children0 = [], []
    for i, move in enumerate(candidates0):
        row, child, _ids = _wave_row_for_child(
            node_id=i, parent_id=None, board_before=ref0["board"], move=move,
            encoder=fixture.encoder, move_vocab=fixture.move_vocab,
        )
        rows0.append(row)
        children0.append(child)
    rows1, children1 = [], []
    for i, move in enumerate(candidates1):
        row, child, _ids = _wave_row_for_child(
            node_id=i, parent_id=None, board_before=ref1["board"], move=move,
            encoder=fixture.encoder, move_vocab=fixture.move_vocab,
        )
        rows1.append(row)
        children1.append(child)

    wave0 = WaveRequest(worker_id=0, turn_id=0, rows=rows0)
    wave1 = WaveRequest(worker_id=1, turn_id=0, rows=rows1)
    responses = fixture.server.service([wave0, wave1])
    assert len(responses) == 2
    assert isinstance(responses[0], WaveResponse)
    assert isinstance(responses[1], WaveResponse)

    # Reference: CachedPositionEvaluator seeded from each worker's own real
    # root prefix, extend()+decode on the REAL child boards -- compared
    # against the server's RAW legal_logits directly (see _reference_decode's
    # own docstring for why raw, not just log-softmaxed, comparison matters).
    for ref, candidates, children, response in (
        (ref0, candidates0, children0, responses[0]),
        (ref1, candidates1, children1, responses[1]),
    ):
        cached = ref["cached"]
        handles = [cached.extend(None, move.uci()) for move in candidates]
        cozy_children = [cozy_bridge.board_to_cozy(c) for c in children]
        raw_logits, ref_evals = _reference_decode(cached, list(zip(handles, cozy_children)))
        assert len(response.rows) == len(ref_evals)

        for row, ((value_stm, legal_logits), pe, cozy_child) in enumerate(
            zip(response.rows, ref_evals, cozy_children)
        ):
            torch.testing.assert_close(
                torch.tensor(value_stm), torch.tensor(pe.value_stm), atol=ATOL, rtol=RTOL
            )
            ref_vocab_ids, _moves, _ucis = _legal_vocab_projection(
                cozy_child, fixture.move_vocab
            )
            ref_legal_logits = raw_logits[row].index_select(
                0, torch.tensor(ref_vocab_ids, dtype=torch.long)
            )
            torch.testing.assert_close(
                torch.tensor(legal_logits), ref_legal_logits, atol=ATOL, rtol=RTOL
            )


def test_depth_two_wave_with_parent_link_matches_reference_and_release_frees_kv():
    """Single-worker depth-2 wave (parent_id referencing a depth-1 node
    minted in an earlier wave) must match CachedPositionEvaluator's own
    extend-chain result via the KV arena's suffix gather; release_turn must
    then zero the KV store for that turn's key without touching other
    still-live turns."""
    fixture = _Fixture()
    ref0 = _reference_root_response(fixture, ["e2e4", "e7e5"])
    ref1 = _reference_root_response(fixture, ["d2d4", "d7d5"])
    fixture.server.register_root(0, 0, ref0["wire_batch_arrays"], ref0["legal_vocab_ids"])
    fixture.server.register_root(1, 0, ref1["wire_batch_arrays"], ref1["legal_vocab_ids"])

    candidate = list(ref0["board"].legal_moves)[0]
    row1, child1, _ids1 = _wave_row_for_child(
        node_id=0, parent_id=None, board_before=ref0["board"], move=candidate,
        encoder=fixture.encoder, move_vocab=fixture.move_vocab,
    )
    fixture.server.service([WaveRequest(worker_id=0, turn_id=0, rows=[row1])])

    reply = list(child1.legal_moves)[0]
    row2, child2, _ids2 = _wave_row_for_child(
        node_id=1, parent_id=0, board_before=child1, move=reply,
        encoder=fixture.encoder, move_vocab=fixture.move_vocab,
    )
    response = fixture.server.service(
        [WaveRequest(worker_id=0, turn_id=0, rows=[row2])]
    )[0]

    cached = ref0["cached"]
    handle1 = cached.extend(None, candidate.uci())
    cozy_child1 = cozy_bridge.board_to_cozy(child1)
    _raw1, _evals1 = _reference_decode(cached, [(handle1, cozy_child1)])  # sets handle1.path_kv
    handle2 = cached.extend(handle1, reply.uci())
    cozy_child2 = cozy_bridge.board_to_cozy(child2)
    raw_logits2, ref_evals2 = _reference_decode(cached, [(handle2, cozy_child2)])
    ref_eval = ref_evals2[0]

    value_stm, legal_logits = response.rows[0]
    torch.testing.assert_close(
        torch.tensor(value_stm), torch.tensor(ref_eval.value_stm), atol=ATOL, rtol=RTOL
    )
    ref_vocab_ids, _moves, _ucis = _legal_vocab_projection(cozy_child2, fixture.move_vocab)
    ref_legal_logits = raw_logits2[0].index_select(
        0, torch.tensor(ref_vocab_ids, dtype=torch.long)
    )
    torch.testing.assert_close(
        torch.tensor(legal_logits), ref_legal_logits, atol=ATOL, rtol=RTOL
    )

    # No leak: releasing worker 0's turn frees exactly its KV, worker 1's
    # still-live turn is untouched.
    assert (0, 0) in fixture.server._turns
    fixture.server.release_turn(0, 0)
    assert (0, 0) not in fixture.server._turns
    assert (1, 0) in fixture.server._turns

    fixture.server.release_turn(1, 0)
    assert len(fixture.server._turns) == 0

    # Idempotent: releasing again (or a never-registered key) must not raise.
    fixture.server.release_turn(0, 0)
    fixture.server.release_turn(99, 99)


def test_wave_request_for_unregistered_turn_raises_fast():
    fixture = _Fixture()
    row = WaveRow(
        node_id=0, parent_id=None, prev_move_vocab_id=0,
        board_state={
            "piece_ids": [0] * 64, "turn_id": 0, "castle_id": 0, "ep_file_id": 0,
            "halfmove_bucket_id": 0, "fullmove_bucket_id": 0,
        },
        legal_vocab_ids=[],
    )
    with pytest.raises(KeyError):
        fixture.server.service([WaveRequest(worker_id=0, turn_id=0, rows=[row])])


def test_service_rejects_unsupported_request_type():
    fixture = _Fixture()
    with pytest.raises(TypeError):
        fixture.server.service([object()])


def test_value_head_guard_raises_at_construction():
    """Default (`require_value_head=True`, unspecified here on purpose --
    this is what every value-dependent-policy caller gets): a value-head-less
    model must still fail fast at construction, same as before this
    parameter existed."""
    move_vocab = _static_vocab()
    model = _tiny_model(vocab_size=len(move_vocab), enable_value_head=False)
    with pytest.raises(ValueError, match="value head"):
        ActorInferenceServer(model=model, device=torch.device("cpu"), dtype=torch.float32)


def test_require_value_head_false_allows_construction_and_serves_zero_placeholder():
    """`require_value_head=False` (the orchestrator's "greedy" gate --
    scripts/eval_vs_stockfish.py's `_run_segment_actor_mode`) must both (a)
    allow constructing the server against a value-head-less model, and (b)
    serve every response's value_stm as the documented `0.0` placeholder --
    checked on BOTH the root-eval and decode-wave paths, since each reuses a
    different tensor-math helper that unconditionally expects a
    "value_logits" key on its input dict -- see
    `_ensure_value_logits_placeholder`."""
    move_vocab = _static_vocab()
    model = _tiny_model(vocab_size=len(move_vocab), enable_value_head=False)
    encoder = BoardStateEncoder()
    server = ActorInferenceServer(
        model=model, device=torch.device("cpu"), dtype=torch.float32,
        require_value_head=False,
    )

    batch, board = _history_batch_for(
        moves=["e2e4", "e7e5"], move_vocab=move_vocab, encoder=encoder
    )
    cozy_board = cozy_bridge.board_to_cozy(board)
    legal_vocab_ids, _moves, _ucis = _legal_vocab_projection(cozy_board, move_vocab)
    root_response = server.register_root(
        0, 0, _plain_batch_arrays(batch), legal_vocab_ids
    )
    assert root_response.value_stm == 0.0

    move = list(board.legal_moves)[0]
    row, _child, _ids = _wave_row_for_child(
        node_id=0,
        parent_id=None,
        board_before=board,
        move=move,
        encoder=encoder,
        move_vocab=move_vocab,
    )
    wave_response = server.service(
        [WaveRequest(worker_id=0, turn_id=0, rows=[row])]
    )[0]
    value_stm, _legal_logits = wave_response.rows[0]
    assert value_stm == 0.0


# ---------------------------------------------------------------------------
# _KVArena unit tests (profile-driven thin-down, change B): the server's own
# KV bookkeeping, independent of the model/protocol plumbing above.
# ---------------------------------------------------------------------------


def test_kv_arena_append_and_gather_suffix_round_trip_across_capacity_growth():
    """Appends more rows than the arena's initial capacity (forcing at
    least one doubling grow), then verifies gather_suffix reconstructs the
    exact original per-node K/V for an arbitrary ancestor-chain index
    matrix -- proves growth preserves already-written rows and that the
    indexed gather is equivalent to reading the original per-row tensors
    directly."""
    rng = random.Random(0)
    num_layers, heads, dim = 2, 3, 4
    arena: _KVArena | None = None
    all_k_rows: list[torch.Tensor] = []  # each [L, H, d]
    all_v_rows: list[torch.Tensor] = []

    from imba_chess.eval.actor_server import _get_or_create_arena

    total_rows = 0
    for wave_size in (5, 7, 6, 10):  # cumulative 28, initial capacity 16 -> grows
        k_rows = torch.tensor(
            [[[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(wave_size)]
              for _ in range(heads)] for _ in range(num_layers)],
            dtype=torch.float32,
        )  # [L, H, B, d]
        v_rows = torch.tensor(
            [[[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(wave_size)]
              for _ in range(heads)] for _ in range(num_layers)],
            dtype=torch.float32,
        )
        arena = _get_or_create_arena(arena, k_rows, v_rows)
        assigned = arena.append(k_rows, v_rows)
        assert assigned == list(range(total_rows, total_rows + wave_size))
        for i in range(wave_size):
            all_k_rows.append(k_rows[:, :, i, :])
            all_v_rows.append(v_rows[:, :, i, :])
        total_rows += wave_size

    assert arena is not None
    assert arena.size == total_rows
    assert arena.k.shape[2] >= total_rows  # grew at least once past 16

    # Ask for an arbitrary (non-contiguous, out-of-order) suffix chain per
    # row and verify each gathered row matches the ORIGINAL tensor for that
    # arena index exactly (bit-for-bit -- this is a pure gather, no math).
    idx = torch.tensor([[0, 5, 12], [27, 1, 1], [3, 3, 3]], dtype=torch.long)
    suffix = arena.gather_suffix(idx)
    assert len(suffix) == num_layers
    for layer in range(num_layers):
        gathered_k, gathered_v = suffix[layer]
        assert gathered_k.shape == (idx.shape[0], heads, idx.shape[1], dim)
        for row in range(idx.shape[0]):
            for pos in range(idx.shape[1]):
                arena_row = int(idx[row, pos])
                torch.testing.assert_close(
                    gathered_k[row, :, pos, :], all_k_rows[arena_row][layer]
                )
                torch.testing.assert_close(
                    gathered_v[row, :, pos, :], all_v_rows[arena_row][layer]
                )
