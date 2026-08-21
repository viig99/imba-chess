from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import chess
import cozy_chess as cc
import torch
import torch.nn.functional as F

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.event_builder import (
    BOS_TOKEN_ID,
    EVENT_TOKEN_ID,
    TARGET_IGNORE_INDEX,
)
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval import cozy_bridge
from imba_chess.eval.search import PositionEval
from imba_chess.model import (
    HSTUChessModel,
    build_hstu_chess_config,
    create_batch_dense_mask,
)


class _SequenceHistory:
    """Incrementally builds the BOS+event sequence used for model inference."""

    def __init__(
        self, *, move_vocab: MoveVocab, board_state_encoder: BoardStateEncoder
    ) -> None:
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder

        self.seq_token_id: list[int] = [BOS_TOKEN_ID]
        self.piece_ids: list[list[int]] = [[0] * 64]
        self.turn_id: list[int] = [0]
        self.castle_id: list[int] = [0]
        self.ep_file_id: list[int] = [0]
        self.halfmove_bucket_id: list[int] = [0]
        self.fullmove_bucket_id: list[int] = [0]
        self.prev_move_id: list[int] = [self._move_vocab.start_id]
        self.target_move_id: list[int] = [TARGET_IGNORE_INDEX]
        self.played_by_elo: list[int] = [0]

        self._prev_move_id_for_next_token = self._move_vocab.start_id

    def append_observed_position(self, board: chess.Board) -> None:
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)

    def record_played_move(self, move_uci: str) -> None:
        self._prev_move_id_for_next_token = int(self._move_vocab.encode(move_uci))

    def _append_from_state(self, state) -> None:
        self.seq_token_id.append(EVENT_TOKEN_ID)
        self.piece_ids.append(list(state.piece_ids))
        self.turn_id.append(int(state.turn_id))
        self.castle_id.append(int(state.castle_id))
        self.ep_file_id.append(int(state.ep_file_id))
        self.halfmove_bucket_id.append(int(state.halfmove_bucket_id))
        self.fullmove_bucket_id.append(int(state.fullmove_bucket_id))
        self.prev_move_id.append(int(self._prev_move_id_for_next_token))
        self.target_move_id.append(TARGET_IGNORE_INDEX)
        self.played_by_elo.append(0)

    def _pop_last(self) -> None:
        self.seq_token_id.pop()
        self.piece_ids.pop()
        self.turn_id.pop()
        self.castle_id.pop()
        self.ep_file_id.pop()
        self.halfmove_bucket_id.pop()
        self.fullmove_bucket_id.pop()
        self.prev_move_id.pop()
        self.target_move_id.pop()
        self.played_by_elo.pop()

    def _build_single_batch(self) -> dict[str, Any]:
        # Single-sequence jagged batch; avoids collate list-copy overhead.
        total_tokens = len(self.seq_token_id)
        return {
            "game_id": ["stockfish_eval"],
            "game_result_white": torch.tensor([0], dtype=torch.long),
            "num_games": 1,
            "total_tokens": total_tokens,
            "seq_lens": torch.tensor([total_tokens], dtype=torch.long),
            "seq_offsets": torch.tensor([0, total_tokens], dtype=torch.long),
            "piece_ids": torch.tensor(self.piece_ids, dtype=torch.long),
            "seq_token_id": torch.tensor(self.seq_token_id, dtype=torch.long),
            "turn_id": torch.tensor(self.turn_id, dtype=torch.long),
            "castle_id": torch.tensor(self.castle_id, dtype=torch.long),
            "ep_file_id": torch.tensor(self.ep_file_id, dtype=torch.long),
            "halfmove_bucket_id": torch.tensor(
                self.halfmove_bucket_id, dtype=torch.long
            ),
            "fullmove_bucket_id": torch.tensor(
                self.fullmove_bucket_id, dtype=torch.long
            ),
            "prev_move_id": torch.tensor(self.prev_move_id, dtype=torch.long),
            "target_move_id": torch.tensor(self.target_move_id, dtype=torch.long),
            "played_by_elo": torch.tensor(self.played_by_elo, dtype=torch.long),
        }

    def build_batch_for_current_position(self, board: chess.Board) -> dict[str, Any]:
        # Add transient current-position token for next-move prediction only.
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)
        try:
            return self._build_single_batch()
        finally:
            self._pop_last()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def load_hstu_checkpoint(
    *,
    checkpoint_path: Path,
    repo_config,
    move_vocab: MoveVocab,
    device: torch.device,
    compile_model: bool,
    require_value_head: bool = False,
) -> tuple[torch.nn.Module, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint must be a model state_dict or Ignite checkpoint containing key 'model'."
        )
    normalized_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint state_dict keys must be strings")
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        normalized_state_dict[new_key] = value
    checkpoint_has_value_head = any(
        key.startswith("value_head.") for key in normalized_state_dict
    )
    if require_value_head and not checkpoint_has_value_head:
        raise ValueError(
            "model_move_policy in {value_rerank,value_search_d2} requires a checkpoint with value_head "
            "parameters, but checkpoint contains no 'value_head.*' keys."
        )

    model_cfg = build_hstu_chess_config(
        repo_config.model,
        move_vocab_size=len(move_vocab),
    )
    if bool(model_cfg.enable_value_head) != bool(checkpoint_has_value_head):
        print(
            "Adjusting runtime model enable_value_head to match checkpoint "
            f"(checkpoint_has_value_head={checkpoint_has_value_head})."
        )
        model_cfg = replace(model_cfg, enable_value_head=checkpoint_has_value_head)

    model: torch.nn.Module = HSTUChessModel(model_cfg).to(device)
    model.load_state_dict(normalized_state_dict, strict=True)
    model.eval()
    compile_enabled = False
    if compile_model:
        attention_dim = int(model_cfg.attention_dim)
        if not _is_power_of_two(attention_dim):
            print(
                "torch.compile disabled for eval: "
                f"model attention_dim={attention_dim} is not a power of two; "
                "this can fail Triton codegen in inference kernels."
            )
        else:
            model = torch.compile(model, dynamic=True, fullgraph=False)
            compile_enabled = True
    return model, compile_enabled


def _autocast_context(device: torch.device, dtype: torch.dtype):
    """Shared inference autocast policy: CUDA + fp16/bf16 only, else a no-op.

    Factored out of _forward_model/CachedPositionEvaluator.evaluate so the
    rollout script's merged-wave executors (cross-game batching) can wrap
    their own model calls with the exact same policy.
    """
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _forward_model(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    return_kv: bool = False,
) -> dict[str, torch.Tensor]:
    seq_offsets = batch["seq_offsets"].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    # Dense mask, not BlockMask: same admitted positions, but attention runs as
    # one fused SDPA call instead of eager flex_attention's ~95 ms/forward of
    # pure dispatch overhead (docs/superpowers/notes/
    # 2026-08-20-rl-throughput-bottleneck.md). Safe here because this path only
    # ever sees play/search batches -- a few sequences of a few hundred tokens.
    # Dataset-sized evaluation (ignite_evaluator, eval_value_loss) must keep
    # BlockMask; create_batch_dense_mask's docstring has the size criterion.
    block_mask = create_batch_dense_mask(
        seq_offsets,
        total_tokens=int(batch["total_tokens"]),
        device=device,
    )
    with torch.inference_mode(), _autocast_context(device, dtype):
        return model(
            batch, block_mask=block_mask, return_loss=False, return_kv=return_kv
        )


def _value_scalar_from_logits(value_logits_last: torch.Tensor) -> float:
    probs = torch.softmax(value_logits_last.float(), dim=-1)
    return float((probs[2] - probs[0]).item())


def _batched_value_scalars(value_logits: torch.Tensor) -> list[float]:
    """Batched twin of _value_scalar_from_logits: [B, 3] -> B floats.

    The per-node version ran a softmax, two index reads and an .item() for
    every evaluated node. At budget 2048 a decode wave carries ~1,321 nodes, so
    that was ~5k torch dispatches per wave over 3-element rows -- measured 11.9
    us/node together with PositionEval construction. Rows are independent under
    softmax(dim=-1), so batching is a pure dispatch win.
    """
    probs = torch.softmax(value_logits.float(), dim=-1)
    return (probs[:, 2] - probs[:, 0]).tolist()


def _batched_legal_log_priors(
    logits: torch.Tensor, id_lists: list[list[int]]
) -> list[list[float]]:
    """Per-node legal-move log-priors for a whole wave.

    Replaces per-node `torch.tensor(ids)` + `index_select` + `log_softmax` +
    `.tolist()` (measured 7.0 us/node) with one gather and one masked
    log_softmax for the wave.

    Nodes have different legal-move counts, so rows are padded to the wave
    maximum. Padded slots are filled with vocab id 0 and then masked to -inf
    BEFORE the log_softmax: exp(-inf) is exactly 0.0, so they contribute
    nothing to either the max or the normalizer, and each row stays normalized
    over exactly its own moves. Filling without masking would leak logits[.,0]
    into every short row -- pinned by
    tests/test_batched_projection.py::test_batched_log_priors_padding_cannot_leak.
    """
    lens = [len(ids) for ids in id_lists]
    width = max(lens)
    flat: list[int] = []
    for ids, n in zip(id_lists, lens):
        flat.extend(ids)
        if n < width:
            flat.extend((0,) * (width - n))
    idx = torch.tensor(flat, device=logits.device, dtype=torch.long).view(
        len(id_lists), width
    )
    picked = logits.gather(1, idx)
    keep = torch.arange(width, device=logits.device).unsqueeze(0) < torch.tensor(
        lens, device=logits.device
    ).unsqueeze(1)
    picked = picked.masked_fill(~keep, float("-inf"))
    rows = torch.log_softmax(picked.float(), dim=1).tolist()
    return [row[:n] for row, n in zip(rows, lens)]

def _project_legal_logits(
    *,
    logits: torch.Tensor,
    board: chess.Board,
    move_vocab: MoveVocab,
) -> tuple[torch.Tensor, list[chess.Move], int, int]:
    legal_moves = list(board.legal_moves)
    legal_move_ids: list[int] = []
    legal_moves_with_ids: list[chess.Move] = []
    for move in legal_moves:
        move_id = move_vocab.token_to_id.get(move.uci())
        if move_id is not None:
            legal_move_ids.append(int(move_id))
            legal_moves_with_ids.append(move)
    total_legal = len(legal_moves)
    mapped_legal = len(legal_move_ids)
    if not legal_move_ids:
        raise RuntimeError(
            "No legal moves mapped to vocab ids for current board "
            f"(total legal={total_legal})."
        )
    # Canonical order: sort by UCI string so python-chess and cozy movegen
    # (Stage 3) produce identical move lists. Gumbel draws and prior-tie
    # breaks in search are index-based, so this order is behavior, not just
    # cosmetics. Sort ids and moves jointly (before index_select) so
    # legal_logits stays aligned to legal_moves_with_ids.
    order = sorted(range(len(legal_moves_with_ids)), key=lambda i: legal_moves_with_ids[i].uci())
    legal_moves_with_ids = [legal_moves_with_ids[i] for i in order]
    legal_move_ids = [legal_move_ids[i] for i in order]
    legal_ids_tensor = torch.tensor(
        legal_move_ids, device=logits.device, dtype=torch.long
    )
    legal_logits = logits.index_select(0, legal_ids_tensor)
    return legal_logits, legal_moves_with_ids, total_legal, mapped_legal


# cozy encodes castling as king-takes-own-rook, so these four raw move strings
# are ambiguous: `e1h1` is O-O when a king stands on e1, but an ordinary rook
# slide otherwise -- the SAME (from, to, promotion) Move with two different
# correct UCIs. They are therefore the only moves whose mapping depends on the
# board, and the only ones _cozy_move_id_and_uci must refuse to memoize.
_CASTLE_RAW_TO_UCI = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}

# Per-vocab memo of cozy Move -> (vocab id | None, standard UCI). Keyed weakly
# on the vocab so two vocabs never share entries and the memo dies with its
# owner. cozy Move is hashable with value semantics (verified in
# tests/test_cozy_move_id_cache.py), which is what makes this sound.
_MOVE_ID_MEMO: "WeakKeyDictionary[MoveVocab, dict[Any, tuple[int | None, str]]]" = (
    WeakKeyDictionary()
)


def _cozy_move_id_and_uci(
    cozy_board: "cc.Board", move: "cc.Move", move_vocab: MoveVocab
) -> tuple[int | None, str]:
    """(vocab id or None, standard UCI) for a cozy move, memoized per vocab.

    Replaces `cozy_move_to_uci(...)` + `token_to_id.get(...)` on the hot path.
    That pair built a fresh Python string for every legal move of every
    evaluated node (10.5M allocations per 20-game run) plus up to two FFI board
    queries, then hashed the string; here a hit is one dict lookup on the Move
    itself. Measured 311 -> 154 ns/move (2.02x, fully separated distributions)
    by scripts/bench_move_id_micro.py on real positions.

    Castling is never memoized -- see _CASTLE_RAW_TO_UCI.
    """
    memo = _MOVE_ID_MEMO.get(move_vocab)
    if memo is None:
        memo = {}
        _MOVE_ID_MEMO[move_vocab] = memo
    hit = memo.get(move)
    if hit is not None:
        return hit

    raw = str(move)
    if raw in _CASTLE_RAW_TO_UCI:
        if cozy_board.piece_on(move.from_square) == cc.Piece.King:
            raw = _CASTLE_RAW_TO_UCI[raw]
        # Board-dependent: deliberately not cached.
        return (move_vocab.token_to_id.get(raw), raw)

    result = (move_vocab.token_to_id.get(raw), raw)
    memo[move] = result
    return result


def _legal_moves_ids_ucis(
    cozy_board: "cc.Board", move_vocab: MoveVocab
) -> tuple[list[int], list["cc.Move"], list[str], int]:
    """(vocab ids, moves, UCIs) in canonical UCI order, plus total legal count.

    The single source of the vocab-mapping + UCI-sort discipline for the cozy
    path: `_project_legal_logits_cozy` wraps this with the per-node tensor
    gather, while `CachedPositionEvaluator.consume_decode_result` calls it
    directly and gathers a whole wave at once. Keeping one implementation is
    what lets tests/test_actor_worker.py keep using
    `_project_legal_logits_cozy` as the oracle for the eval path's independent
    twin.

    Single pass on purpose: this runs once per evaluated search node, so the
    old build-all-UCIs-then-filter shape allocated a throwaway list per node
    on top of a string per move. Returns empty lists when nothing maps; callers
    decide whether that is an error.
    """
    legal_moves = list(cozy_board.generate_moves())
    ids: list[int] = []
    moves: list[cc.Move] = []
    ucis: list[str] = []
    for move in legal_moves:
        move_id, uci = _cozy_move_id_and_uci(cozy_board, move, move_vocab)
        if move_id is not None:
            ids.append(int(move_id))
            moves.append(move)
            ucis.append(uci)
    if not ids:
        return [], [], [], len(legal_moves)
    # Canonical order: sort by UCI string so python-chess and cozy movegen
    # produce identical move lists (Stage 3 Step 0). Sort ids/moves/ucis
    # jointly so the gathered logits stay aligned to both moves and ucis.
    order = sorted(range(len(moves)), key=lambda i: ucis[i])
    return (
        [ids[i] for i in order],
        [moves[i] for i in order],
        [ucis[i] for i in order],
        len(legal_moves),
    )


def _project_legal_logits_cozy(
    *,
    logits: torch.Tensor,
    cozy_board: cc.Board,
    move_vocab: MoveVocab,
) -> tuple[torch.Tensor, list[cc.Move], list[str], int, int]:
    """Cozy-native twin of `_project_legal_logits`: same vocab-mapping and
    UCI-sort discipline (shared via `_legal_moves_ids_ucis`), but movegen goes
    through cozy-chess `generate_moves` and the per-move UCI/id pair comes from
    `_cozy_move_id_and_uci` (memoized; castling-aware). `legal_ucis` is
    index-aligned with the returned `legal_moves`.

    Still the reference implementation the eval path is cross-checked against;
    the rollout wave path calls `_legal_moves_ids_ucis` plus a batched gather
    instead, to avoid one torch dispatch per node.
    """
    ids, moves, ucis, total_legal = _legal_moves_ids_ucis(cozy_board, move_vocab)
    if not ids:
        raise RuntimeError(
            "No legal moves mapped to vocab ids for current board "
            f"(total legal={total_legal})."
        )
    legal_ids_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
    legal_logits = logits.index_select(0, legal_ids_tensor)
    return legal_logits, moves, ucis, total_legal, len(ids)


class _CachedNode:
    """Search-node handle: parent link + the move that led here.

    path_kv is filled after this node is evaluated: the stacked per-layer
    K/V of every token on the root->self line, shapes [L, H, depth+1, d].
    A child's decode suffix is exactly its parent's path_kv. Parents are
    always evaluated before children in every strategy, so the path is
    complete at evaluate() time.
    """

    __slots__ = ("parent", "move_id", "depth", "path_kv")

    def __init__(self, parent: "_CachedNode | None", move_id: int, depth: int) -> None:
        self.parent = parent
        self.move_id = move_id
        self.depth = depth
        self.path_kv: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class _DecodeRequest:
    """CPU-side pre-work for one wave of forward_decode, plus what consuming
    the result needs afterward.

    Built by CachedPositionEvaluator.build_decode_request; consumed by
    consume_decode_result once the model call (forward_decode, single-game,
    or forward_decode_grouped, cross-game merged) has produced logits and
    per-layer (k, v). prefix_kv/prefix_len are carried alongside so the
    rollout script's cross-game executor can read each game's own prefix
    without reaching into evaluator internals -- needed to build the merged
    call's prefix_kv_grouped/prefix_lens.
    """

    nodes: list[_CachedNode]
    boards: list[cc.Board]
    new_token_batch: dict[str, Any]
    positions: torch.Tensor
    suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None
    suffix_positions: torch.Tensor | None
    suffix_mask: torch.Tensor | None
    prefix_kv: Any
    prefix_len: int


class CachedPositionEvaluator:
    """PositionEvaluator over a per-turn prefix K/V cache + one-token decodes.

    The root forward's last token is the current-position token every
    candidate sequence starts from, so its kv_caches are the shared prefix
    and each search node adds exactly one token relative to its parent.
    Constructed fresh each model turn.
    """

    def __init__(
        self,
        *,
        model,
        move_vocab: MoveVocab,
        board_state_encoder: BoardStateEncoder,
        device: torch.device,
        dtype: torch.dtype,
        prefix_kv,
        prefix_len: int,
    ) -> None:
        self._model = model
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder
        self._device = device
        self._dtype = dtype
        self._prefix_kv = prefix_kv
        self._prefix_len = int(prefix_len)

    def extend(self, handle, move_uci: str):
        """handle is opaque to the caller; move_uci only feeds vocab encoding
        (no board is needed here -- the parent's path_kv already captures
        everything about the position)."""
        parent = handle if isinstance(handle, _CachedNode) else None
        depth = parent.depth + 1 if parent is not None else 0
        return _CachedNode(parent, int(self._move_vocab.encode(move_uci)), depth)

    def build_decode_request(self, batch) -> _DecodeRequest:
        """All CPU pre-work for one wave: encode boards, token tensors,
        suffix gather, positions -- everything before the model call.

        `batch` is `list[(handle, cozy_board)]` (cozy-chess Board, Stage 3);
        encoding goes through `encode_cozy`.

        Precondition: batch is non-empty (evaluate() guards the empty case
        before calling this; the rollout script's decode-wave payloads are
        also always non-empty, since every EvalRequest-yielding generator
        only yields a non-empty batch).
        """
        nodes: list[_CachedNode] = [handle for handle, _ in batch]
        boards = [cozy_board for _, cozy_board in batch]
        states = [self._board_state_encoder.encode_cozy(cozy_board) for cozy_board in boards]
        wave_size = len(batch)

        new_token_batch = {
            "piece_ids": torch.tensor(
                [state.piece_ids for state in states], dtype=torch.long
            ),
            "seq_token_id": torch.full((wave_size,), EVENT_TOKEN_ID, dtype=torch.long),
            "turn_id": torch.tensor([state.turn_id for state in states], dtype=torch.long),
            "castle_id": torch.tensor(
                [state.castle_id for state in states], dtype=torch.long
            ),
            "ep_file_id": torch.tensor(
                [state.ep_file_id for state in states], dtype=torch.long
            ),
            "halfmove_bucket_id": torch.tensor(
                [state.halfmove_bucket_id for state in states], dtype=torch.long
            ),
            "fullmove_bucket_id": torch.tensor(
                [state.fullmove_bucket_id for state in states], dtype=torch.long
            ),
            "prev_move_id": torch.tensor(
                [node.move_id for node in nodes], dtype=torch.long
            ),
        }
        positions = torch.tensor(
            [self._prefix_len + node.depth for node in nodes], dtype=torch.long
        )
        max_suffix = max(node.depth for node in nodes)

        suffix_kv = suffix_positions = suffix_mask = None
        if max_suffix > 0:
            suffix_kv, suffix_positions, suffix_mask = self._wave_suffixes(
                nodes, max_suffix
            )

        return _DecodeRequest(
            nodes=nodes,
            boards=boards,
            new_token_batch=new_token_batch,
            positions=positions,
            suffix_kv=suffix_kv,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
            prefix_kv=self._prefix_kv,
            prefix_len=self._prefix_len,
        )

    def consume_decode_result(
        self, request: _DecodeRequest, out: dict[str, torch.Tensor]
    ) -> list[PositionEval]:
        """path_kv extension + logits->PositionEval, given a model output.

        `out` must have the forward_decode/forward_decode_grouped return
        shape: "kv" a per-layer list of (k, v) each [B, H, 1, d], "logits"
        and "value_logits" each [B, ...], rows in request.nodes/boards order.
        """
        # Stack the wave's per-layer (k, v) once, then extend each node's
        # root->self path cache so descendants get their suffix for free.
        k_all = torch.stack([k for k, _ in out["kv"]], dim=0)  # [L, B, H, 1, d]
        v_all = torch.stack([v for _, v in out["kv"]], dim=0)
        for row, node in enumerate(request.nodes):
            own_k, own_v = k_all[:, row], v_all[:, row]
            if node.parent is None:
                node.path_kv = (own_k, own_v)
            else:
                parent_k, parent_v = node.parent.path_kv
                node.path_kv = (
                    torch.cat([parent_k, own_k], dim=2),
                    torch.cat([parent_v, own_v], dim=2),
                )
        # One device->host transfer per wave instead of two syncs per node.
        logits = out["logits"].float().cpu()
        value_logits = out["value_logits"].float().cpu()

        # Per-node work is pure Python (movegen + vocab mapping + canonical
        # sort); every tensor op is hoisted to one call for the whole wave. At
        # budget 2048 a wave carries ~1,321 nodes, and the old shape ran a
        # softmax, a torch.tensor, an index_select and a log_softmax per node
        # over ~31-element rows, where dispatch dominated the arithmetic.
        per_node = [
            _legal_moves_ids_ucis(cozy_board, self._move_vocab)
            for cozy_board in request.boards
        ]
        values = _batched_value_scalars(value_logits)

        id_lists = [ids for ids, _, _, _ in per_node]
        prior_rows: list[list[float]] = [[] for _ in id_lists]
        if all(id_lists):
            prior_rows = _batched_legal_log_priors(logits, id_lists)
        else:
            # A node whose legal moves map to nothing in the vocab yields an
            # empty PositionEval, matching the old RuntimeError branch. Gather
            # only the rows that have moves so the batch stays rectangular.
            keep = [row for row, ids in enumerate(id_lists) if ids]
            if keep:
                sub = _batched_legal_log_priors(
                    logits.index_select(
                        0, torch.tensor(keep, device=logits.device, dtype=torch.long)
                    ),
                    [id_lists[row] for row in keep],
                )
                for row, priors in zip(keep, sub):
                    prior_rows[row] = priors

        return [
            PositionEval(
                value_stm=values[row],
                legal_moves=moves,
                legal_ucis=ucis,
                legal_log_priors=prior_rows[row],
            )
            for row, (_ids, moves, ucis, _total) in enumerate(per_node)
        ]

    def evaluate(self, batch):
        if not batch:
            return []
        request = self.build_decode_request(batch)
        with torch.inference_mode(), _autocast_context(self._device, self._dtype):
            out = self._model.forward_decode(
                new_token_batch=request.new_token_batch,
                positions=request.positions,
                prefix_kv=request.prefix_kv,
                suffix_kv=request.suffix_kv,
                suffix_positions=request.suffix_positions,
                suffix_mask=request.suffix_mask,
            )
        return self.consume_decode_result(request, out)

    def _wave_suffixes(self, nodes, max_suffix: int):
        """Padded per-layer ancestor K/V for one wave.

        Each node's suffix is its parent's path_kv ([L, H, depth, d]); rows
        are padded on the token dim to the wave max and stacked, then split
        back into the per-layer [B, H, s, d] pairs forward_decode expects.
        """
        ref_k, ref_v = self._prefix_kv[0]
        num_layers = len(self._prefix_kv)
        heads = ref_k.size(0)
        zero_k = ref_k.new_zeros((num_layers, heads, max_suffix, ref_k.size(-1)))
        zero_v = ref_v.new_zeros((num_layers, heads, max_suffix, ref_v.size(-1)))
        rows_k: list[torch.Tensor] = []
        rows_v: list[torch.Tensor] = []
        for node in nodes:
            if node.parent is None:
                rows_k.append(zero_k)
                rows_v.append(zero_v)
                continue
            path_k, path_v = node.parent.path_kv
            pad = max_suffix - node.depth
            rows_k.append(F.pad(path_k, (0, 0, 0, pad)) if pad else path_k)
            rows_v.append(F.pad(path_v, (0, 0, 0, pad)) if pad else path_v)
        stacked_k = torch.stack(rows_k, dim=0)  # [B, L, H, s, d_qk]
        stacked_v = torch.stack(rows_v, dim=0)
        suffix_kv = list(zip(stacked_k.unbind(dim=1), stacked_v.unbind(dim=1)))
        suffix_positions = (
            torch.arange(max_suffix, device=self._device).view(1, -1)
            + self._prefix_len
        ).expand(len(nodes), -1)
        suffix_mask = torch.tensor(
            [[i < node.depth for i in range(max_suffix)] for node in nodes],
            dtype=torch.bool,
            device=self._device,
        )
        return suffix_kv, suffix_positions, suffix_mask
