from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import chess
import imba_chess_native as cc
import torch


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
    order = sorted(
        range(len(legal_moves_with_ids)), key=lambda i: legal_moves_with_ids[i].uci()
    )
    legal_moves_with_ids = [legal_moves_with_ids[i] for i in order]
    legal_move_ids = [legal_move_ids[i] for i in order]
    legal_ids_tensor = torch.tensor(
        legal_move_ids, device=logits.device, dtype=torch.long
    )
    legal_logits = logits.index_select(0, legal_ids_tensor)
    return legal_logits, legal_moves_with_ids, total_legal, mapped_legal


def _project_legal_logits_cozy(
    *,
    logits: torch.Tensor,
    cozy_board: cc.Board,
    move_vocab: MoveVocab,
) -> tuple[torch.Tensor, list[cc.Move], list[str], int, int]:
    """Native twin of `_project_legal_logits`: the same vocab-mapping and
    UCI-sort discipline, delegated whole to `cozy_bridge.project_legal_moves`,
    with only the per-node tensor gather left here. `legal_ucis` is
    index-aligned with the returned `legal_moves`.

    Still the reference implementation the eval path is cross-checked against;
    the rollout wave path calls the bridge directly plus a batched gather
    instead, to avoid one torch dispatch per node.
    """
    ids, moves, ucis, _forcing, total_legal = cozy_bridge.project_legal_moves(
        cozy_board, move_vocab
    )
    if not ids:
        raise RuntimeError(
            "No legal moves mapped to vocab ids for current board "
            f"(total legal={total_legal})."
        )
    legal_ids_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
    legal_logits = logits.index_select(0, legal_ids_tensor)
    return legal_logits, moves, ucis, total_legal, len(ids)


class _KVArena:
    """Growable per-turn store for one decode-token K/V row per search node.

    `k` and `v` are `[L, H, capacity, d]`. Nodes retain only append-only
    ancestor row indices; one indexed gather reconstructs a wave's parent
    suffixes without persistent root-to-node tensor copies.
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
        new_k = self.k.new_zeros(
            (self.k.shape[0], self.k.shape[1], new_capacity, self.k.shape[3])
        )
        new_v = self.v.new_zeros(
            (self.v.shape[0], self.v.shape[1], new_capacity, self.v.shape[3])
        )
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


def _padded_chain_indices(
    chains: list[list[int]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize all arena ancestor indices and their mask in two transfers."""
    max_suffix = max((len(chain) for chain in chains), default=0)
    padded = [chain + [0] * (max_suffix - len(chain)) for chain in chains]
    idx = torch.tensor(padded, dtype=torch.long, device=device)
    lengths = torch.tensor([len(chain) for chain in chains], device=device)
    mask = torch.arange(max_suffix, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return idx, mask


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


class _CachedNode:
    """Search-node handle with its append-only K/V arena ancestor chain."""

    __slots__ = ("parent", "move_id", "depth", "arena_chain")

    def __init__(self, parent: "_CachedNode | None", move_id: int, depth: int) -> None:
        self.parent = parent
        self.move_id = move_id
        self.depth = depth
        self.arena_chain: list[int] | None = None


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
        self._arena: _KVArena | None = None

    def extend(self, handle, move_uci: str):
        """Create an opaque child handle backed by the shared K/V arena."""
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
        states = [
            self._board_state_encoder.encode_cozy(cozy_board) for cozy_board in boards
        ]
        wave_size = len(batch)

        new_token_batch = {
            "piece_ids": torch.tensor(
                [state.piece_ids for state in states], dtype=torch.long
            ),
            "seq_token_id": torch.full((wave_size,), EVENT_TOKEN_ID, dtype=torch.long),
            "turn_id": torch.tensor(
                [state.turn_id for state in states], dtype=torch.long
            ),
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
            if self._arena is None:
                raise RuntimeError("Missing KV arena for non-root decode wave")
            parent_chains: list[list[int]] = []
            for node in nodes:
                if node.parent is None:
                    parent_chains.append([])
                    continue
                if node.parent.arena_chain is None:
                    raise RuntimeError(
                        "Cannot decode a child before evaluating its parent"
                    )
                parent_chains.append(node.parent.arena_chain)
            idx, suffix_mask = _padded_chain_indices(parent_chains, device=self._device)
            suffix_kv = self._arena.gather_suffix(idx)
            suffix_positions = (
                (torch.arange(max_suffix, device=self._device) + self._prefix_len)
                .unsqueeze(0)
                .expand(wave_size, -1)
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
        """Append this wave's K/V once, then project model outputs.

        `out` must have the forward_decode/forward_decode_grouped return
        shape: "kv" a per-layer list of (k, v) each [B, H, 1, d], "logits"
        and "value_logits" each [B, ...], rows in request.nodes/boards order.
        """
        k_stack = torch.stack([k for k, _ in out["kv"]], dim=0)  # [L, B, H, 1, d]
        v_stack = torch.stack([v for _, v in out["kv"]], dim=0)
        k_rows = k_stack.squeeze(3).permute(0, 2, 1, 3)  # [L, H, B, d]
        v_rows = v_stack.squeeze(3).permute(0, 2, 1, 3)
        self._arena = _get_or_create_arena(self._arena, k_rows, v_rows)
        assigned_rows = self._arena.append(k_rows, v_rows)
        for node, own_row in zip(request.nodes, assigned_rows):
            parent_chain = [] if node.parent is None else node.parent.arena_chain
            if parent_chain is None:
                raise RuntimeError("Cannot store a child before evaluating its parent")
            node.arena_chain = parent_chain + [own_row]

        # Arena operations only queue CUDA work. Move generation and vocab
        # mapping overlap the first device-to-host synchronization.
        per_node = [
            cozy_bridge.project_legal_moves(cozy_board, self._move_vocab)
            for cozy_board in request.boards
        ]

        # One device->host transfer per wave instead of two syncs per node.
        logits = out["logits"].float().cpu()
        value_logits = out["value_logits"].float().cpu()

        # Every remaining tensor op is one call for the whole wave. At budget
        # 2048 a wave carries ~1,321 nodes, and the old shape ran a softmax, a
        # torch.tensor, an index_select and a log_softmax per node over
        # ~31-element rows, where dispatch dominated the arithmetic.
        values = _batched_value_scalars(value_logits)

        id_lists = [ids for ids, _, _, _, _ in per_node]
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
                legal_forcing=forcing,
            )
            for row, (_ids, moves, ucis, forcing, _total) in enumerate(per_node)
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
