from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn.functional as F

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.event_builder import (
    BOS_TOKEN_ID,
    EVENT_TOKEN_ID,
    TARGET_IGNORE_INDEX,
)
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.search import PositionEval
from imba_chess.model import HSTUChessModel, build_hstu_chess_config, create_batch_block_mask


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
    block_mask = create_batch_block_mask(
        seq_offsets=seq_offsets,
        total_tokens=int(batch["total_tokens"]),
        device=device,
    )
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_amp
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        return model(
            batch, block_mask=block_mask, return_loss=False, return_kv=return_kv
        )


def _value_scalar_from_logits(value_logits_last: torch.Tensor) -> float:
    probs = torch.softmax(value_logits_last.float(), dim=-1)
    return float((probs[2] - probs[0]).item())


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
    legal_ids_tensor = torch.tensor(
        legal_move_ids, device=logits.device, dtype=torch.long
    )
    legal_logits = logits.index_select(0, legal_ids_tensor)
    return legal_logits, legal_moves_with_ids, total_legal, mapped_legal


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
        value_net=None,
        value_net_alpha: float = 1.0,
    ) -> None:
        self._model = model
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder
        self._device = device
        self._dtype = dtype
        self._prefix_kv = prefix_kv
        self._prefix_len = int(prefix_len)
        self._value_net = value_net
        self._value_net_alpha = value_net_alpha

    def extend(self, handle, board_before: chess.Board, move: chess.Move):
        parent = handle if isinstance(handle, _CachedNode) else None
        depth = parent.depth + 1 if parent is not None else 0
        return _CachedNode(parent, int(self._move_vocab.encode(move.uci())), depth)

    def evaluate(self, batch):
        if not batch:
            return []
        nodes: list[_CachedNode] = [handle for handle, _ in batch]
        boards = [board for _, board in batch]
        states = [self._board_state_encoder.encode(board) for board in boards]
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

        use_amp = self._device.type == "cuda" and self._dtype in (
            torch.float16,
            torch.bfloat16,
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            suffix_kv = suffix_positions = suffix_mask = None
            if max_suffix > 0:
                suffix_kv, suffix_positions, suffix_mask = self._wave_suffixes(
                    nodes, max_suffix
                )
            out = self._model.forward_decode(
                new_token_batch=new_token_batch,
                positions=positions,
                prefix_kv=self._prefix_kv,
                suffix_kv=suffix_kv,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
            # Stack the wave's per-layer (k, v) once, then extend each node's
            # root->self path cache so descendants get their suffix for free.
            k_all = torch.stack([k for k, _ in out["kv"]], dim=0)  # [L, B, H, 1, d]
            v_all = torch.stack([v for _, v in out["kv"]], dim=0)
            for row, node in enumerate(nodes):
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

            net_scalars = None
            if self._value_net is not None and self._value_net_alpha > 0.0:
                net_logits = self._value_net(new_token_batch).float().cpu()
                net_probs = torch.softmax(net_logits, dim=-1)
                net_scalars = net_probs[:, 2] - net_probs[:, 0]

        alpha = self._value_net_alpha
        results = []
        for row, board in enumerate(boards):
            value_stm = _value_scalar_from_logits(value_logits[row])
            if net_scalars is not None:
                value_stm = (1.0 - alpha) * value_stm + alpha * float(net_scalars[row])
            try:
                legal_logits, legal_moves, _, _ = _project_legal_logits(
                    logits=logits[row], board=board, move_vocab=self._move_vocab
                )
                log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
            except RuntimeError:
                legal_moves, log_priors = [], []
            results.append(
                PositionEval(
                    value_stm=value_stm,
                    legal_moves=legal_moves,
                    legal_log_priors=log_priors,
                )
            )
        return results

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
