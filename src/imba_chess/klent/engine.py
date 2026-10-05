"""Batched one-token-per-game decoding over persistent per-slot K/V caches.

Every slot holds one game. A slot's cache row is [BOS, event_1, ..., event_n];
each step decodes exactly one event token for EVERY slot via the model's
one-query-per-game grouped decode, then writes that token's K/V at the slot's
current length. BOS K/V never changes, so it is computed once and copied in on
reset.
"""

import torch

from imba_chess.data.event_builder import BOS_TOKEN_ID, EVENT_TOKEN_ID
from imba_chess.eval.position_evaluator import _autocast_context
from imba_chess.model import create_batch_dense_mask

TOKEN_KEYS = (
    "piece_ids",
    "turn_id",
    "castle_id",
    "ep_file_id",
    "halfmove_bucket_id",
    "fullmove_bucket_id",
    "prev_move_id",
)
_BUCKET = 64


class SlotEngine:
    def __init__(self, model, *, slots, start_id, device, dtype, max_tokens=None):
        if model.training:
            raise ValueError("SlotEngine requires an evaluation-mode model")
        device = torch.device(device)
        self.model, self.slots, self.device, self.dtype = model, slots, device, dtype
        self.max_tokens = max_tokens or model.config.max_position_embeddings
        cfg = model.config
        shape = (slots, cfg.num_heads, self.max_tokens)
        self.k = [
            torch.zeros(*shape, cfg.attention_dim, dtype=dtype, device=device)
            for _ in model.layers
        ]
        self.v = [
            torch.zeros(*shape, cfg.linear_hidden_dim, dtype=dtype, device=device)
            for _ in model.layers
        ]
        self.lengths = [0] * slots
        bos = dict(
            piece_ids=torch.zeros(1, 64, dtype=torch.long),
            seq_token_id=torch.tensor([BOS_TOKEN_ID]),
            **{key: torch.zeros(1, dtype=torch.long) for key in TOKEN_KEYS[1:-1]},
            prev_move_id=torch.tensor([start_id]),
            seq_offsets=torch.tensor([0, 1]),
            total_tokens=1,
        )
        with torch.inference_mode(), _autocast_context(device, dtype):
            out = model(
                bos,
                block_mask=create_batch_dense_mask(
                    bos["seq_offsets"], total_tokens=1, device=device
                ),
                return_loss=False,
                return_kv=True,
            )
        # kv_caches: per layer (k [H, 1, d], v [H, 1, d]).
        self.bos = [(k[:, 0].to(dtype), v[:, 0].to(dtype)) for k, v in out["kv_caches"]]

    def reset(self, slots):
        index = torch.as_tensor(list(slots), device=self.device)
        for layer, (k, v) in enumerate(self.bos):
            self.k[layer][index, :, 0] = k
            self.v[layer][index, :, 0] = v
        for slot in slots:
            self.lengths[slot] = 1

    @torch.inference_mode()
    def step(self, tokens):
        """tokens: CPU tensors with a leading [slots] dim for TOKEN_KEYS.

        Returns device outputs (logits, q, value_logits) for the new tokens.
        """
        lengths = self.lengths
        if min(lengths) < 1:
            raise RuntimeError("every slot must be reset before decoding")
        if max(lengths) >= self.max_tokens:
            raise RuntimeError("slot context is full")
        width = min(self.max_tokens, -(-max(lengths) // _BUCKET) * _BUCKET)
        positions = torch.tensor(lengths, device=self.device)
        batch = {key: tokens[key].to(self.device, non_blocking=True) for key in TOKEN_KEYS}
        batch["seq_token_id"] = torch.full(
            (self.slots,), EVENT_TOKEN_ID, dtype=torch.long, device=self.device
        )
        with _autocast_context(self.device, self.dtype):
            out = self.model.forward_decode_grouped(
                new_token_batch=batch,
                positions=positions,
                group_index=torch.arange(self.slots, device=self.device),
                prefix_kv_grouped=[
                    (k[:, :, :width], v[:, :, :width]) for k, v in zip(self.k, self.v)
                ],
                prefix_lens=positions,
                prefix_lens_list=lengths,
                group_sizes=[1] * self.slots,
                one_query_per_game=True,
            )
        rows = torch.arange(self.slots, device=self.device)
        for layer, (k_new, v_new) in enumerate(out["kv"]):
            self.k[layer][rows, :, positions] = k_new[:, :, 0].to(self.dtype)
            self.v[layer][rows, :, positions] = v_new[:, :, 0].to(self.dtype)
        self.lengths = [n + 1 for n in lengths]
        return out
