# Prefix-Cache Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each search evaluation O(1 new token) instead of O(full game history) by caching the trunk's per-layer K/V for the shared root prefix and decoding search nodes as single-token queries against it, per `docs/superpowers/specs/2026-07-04-prefix-cache-decode-design.md`.

**Architecture:** The root forward `_select_model_move` already runs becomes the prefill (`return_kv=True`). Every search node adds exactly one token relative to its parent. A new `CachedPositionEvaluator` (replacing `_HistoryPositionEvaluator` outright) implements the same `PositionEvaluator` protocol: `extend` is O(1) bookkeeping, `evaluate` is one batched decode wave using hand-rolled two-part attention (broadcast shared prefix + padded per-node suffixes, softmax over the concatenation). The search module `src/imba_chess/eval/search.py` does not change at all.

**Tech Stack:** PyTorch (plain matmuls for decode — no flex_attention, no compile requirement), python-chess, pytest. fp32 CPU equivalence tests are the correctness gate.

## Global Constraints

- **No training-path change** beyond the additive `return_kv` flag: `forward(batch, *, block_mask=None, return_loss=True)` callers with no new kwargs must behave byte-identically. No weight/checkpoint changes.
- Decode attention must replicate flex_attention semantics exactly: scale `1/sqrt(attention_dim)` applied to `q·k` **before** adding the relative bias; bias = `_ps_w[h, clamp((k_pos - q_pos) + max_seq_len - 1, 0, 2*max_seq_len - 2)]`; softmax over all attended keys; the new token **attends to itself** (causal `<=`).
- Cached K/V are the **post-silu** per-head tensors exactly as flex_attention consumes them: prefix `[H, T, d]`, per-node new-token `[H, 1, d]` (d = `attention_dim` for K, `linear_hidden_dim` for V, per head).
- Equivalence gates run on fp32 CPU with tolerance `atol=1e-5, rtol=1e-5`.
- Replace outright: `_HistoryPositionEvaluator`, `_forward_last_token_outputs`, `_merge_single_sequence_batches`, `_SEARCH_EVAL_MAX_TOKENS_PER_CHUNK`, and `_SequenceHistory.clone()` are deleted; no fallback flag; no new config keys.
- `src/imba_chess/eval/search.py` and `tests/test_search.py` must be untouched by this plan.
- Test command: `.venv/bin/python -m pytest ...` (in a worktree: prefix with `PYTHONPATH=src`).

---

### Task 1: Layer-level decode + `return_kv` + `PositionEmbedding.at_positions`

**Files:**
- Modify: `src/imba_chess/model/hstu_attention.py`
- Modify: `src/imba_chess/model/position_embedding.py`
- Test: `tests/test_prefix_decode.py` (new)

**Interfaces:**
- Consumes: existing `SequentialTransductionUnitJagged.forward(x, block_mask)`.
- Produces (used by Task 2):
  - `SequentialTransductionUnitJagged.forward(x, block_mask=None, return_kv=False)` — when `return_kv=True` returns `(out [S, D], (k [H, S, d_qk], v [H, S, d_v]))`, else `out` exactly as today.
  - `SequentialTransductionUnitJagged.forward_decode(x_new [B, D], *, prefix_k [H, T, d_qk], prefix_v [H, T, d_v], q_positions [B], suffix_k [B, H, s, d_qk] | None, suffix_v [B, H, s, d_v] | None, suffix_positions [B, s] | None, suffix_mask [B, s] bool | None) -> (x_out [B, D], k_new [B, H, 1, d_qk], v_new [B, H, 1, d_v])`.
  - `PositionEmbedding.at_positions(content [B, D], positions [B]) -> [B, D]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prefix_decode.py`:

```python
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imba_chess.model import create_batch_block_mask
from imba_chess.model.hstu_attention import SequentialTransductionUnitJagged
from imba_chess.model.position_embedding import PositionEmbedding

ATOL = 1e-5
RTOL = 1e-5


def _layer() -> SequentialTransductionUnitJagged:
    torch.manual_seed(0)
    return SequentialTransductionUnitJagged(
        embedding_dim=32,
        linear_hidden_dim=8,
        attention_dim=8,
        dropout_ratio=0.0,
        num_heads=2,
        max_seq_len=64,
    ).eval()


def _full_forward(layer, x):
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, x.size(0)]),
        total_tokens=x.size(0),
        device=x.device,
    )
    return layer(x=x, block_mask=block_mask)


def test_forward_return_kv_output_unchanged():
    layer = _layer()
    x = torch.randn(10, 32)
    full = _full_forward(layer, x)
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, 10]), total_tokens=10, device=x.device
    )
    out, (k, v) = layer(x=x, block_mask=block_mask, return_kv=True)
    torch.testing.assert_close(out, full, atol=ATOL, rtol=RTOL)
    assert k.shape == (2, 10, 8)  # [H, S, attention_dim]
    assert v.shape == (2, 10, 8)  # [H, S, linear_hidden_dim]


def test_layer_decode_matches_full_forward_token_by_token():
    layer = _layer()
    S, T = 13, 9  # prefill 9 tokens, decode tokens 9..12 sequentially
    x = torch.randn(S, 32)
    full = _full_forward(layer, x)

    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, T]), total_tokens=T, device=x.device
    )
    out_prefix, (prefix_k, prefix_v) = layer(
        x=x[:T], block_mask=block_mask, return_kv=True
    )
    torch.testing.assert_close(out_prefix, full[:T], atol=ATOL, rtol=RTOL)

    suffix_k_parts: list[torch.Tensor] = []
    suffix_v_parts: list[torch.Tensor] = []
    for i in range(T, S):
        if suffix_k_parts:
            suffix_k = torch.cat(suffix_k_parts, dim=2)  # [1, H, s, d]
            suffix_v = torch.cat(suffix_v_parts, dim=2)
            s = suffix_k.size(2)
            suffix_positions = torch.arange(T, T + s).view(1, s)
            suffix_mask = torch.ones(1, s, dtype=torch.bool)
        else:
            suffix_k = suffix_v = suffix_positions = suffix_mask = None
        x_out, k_new, v_new = layer.forward_decode(
            x[i : i + 1],
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            q_positions=torch.tensor([i]),
            suffix_k=suffix_k,
            suffix_v=suffix_v,
            suffix_positions=suffix_positions,
            suffix_mask=suffix_mask,
        )
        torch.testing.assert_close(x_out.squeeze(0), full[i], atol=ATOL, rtol=RTOL)
        suffix_k_parts.append(k_new)
        suffix_v_parts.append(v_new)


def test_layer_decode_batched_wave_with_mixed_suffix_lengths():
    layer = _layer()
    T = 7
    prefix = torch.randn(T, 32)
    block_mask = create_batch_block_mask(
        seq_offsets=torch.tensor([0, T]), total_tokens=T, device=prefix.device
    )
    _, (prefix_k, prefix_v) = layer(x=prefix, block_mask=block_mask, return_kv=True)

    # Node A: depth 0 (no suffix). Node B: depth 1 (one ancestor token).
    tok_a = torch.randn(1, 32)
    tok_b_parent = torch.randn(1, 32)
    tok_b = torch.randn(1, 32)

    # References via full forwards over explicit sequences.
    full_a = _full_forward(layer, torch.cat([prefix, tok_a]))[T]
    full_b = _full_forward(layer, torch.cat([prefix, tok_b_parent, tok_b]))[T + 1]

    # Evaluate B's parent first to obtain its (k, v).
    _, kp, vp = layer.forward_decode(
        tok_b_parent,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        q_positions=torch.tensor([T]),
    )

    # One wave containing A (depth 0, padded suffix) and B (depth 1).
    x_new = torch.cat([tok_a, tok_b])  # [2, 32]
    suffix_k = torch.cat([torch.zeros_like(kp), kp])  # [2, H, 1, d]
    suffix_v = torch.cat([torch.zeros_like(vp), vp])
    suffix_positions = torch.tensor([[0], [T]])
    suffix_mask = torch.tensor([[False], [True]])
    x_out, _, _ = layer.forward_decode(
        x_new,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        q_positions=torch.tensor([T, T + 1]),
        suffix_k=suffix_k,
        suffix_v=suffix_v,
        suffix_positions=suffix_positions,
        suffix_mask=suffix_mask,
    )
    torch.testing.assert_close(x_out[0], full_a, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(x_out[1], full_b, atol=ATOL, rtol=RTOL)


def test_position_embedding_at_positions_matches_forward():
    torch.manual_seed(1)
    pe = PositionEmbedding(max_seq_len=16, embedding_dim=8, dropout_rate=0.0).eval()
    content = torch.randn(5, 8)
    offsets = torch.tensor([0, 5])
    full = pe(content, offsets)
    picked = pe.at_positions(content, torch.arange(5))
    torch.testing.assert_close(picked, full, atol=ATOL, rtol=RTOL)
    # Clamp behavior matches forward's clamp.
    over = pe.at_positions(content[:1], torch.tensor([99]))
    ref = pe.at_positions(content[:1], torch.tensor([15]))
    torch.testing.assert_close(over, ref, atol=ATOL, rtol=RTOL)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_decode.py -v`
Expected: FAIL — `test_forward_return_kv_output_unchanged` with `TypeError` (unexpected keyword `return_kv`), others with `AttributeError` (`forward_decode` / `at_positions`).

- [ ] **Step 3: Add `at_positions` to `src/imba_chess/model/position_embedding.py`**

Append inside `PositionEmbedding`:

```python
    def at_positions(
        self, content: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Same combine as forward() but with caller-supplied absolute positions.

        Used by the decode path, where positions are prefix_len + suffix depth
        rather than derived from jagged offsets.
        """
        positions = torch.clamp(positions, max=self.max_seq_len - 1)
        x = content * (self._embedding_dim**0.5) + self.embedding(positions)
        return self.dropout(x)
```

- [ ] **Step 4: Add `return_kv` and `forward_decode` to `src/imba_chess/model/hstu_attention.py`**

Replace the existing `forward` method with:

```python
    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask | None = None,
        return_kv: bool = False,
    ):
        # x: [S, D] — total tokens across all sessions
        S = x.size(0)
        x = x.unsqueeze(0)
        normed_x = self._norm_input(x)
        uvqk_x = self._uvqk(
            normed_x
        )  # shape: [1, S, linear_dim * 2 * num_heads + attention_dim * 2 * num_heads]
        uvqk_x = F.silu(uvqk_x)
        u, v, q, k = torch.split(
            uvqk_x,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )

        q_heads = self._reshape_uvqk_for_mm(q, self._num_heads, self._attention_dim)
        k_heads = self._reshape_uvqk_for_mm(k, self._num_heads, self._attention_dim)
        v_heads = self._reshape_uvqk_for_mm(v, self._num_heads, self._linear_dim)

        # output shape: [1, num_heads, S, linear_dim]
        attn_output: torch.Tensor = flex_attention(
            query=q_heads,
            key=k_heads,
            value=v_heads,
            block_mask=block_mask,
            score_mod=self._generate_rab_score_mod(),
            kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1},
        )  # type: ignore

        attn_output = self._norm_attn_output(
            attn_output.permute(0, 2, 1, 3).reshape(
                1, S, self._num_heads * self._linear_dim
            )
        )

        o_input = F.dropout(
            u * attn_output, p=self._dropout_ratio, training=self.training
        )
        out = (self._o(o_input) + x).squeeze(0)
        if return_kv:
            return out, (k_heads.squeeze(0), v_heads.squeeze(0))
        return out

    def _relative_bias(
        self, k_positions: torch.Tensor, q_positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-head relative bias for decode: [B, H, 1, K] from positions.

        k_positions: [B, K] (or [1, K] broadcastable), q_positions: [B].
        Replicates _position_score_mod's clamped (k_idx - q_idx) indexing.
        """
        rel = torch.clamp(
            k_positions - q_positions.view(-1, 1) + (self._max_seq_len - 1),
            0,
            2 * self._max_seq_len - 2,
        )  # [B, K]
        # _ps_w: [H, 2*max-1]; gather -> [H, B, K] -> [B, H, 1, K]
        return self._ps_w[:, rel].permute(1, 0, 2).unsqueeze(2)

    def forward_decode(
        self,
        x_new: torch.Tensor,
        *,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        q_positions: torch.Tensor,
        suffix_k: torch.Tensor | None = None,
        suffix_v: torch.Tensor | None = None,
        suffix_positions: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one new token per batch row against cached K/V.

        x_new: [B, D]; prefix_k/v: [H, T, d] shared across rows;
        suffix_k/v: [B, H, s, d] per-row ancestor tokens (zero-padded, with
        suffix_mask [B, s] marking real entries); q_positions/suffix_positions
        are absolute within each row's virtual sequence. The new token always
        attends to prefix + its real suffix + itself (causal <=), replicating
        forward()'s flex_attention semantics: scores scaled by
        1/sqrt(attention_dim) then biased by _ps_w, softmax over all keys.

        Returns (x_out [B, D], k_new [B, H, 1, d_qk], v_new [B, H, 1, d_v]).
        """
        batch_size = x_new.size(0)
        x = x_new.unsqueeze(1)  # [B, 1, D]
        normed_x = self._norm_input(x)
        uvqk_x = F.silu(self._uvqk(normed_x))
        u, v, q, k = torch.split(
            uvqk_x,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )
        q_heads = self._reshape_uvqk_for_mm(q, self._num_heads, self._attention_dim)
        k_new = self._reshape_uvqk_for_mm(k, self._num_heads, self._attention_dim)
        v_new = self._reshape_uvqk_for_mm(v, self._num_heads, self._linear_dim)

        scale = self._attention_dim**-0.5
        prefix_len = prefix_k.size(1)
        device = x_new.device
        bias_dtype = q_heads.dtype

        # Scores vs the shared prefix (broadcast, never materialized per row).
        prefix_scores = (
            torch.einsum("bhqd,htd->bhqt", q_heads, prefix_k.to(q_heads.dtype)) * scale
        )
        prefix_positions = torch.arange(prefix_len, device=device).view(1, prefix_len)
        prefix_scores = prefix_scores + self._relative_bias(
            prefix_positions, q_positions
        ).to(bias_dtype)

        score_parts = [prefix_scores]
        has_suffix = suffix_k is not None and suffix_k.size(2) > 0
        if has_suffix:
            suffix_scores = (
                torch.einsum("bhqd,bhsd->bhqs", q_heads, suffix_k.to(q_heads.dtype))
                * scale
            )
            suffix_scores = suffix_scores + self._relative_bias(
                suffix_positions, q_positions
            ).to(bias_dtype)
            suffix_scores = suffix_scores.masked_fill(
                ~suffix_mask.view(batch_size, 1, 1, -1), float("-inf")
            )
            score_parts.append(suffix_scores)

        # Self-attention term: distance 0.
        self_scores = (q_heads * k_new).sum(dim=-1, keepdim=True) * scale
        self_scores = self_scores + self._ps_w[:, self._max_seq_len - 1].view(
            1, -1, 1, 1
        ).to(bias_dtype)
        score_parts.append(self_scores)

        scores = torch.cat(score_parts, dim=-1)  # [B, H, 1, T + s + 1]
        weights = torch.softmax(scores.float(), dim=-1).to(q_heads.dtype)

        attn_output = torch.einsum(
            "bhqt,htd->bhqd", weights[..., :prefix_len], prefix_v.to(weights.dtype)
        )
        offset = prefix_len
        if has_suffix:
            suffix_len = suffix_k.size(2)
            attn_output = attn_output + torch.einsum(
                "bhqs,bhsd->bhqd",
                weights[..., offset : offset + suffix_len],
                suffix_v.to(weights.dtype),
            )
            offset += suffix_len
        attn_output = attn_output + weights[..., offset:] * v_new

        attn_output = self._norm_attn_output(
            attn_output.permute(0, 2, 1, 3).reshape(
                batch_size, 1, self._num_heads * self._linear_dim
            )
        )
        o_input = F.dropout(
            u * attn_output, p=self._dropout_ratio, training=self.training
        )
        x_out = (self._o(o_input) + x).squeeze(1)
        return x_out, k_new, v_new
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prefix_decode.py -v`
Expected: PASS (4 tests). If the token-by-token test fails with small but nonzero differences, check the bias-before/after-scale order and the self-term's distance-0 bias index (`_ps_w[:, max_seq_len - 1]`). If only the batched-wave test fails, check the suffix mask fill and `at_positions` is not involved (layer-level only).

- [ ] **Step 6: Run the model test suite for regressions**

Run: `.venv/bin/python -m pytest tests/test_hstu_model.py -q`
Expected: PASS (unchanged count — `return_kv` defaults preserve behavior).

- [ ] **Step 7: Commit**

```bash
git add src/imba_chess/model/hstu_attention.py src/imba_chess/model/position_embedding.py tests/test_prefix_decode.py
git commit -m "feat: layer-level KV prefill/decode with exact-equivalence tests"
```

---

### Task 2: Model-level `forward(return_kv)` + `forward_decode`

**Files:**
- Modify: `src/imba_chess/model/hstu_model.py`
- Test: `tests/test_prefix_decode.py` (append)

**Interfaces:**
- Consumes (Task 1): layer `forward(..., return_kv=True)` and `forward_decode(...)`, `PositionEmbedding.at_positions`.
- Produces (used by Task 3):
  - `HSTUChessModel.forward(batch, *, block_mask=None, return_loss=True, return_kv=False)` — when `return_kv=True`, output dict gains `"kv_caches": list[tuple[Tensor [H,S,d_qk], Tensor [H,S,d_v]]]` (one per layer).
  - `HSTUChessModel.forward_decode(*, new_token_batch: dict[str, Tensor], positions: Tensor [B], prefix_kv: list[tuple[Tensor, Tensor]], suffix_kv: list[tuple[Tensor [B,H,s,d_qk], Tensor [B,H,s,d_v]]] | None = None, suffix_positions: Tensor [B,s] | None = None, suffix_mask: Tensor [B,s] | None = None) -> dict` with keys `"logits" [B, V]`, `"value_logits" [B, 3]` (when the head exists), `"kv": list[tuple[Tensor [B,H,1,d_qk], Tensor [B,H,1,d_v]]]`. `new_token_batch` needs exactly the per-token id keys `_build_content` reads: `piece_ids [B,64]`, `seq_token_id`, `turn_id`, `castle_id`, `ep_file_id`, `halfmove_bucket_id`, `fullmove_bucket_id`, `prev_move_id` (each `[B]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prefix_decode.py`:

```python
from imba_chess.config import ModelConfig
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


def _tiny_model(vocab_size: int = 32) -> HSTUChessModel:
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
            enable_value_head=True,
        ),
        move_vocab_size=vocab_size,
    )
    return HSTUChessModel(config).eval()


def _random_token_ids(n: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "piece_ids": torch.randint(0, 13, (n, 64), generator=g),
        "seq_token_id": torch.randint(0, 2, (n,), generator=g),
        "turn_id": torch.randint(0, 2, (n,), generator=g),
        "castle_id": torch.randint(0, 16, (n,), generator=g),
        "ep_file_id": torch.randint(0, 9, (n,), generator=g),
        "halfmove_bucket_id": torch.randint(0, 50, (n,), generator=g),
        "fullmove_bucket_id": torch.randint(0, 100, (n,), generator=g),
        "prev_move_id": torch.randint(0, 32, (n,), generator=g),
    }


def _full_batch(token_ids: dict[str, torch.Tensor]) -> dict:
    n = token_ids["piece_ids"].size(0)
    batch = dict(token_ids)
    batch.update(
        {
            "total_tokens": n,
            "seq_offsets": torch.tensor([0, n]),
            "target_move_id": torch.full((n,), -100, dtype=torch.long),
        }
    )
    return batch


def test_model_decode_matches_full_forward_over_depths():
    model = _tiny_model()
    T, max_depth = 9, 4
    ids = _random_token_ids(T + max_depth, seed=7)
    prefix_ids = {key: value[:T] for key, value in ids.items()}

    with torch.no_grad():
        full = model(_full_batch(ids), return_loss=False)
        prefill = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)

    prefix_kv = prefill["kv_caches"]
    suffix_kv = None
    suffix_positions = suffix_mask = None
    for depth in range(max_depth):
        i = T + depth
        step_ids = {key: value[i : i + 1] for key, value in ids.items()}
        with torch.no_grad():
            out = model.forward_decode(
                new_token_batch=step_ids,
                positions=torch.tensor([i]),
                prefix_kv=prefix_kv,
                suffix_kv=suffix_kv,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
        torch.testing.assert_close(
            out["logits"].squeeze(0), full["logits"][i], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out["value_logits"].squeeze(0),
            full["value_logits"][i],
            atol=ATOL,
            rtol=RTOL,
        )
        # Grow the suffix cache with this token's per-layer (k, v).
        if suffix_kv is None:
            suffix_kv = [(k, v) for k, v in out["kv"]]
        else:
            suffix_kv = [
                (torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2))
                for (sk, sv), (k, v) in zip(suffix_kv, out["kv"])
            ]
        s = suffix_kv[0][0].size(2)
        suffix_positions = torch.arange(T, T + s).view(1, s)
        suffix_mask = torch.ones(1, s, dtype=torch.bool)


def test_model_decode_mixed_depth_wave():
    model = _tiny_model()
    T = 8
    ids = _random_token_ids(T + 3, seed=11)  # prefix + [a, b_parent, b]
    prefix_ids = {key: value[:T] for key, value in ids.items()}
    tok_a = {key: value[T : T + 1] for key, value in ids.items()}
    tok_bp = {key: value[T + 1 : T + 2] for key, value in ids.items()}
    tok_b = {key: value[T + 2 : T + 3] for key, value in ids.items()}

    seq_a = {key: torch.cat([prefix_ids[key], tok_a[key]]) for key in ids}
    seq_b = {
        key: torch.cat([prefix_ids[key], tok_bp[key], tok_b[key]]) for key in ids
    }
    with torch.no_grad():
        full_a = model(_full_batch(seq_a), return_loss=False)
        full_b = model(_full_batch(seq_b), return_loss=False)
        prefill = model(_full_batch(prefix_ids), return_loss=False, return_kv=True)
        parent_out = model.forward_decode(
            new_token_batch=tok_bp,
            positions=torch.tensor([T]),
            prefix_kv=prefill["kv_caches"],
        )
        wave_ids = {key: torch.cat([tok_a[key], tok_b[key]]) for key in ids}
        suffix_kv = [
            (
                torch.cat([torch.zeros_like(k), k], dim=0),
                torch.cat([torch.zeros_like(v), v], dim=0),
            )
            for k, v in parent_out["kv"]
        ]
        wave = model.forward_decode(
            new_token_batch=wave_ids,
            positions=torch.tensor([T, T + 1]),
            prefix_kv=prefill["kv_caches"],
            suffix_kv=suffix_kv,
            suffix_positions=torch.tensor([[0], [T]]),
            suffix_mask=torch.tensor([[False], [True]]),
        )
    torch.testing.assert_close(
        wave["logits"][0], full_a["logits"][T], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        wave["logits"][1], full_b["logits"][T + 1], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        wave["value_logits"][1], full_b["value_logits"][T + 1], atol=ATOL, rtol=RTOL
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prefix_decode.py -v -k model_decode`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'return_kv'`.

- [ ] **Step 3: Implement in `src/imba_chess/model/hstu_model.py`**

3a. Change `forward`'s signature and trunk loop. Current:

```python
    def forward(
        self,
        batch: dict[str, Any],
        *,
        block_mask: BlockMask | None = None,
        return_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
```

becomes:

```python
    def forward(
        self,
        batch: dict[str, Any],
        *,
        block_mask: BlockMask | None = None,
        return_loss: bool = True,
        return_kv: bool = False,
    ) -> dict[str, torch.Tensor]:
```

and the trunk loop:

```python
        for layer in self.layers:
            x = layer(x=x, block_mask=block_mask)
```

becomes:

```python
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            if return_kv:
                x, layer_kv = layer(x=x, block_mask=block_mask, return_kv=True)
                kv_caches.append(layer_kv)
            else:
                x = layer(x=x, block_mask=block_mask)
```

and immediately after `output` is first constructed (after the
`"policy_logits"` line), add:

```python
        if return_kv:
            output["kv_caches"] = kv_caches  # type: ignore[assignment]
```

3b. Add `forward_decode` as a method right after `forward`:

```python
    def forward_decode(
        self,
        *,
        new_token_batch: dict[str, Any],
        positions: torch.Tensor,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]],
        suffix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        suffix_positions: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Decode one new token per batch row against per-layer cached K/V.

        Inference-only companion to forward(return_kv=True): new_token_batch
        carries the per-token id tensors _build_content reads; positions are
        absolute (prefix_len + suffix depth). Returns logits/value_logits at
        the new tokens plus each layer's (k, v) for growing suffix caches.
        """
        assert not self.training, "forward_decode is inference-only"
        device = self.piece_square_embedding.weight.device
        positions = positions.to(device=device, dtype=torch.long)
        content = self._build_content(new_token_batch)
        x = self.position_embedding.at_positions(content, positions)

        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.layers):
            prefix_k, prefix_v = prefix_kv[layer_idx]
            if suffix_kv is not None:
                layer_suffix_k, layer_suffix_v = suffix_kv[layer_idx]
            else:
                layer_suffix_k = layer_suffix_v = None
            x, k_new, v_new = layer.forward_decode(
                x,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                q_positions=positions,
                suffix_k=layer_suffix_k,
                suffix_v=layer_suffix_v,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )
            new_kv.append((k_new, v_new))

        x = self.final_norm(x)
        output: dict[str, Any] = {
            "logits": self.prediction_head(x),
            "kv": new_kv,
        }
        if self.value_head is not None:
            output["value_logits"] = self.value_head(x)
        return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prefix_decode.py -v`
Expected: PASS (6 tests). If depth-0 passes but deeper depths fail, the suffix position bookkeeping is off by one; if only value_logits mismatch, `final_norm`/head wiring differs from `forward`.

- [ ] **Step 5: Full-suite regression check**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all existing tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/model/hstu_model.py tests/test_prefix_decode.py
git commit -m "feat: model-level prefill/decode API with equivalence tests"
```

---

### Task 3: `CachedPositionEvaluator` — switch the eval script, rewrite dummies

**Files:**
- Modify: `scripts/eval_vs_stockfish.py`
- Modify: `tests/test_eval_vs_stockfish.py` (dummy models + `_select_model_move` call sites)

**Interfaces:**
- Consumes (Task 2): `forward(..., return_kv=True)` → `output["kv_caches"]`; `HSTUChessModel.forward_decode(...)` exactly as specified in Task 2's Produces block.
- Produces: `CachedPositionEvaluator(model, move_vocab, board_state_encoder, device, dtype, prefix_kv, prefix_len)` implementing the `PositionEvaluator` protocol; `_select_model_move` gains required kwarg `board_state_encoder` and passes `root_handle=None` to strategies.

- [ ] **Step 1: Delete the replaced plumbing in `scripts/eval_vs_stockfish.py`**

Delete entirely: the `_HistoryPositionEvaluator` class, `_forward_last_token_outputs`, `_merge_single_sequence_batches`, the `_SEARCH_EVAL_MAX_TOKENS_PER_CHUNK` constant (and its comment), and the `clone` method of `_SequenceHistory`.

- [ ] **Step 2: Extend `_forward_model` with `return_kv`**

Current signature `def _forward_model(*, model, batch, device, dtype)` becomes
`def _forward_model(*, model, batch, device, dtype, return_kv: bool = False)`, and its final line

```python
    with torch.inference_mode(), autocast_ctx:
        return model(batch, block_mask=block_mask, return_loss=False)
```

becomes

```python
    with torch.inference_mode(), autocast_ctx:
        return model(
            batch, block_mask=block_mask, return_loss=False, return_kv=return_kv
        )
```

- [ ] **Step 3: Add `CachedPositionEvaluator` where `_HistoryPositionEvaluator` was**

```python
class _CachedNode:
    """Search-node handle: parent link + the move that led here.

    kv is filled after this node is evaluated: one (k [H,1,d_qk], v [H,1,d_v])
    per trunk layer. Parents are always evaluated before children in every
    strategy, so ancestor chains are complete at evaluate() time.
    """

    __slots__ = ("parent", "move_id", "depth", "kv")

    def __init__(self, parent: "_CachedNode | None", move_id: int, depth: int) -> None:
        self.parent = parent
        self.move_id = move_id
        self.depth = depth
        self.kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None


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

    def extend(self, handle, board_before: chess.Board, move: chess.Move):
        parent = handle if isinstance(handle, _CachedNode) else None
        depth = parent.depth + 1 if parent is not None else 0
        return _CachedNode(parent, int(self._move_vocab.encode(move.uci())), depth)

    def evaluate(self, batch):
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

        chains: list[list[_CachedNode]] = []
        for node in nodes:
            chain: list[_CachedNode] = []
            ancestor = node.parent
            while ancestor is not None:
                chain.append(ancestor)
                ancestor = ancestor.parent
            chain.reverse()  # oldest first; chain[i].depth == i
            chains.append(chain)
        max_suffix = max(len(chain) for chain in chains)

        suffix_kv = suffix_positions = suffix_mask = None
        if max_suffix > 0:
            num_layers = len(self._prefix_kv)
            suffix_kv = []
            for layer_idx in range(num_layers):
                ref_k, ref_v = self._prefix_kv[layer_idx]
                heads = ref_k.size(0)
                layer_k = torch.zeros(
                    wave_size, heads, max_suffix, ref_k.size(-1),
                    dtype=ref_k.dtype, device=ref_k.device,
                )
                layer_v = torch.zeros(
                    wave_size, heads, max_suffix, ref_v.size(-1),
                    dtype=ref_v.dtype, device=ref_v.device,
                )
                for row, chain in enumerate(chains):
                    for i, ancestor in enumerate(chain):
                        ancestor_k, ancestor_v = ancestor.kv[layer_idx]
                        layer_k[row, :, i : i + 1] = ancestor_k
                        layer_v[row, :, i : i + 1] = ancestor_v
                suffix_kv.append((layer_k, layer_v))
            suffix_positions = (
                torch.arange(max_suffix).view(1, -1) + self._prefix_len
            ).expand(wave_size, -1).to(self._device)
            suffix_mask = torch.tensor(
                [
                    [i < len(chain) for i in range(max_suffix)]
                    for chain in chains
                ],
                dtype=torch.bool,
                device=self._device,
            )

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
            out = self._model.forward_decode(
                new_token_batch=new_token_batch,
                positions=positions,
                prefix_kv=self._prefix_kv,
                suffix_kv=suffix_kv,
                suffix_positions=suffix_positions,
                suffix_mask=suffix_mask,
            )

        for row, node in enumerate(nodes):
            node.kv = [
                (k[row : row + 1].squeeze(0), v[row : row + 1].squeeze(0))
                for k, v in out["kv"]
            ]

        logits = out["logits"]
        value_logits = out["value_logits"]
        results = []
        for row, board in enumerate(boards):
            value_stm = _value_scalar_from_logits(value_logits[row])
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
```

Note: `node.kv` entries are stored as `[H, 1, d]` (squeezed batch dim) and re-inserted into `[wave, H, s, d]` slots — shapes line up because `layer_k[row, :, i:i+1]` is `[H, 1, d]`.

- [ ] **Step 4: Rewire `_select_model_move`**

Add required kwarg `board_state_encoder: BoardStateEncoder` (after `move_vocab`). Change the root forward call and evaluator construction from:

```python
    output = _forward_model(
        model=model,
        batch=batch,
        device=device,
        dtype=dtype,
    )
    ...
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    evaluator = _HistoryPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        device=device,
        dtype=dtype,
        policy_name=policy,
    )
```

to:

```python
    output = _forward_model(
        model=model,
        batch=batch,
        device=device,
        dtype=dtype,
        return_kv=True,
    )
    ...
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    evaluator = CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        device=device,
        dtype=dtype,
        prefix_kv=output["kv_caches"],
        prefix_len=int(batch["total_tokens"]),
    )
```

and in every strategy dispatch replace `root_handle=history,` with `root_handle=None,` (the `history` parameter stays — the game loop still owns it — but strategies no longer receive it).

In `_run_segment`, pass `board_state_encoder=board_state_encoder,` in the `_select_model_move(...)` call.

- [ ] **Step 5: Rewrite the dummy models in `tests/test_eval_vs_stockfish.py`**

Replace all five dummy model classes with cached-protocol versions. Key contract: `forward(batch, *, block_mask=None, return_loss=False, return_kv=False)` (root call; add `kv_caches` when `return_kv`), and `forward_decode(*, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None)` keyed on the new token's own features. Both increment `forward_calls`. Full replacement code:

```python
def _dummy_kv(total_tokens: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # One fake layer; the evaluator only threads shapes through.
    return [(torch.zeros(1, total_tokens, 1), torch.zeros(1, total_tokens, 1))]


def _dummy_decode_kv(batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.zeros(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))]


class _DummyValueRerankModel(torch.nn.Module):
    """Root prefers e2e4; value head favors positions reached via d2d4."""

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        value_logits[last, 1] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        prev_ids = new_token_batch["prev_move_id"]
        for row in range(batch_size):
            move_id = int(prev_ids[row].item())
            if move_id == self.move_vocab.token_to_id["e2e4"]:
                value_logits[row] = torch.tensor([0.0, 0.0, 4.0])
            elif move_id == self.move_vocab.token_to_id["d2d4"]:
                value_logits[row] = torch.tensor([4.0, 0.0, 0.0])
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }


class _DummyNoValueModel(torch.nn.Module):
    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 1.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 0.5
        out = {"logits": logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out


class _DummyValueSearchD2Model(torch.nn.Module):
    """Depth-1 nodes get opponent priors; depth-2 values depend on the line.

    The root move is recovered from the node's board (piece_ids): after e2e4
    a white pawn (id 1) sits on e4 (square 28); after d2d4, on d4 (27).
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        value_logits[last, 1] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        prev_ids = new_token_batch["prev_move_id"]
        piece_ids = new_token_batch["piece_ids"]
        root_moves = {
            self.move_vocab.token_to_id["e2e4"],
            self.move_vocab.token_to_id["d2d4"],
        }
        for row in range(batch_size):
            prev = int(prev_ids[row].item())
            if prev in root_moves:
                # Depth 1: opponent to move after our root move.
                logits[row, self.move_vocab.token_to_id["e7e5"]] = 3.0
                logits[row, self.move_vocab.token_to_id["d7d5"]] = 2.5
                value_logits[row, 1] = 1.0
                continue
            # Depth 2: root move recovered from the board.
            root_is_e4 = int(piece_ids[row, chess.E4].item()) == 1
            if prev == self.move_vocab.token_to_id["e7e5"]:
                value_logits[row] = (
                    torch.tensor([4.0, 0.0, 0.0])
                    if root_is_e4
                    else torch.tensor([0.0, 1.0, 2.0])
                )
            elif prev == self.move_vocab.token_to_id["d7d5"]:
                value_logits[row] = (
                    torch.tensor([2.0, 1.0, 0.0])
                    if root_is_e4
                    else torch.tensor([0.0, 0.0, 4.0])
                )
            else:
                value_logits[row, 1] = 1.0
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }


class _DummyMatePreferenceModel(torch.nn.Module):
    """Prefers a quiet move by policy logit; only the value modes should find mate."""

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["a1b1"]] = 4.0
        logits[last, self.move_vocab.token_to_id["a1a8"]] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        return {
            "logits": torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32),
            "value_logits": torch.zeros((batch_size, 3), dtype=torch.float32),
            "kv": _dummy_decode_kv(batch_size),
        }


class _DummyHalvingModel(torch.nn.Module):
    """Root policy prefers e2e4; value head says the d2d4 subtree is winning.

    Value is read from the side-to-move POV, so the sign is keyed on the new
    token's turn_id; the root move is recovered from the board's d4 square.
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(self, *, new_token_batch, positions, prefix_kv, suffix_kv=None, suffix_positions=None, suffix_mask=None):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        piece_ids = new_token_batch["piece_ids"]
        turn_ids = new_token_batch["turn_id"]
        for row in range(batch_size):
            logits[row, self.move_vocab.token_to_id["e2e4"]] = 4.0
            logits[row, self.move_vocab.token_to_id["d2d4"]] = 3.0
            good_for_white = int(piece_ids[row, chess.D4].item()) == 1
            stm_is_white = int(turn_ids[row].item()) == 0
            if good_for_white == stm_is_white:
                value_logits[row] = torch.tensor([0.0, 0.0, 3.0])
            else:
                value_logits[row] = torch.tensor([3.0, 0.0, 0.0])
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }
```

Also update every `module._select_model_move(...)` call in the test file to add `board_state_encoder=BoardStateEncoder(),` after `move_vocab=move_vocab,`. The `forward_calls` assertions stay `2`, `3`, `2 -> stays as currently written (1 for both mate tests)`, because one decode wave replaces one chunked batch call exactly (rerank: root + 1 wave = 2; d2: root + board1 wave + board2 wave = 3; mate short-circuits: 1). Update each assertion's comment to say "prefill + N decode waves".

- [ ] **Step 6: Run the eval test suite**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py tests/test_search.py -v`
Expected: PASS. `tests/test_search.py` must be untouched and passing (its dummies are torch-free `PositionEvaluator`s, unaffected by the model protocol change).

- [ ] **Step 7: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/eval_vs_stockfish.py tests/test_eval_vs_stockfish.py
git commit -m "feat: switch eval search to CachedPositionEvaluator (prefix-cache decode)"
```

---

### Task 4: End-to-end equivalence vs full-forward reference + cleanup audit

**Files:**
- Test: `tests/test_prefix_decode.py` (append)
- Verify-only: `scripts/eval_vs_stockfish.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3; `_SequenceHistory`, `_select_model_move`, `CachedPositionEvaluator` from the script (loaded via the test file's `_load_eval_script_module` pattern); `select_value_search_d2` from `imba_chess.eval.search`.

- [ ] **Step 1: Write the failing test (it fails only if Tasks 1–3 broke something — this is the gate that ties them together)**

Append to `tests/test_prefix_decode.py`:

```python
import importlib.util
import sys
from pathlib import Path

import chess

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.eval.search import PositionEval, select_value_search_d2


def _load_eval_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_vs_stockfish.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_vs_stockfish_script_pd", script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FullForwardReferenceEvaluator:
    """Reference PositionEvaluator: rebuilds every sequence and runs a full
    forward — the uncached ground truth the cached path must reproduce."""

    def __init__(self, *, module, model, move_vocab, board_state_encoder, played):
        self._module = module
        self._model = model
        self._move_vocab = move_vocab
        self._encoder = board_state_encoder
        self._played = played  # list[(board_before, move_uci)] real game so far

    def _fresh_history(self):
        history = self._module._SequenceHistory(
            move_vocab=self._move_vocab, board_state_encoder=self._encoder
        )
        for board_before, move_uci in self._played:
            history.append_observed_position(board_before)
            history.record_played_move(move_uci)
        return history

    def extend(self, handle, board_before, move):
        line = list(handle) if handle is not None else []
        return line + [(board_before.copy(stack=False), move.uci())]

    def evaluate(self, batch):
        results = []
        for handle, board in batch:
            history = self._fresh_history()
            for board_before, move_uci in handle:
                history.append_observed_position(board_before)
                history.record_played_move(move_uci)
            full_batch = history.build_batch_for_current_position(board)
            with torch.no_grad():
                out = self._model(full_batch, return_loss=False)
            logits = out["logits"][-1]
            value_stm = self._module._value_scalar_from_logits(
                out["value_logits"][-1]
            )
            try:
                legal_logits, legal_moves, _, _ = self._module._project_legal_logits(
                    logits=logits, board=board, move_vocab=self._move_vocab
                )
                log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
            except RuntimeError:
                legal_moves, log_priors = [], []
            results.append(PositionEval(value_stm, legal_moves, log_priors))
        return results


def _static_vocab():
    from imba_chess.data.move_vocab import all_possible_uci_moves

    return MoveVocab.build(
        all_possible_uci_moves(), config=MoveVocabConfig(include_unk=False)
    )


def test_cached_evaluator_matches_full_forward_reference():
    module = _load_eval_script_module()
    move_vocab = _static_vocab()
    model = _tiny_model(vocab_size=len(move_vocab))
    encoder = BoardStateEncoder()

    board = chess.Board()
    played = []
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=encoder
    )
    for move_uci in ["e2e4", "e7e5", "g1f3"]:
        played.append((board.copy(stack=False), move_uci))
        history.append_observed_position(board)
        history.record_played_move(move_uci)
        board.push_uci(move_uci)

    root_batch = history.build_batch_for_current_position(board)
    with torch.no_grad():
        prefill = model(root_batch, return_loss=False, return_kv=True)

    cached = module.CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=prefill["kv_caches"],
        prefix_len=int(root_batch["total_tokens"]),
    )
    reference = _FullForwardReferenceEvaluator(
        module=module,
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        played=played,
    )

    # Depth 1: two candidate moves. Depth 2: one reply under each.
    candidates = [chess.Move.from_uci("b8c6"), chess.Move.from_uci("d7d6")]
    cached_handles = [cached.extend(None, board, move) for move in candidates]
    ref_handles = [reference.extend(None, board, move) for move in candidates]
    boards1 = []
    for move in candidates:
        board1 = board.copy()
        board1.push(move)
        boards1.append(board1)

    cached_evals = cached.evaluate(list(zip(cached_handles, boards1)))
    ref_evals = reference.evaluate(list(zip(ref_handles, boards1)))
    for got, want in zip(cached_evals, ref_evals):
        assert abs(got.value_stm - want.value_stm) < 1e-5
        assert [m.uci() for m in got.legal_moves] == [
            m.uci() for m in want.legal_moves
        ]
        for a, b in zip(got.legal_log_priors, want.legal_log_priors):
            assert abs(a - b) < 1e-5

    # One depth-2 node under each candidate, evaluated in a single wave.
    replies = [list(b1.legal_moves)[0] for b1 in boards1]
    cached2 = [
        cached.extend(handle, b1, reply)
        for handle, b1, reply in zip(cached_handles, boards1, replies)
    ]
    ref2 = [
        reference.extend(handle, b1, reply)
        for handle, b1, reply in zip(ref_handles, boards1, replies)
    ]
    boards2 = []
    for b1, reply in zip(boards1, replies):
        b2 = b1.copy()
        b2.push(reply)
        boards2.append(b2)
    cached_evals2 = cached.evaluate(list(zip(cached2, boards2)))
    ref_evals2 = reference.evaluate(list(zip(ref2, boards2)))
    for got, want in zip(cached_evals2, ref_evals2):
        assert abs(got.value_stm - want.value_stm) < 1e-5
        for a, b in zip(got.legal_log_priors, want.legal_log_priors):
            assert abs(a - b) < 1e-5


def test_strategy_picks_identical_move_cached_vs_reference():
    module = _load_eval_script_module()
    move_vocab = _static_vocab()
    model = _tiny_model(vocab_size=len(move_vocab))
    encoder = BoardStateEncoder()

    board = chess.Board()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=encoder
    )
    root_batch = history.build_batch_for_current_position(board)
    with torch.no_grad():
        prefill = model(root_batch, return_loss=False, return_kv=True)
        root_logits = prefill["logits"][-1]
    legal_logits, legal_moves, _, _ = module._project_legal_logits(
        logits=root_logits, board=board, move_vocab=move_vocab
    )
    log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()

    cached = module.CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=prefill["kv_caches"],
        prefix_len=int(root_batch["total_tokens"]),
    )
    reference = _FullForwardReferenceEvaluator(
        module=module,
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        played=[],
    )

    chosen_cached, _ = select_value_search_d2(
        evaluator=cached,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=log_priors,
        top_k=4,
        lam=0.05,
    )
    chosen_ref, _ = select_value_search_d2(
        evaluator=reference,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=log_priors,
        top_k=4,
        lam=0.05,
    )
    assert legal_moves[chosen_cached].uci() == legal_moves[chosen_ref].uci()
```

- [ ] **Step 2: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_prefix_decode.py -v`
Expected: PASS (all 8). A failure here with Tasks 1–3 green individually means an integration seam bug — most likely position indexing (`prefix_len` vs suffix depth) or the `extend(None, ...)` root-handle path.

- [ ] **Step 3: Cleanup audit**

Run:
```bash
grep -n "_HistoryPositionEvaluator\|_forward_last_token_outputs\|_merge_single_sequence_batches\|_SEARCH_EVAL_MAX_TOKENS_PER_CHUNK\|\.clone()" scripts/eval_vs_stockfish.py; echo "exit=$?"
```
Expected: `exit=1` (no matches — all replaced plumbing gone, no stray `.clone()` calls).

- [ ] **Step 4: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_prefix_decode.py
git commit -m "test: end-to-end cached-vs-full-forward equivalence gate"
```

---

## Post-implementation (manual, not part of the plan)

GPU timing check once the local card is free: same checkpoint, halving @ 256
vs SF1400, ~5 games via `eval_best_checkpoint.sh` (delete the old halving
JSON first). Acceptance: ≥5× faster per move; expectation 10–30×. Then run
the SF1800 evals on the cached path and update README results + the
prefix-caching limitation bullet.
