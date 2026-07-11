# Search-Backed Value Distillation (Expert Iteration, Phase 1a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the trunk value head's constant-per-game training target with a position-resolved one, blended from the game's real outcome and `value_search_halving`'s own lookahead, gated by a single hyperparameter `β`.

**Architecture:** An offline script (`scripts/generate_search_rollouts.py`) replays the training split of the Lichess dataset and calls the existing `select_value_search_halving` search at sampled positions, writing one parquet row per sampled position. At training time, `EventBuilder` optionally looks up a rollout row for each token by `(game_id, ply)` and blends it into a per-token soft value target; `HSTUChessModel`'s value-loss branch uses soft cross-entropy for tokens that have one and the existing hard cross-entropy for every other token. Everything is off by default (`[expert_iteration].rollout_path` unset), reproducing today's training byte-for-bit.

**Tech Stack:** Python, PyTorch, python-chess, pyarrow (parquet), pytest. No new training framework — reuses `scripts/train.py`'s existing Ignite loop.

## Global Constraints

- Spec of record: `docs/superpowers/specs/2026-07-07-expert-iteration-distillation-design.md`. Every task below implements one part of it; do not add Phase 1b (policy target) or Phase 2 (self-play/GRPO) behavior — those are explicitly out of scope for this plan.
- **Unset `[expert_iteration].rollout_path` (the default) must reproduce today's training byte-for-byte.** Every task that touches the training path must preserve this.
- Rollout sampling only ever reads the **train** split of `LichessDataset`. Never wire rollouts into the val/test loaders in `scripts/train.py`.
- `root_wdl_unsearched` in a rollout row is a frozen snapshot from the checkpoint that generated the rollout — never recomputed live during training.
- Follow existing repo conventions throughout: `from __future__ import annotations`, frozen dataclasses for config sections, `pytest.importorskip("torch")` at the top of any test file that needs torch, tests colocated in `tests/` mirroring `src/imba_chess/...` module paths.

---

## Task 1: Extract shared position-evaluator infra out of `scripts/eval_vs_stockfish.py`

`scripts/generate_search_rollouts.py` (Task 5) needs the exact same "build a `PositionEvaluator` from a loaded checkpoint and a live game replay" machinery `scripts/eval_vs_stockfish.py` already has: `_SequenceHistory`, `CachedPositionEvaluator`, `_CachedNode`, `_forward_model`, `_project_legal_logits`, `_value_scalar_from_logits`, `_is_power_of_two`, and `_load_model` (~250 lines). Duplicating this into a second script would drift the two copies apart. Extract it once into a shared module; `eval_vs_stockfish.py` re-imports the same names so its behavior and its existing tests are unaffected.

**Files:**
- Create: `src/imba_chess/eval/position_evaluator.py`
- Modify: `scripts/eval_vs_stockfish.py` (delete the moved definitions, add an import, rename one call site)
- No test file created in this task — this is a pure refactor; the safety net is the existing `tests/test_eval_vs_stockfish.py` suite staying green.

**Interfaces:**
- Produces (for Task 5 to consume): `imba_chess.eval.position_evaluator.load_hstu_checkpoint(*, checkpoint_path: Path, repo_config, move_vocab: MoveVocab, device: torch.device, compile_model: bool, require_value_head: bool = False) -> tuple[torch.nn.Module, bool]`, `imba_chess.eval.position_evaluator._SequenceHistory`, `imba_chess.eval.position_evaluator.CachedPositionEvaluator`, `imba_chess.eval.position_evaluator._forward_model(*, model, batch, device, dtype, return_kv=False) -> dict[str, torch.Tensor]`, `imba_chess.eval.position_evaluator._project_legal_logits(*, logits, board, move_vocab) -> tuple[torch.Tensor, list[chess.Move], int, int]`.

- [ ] **Step 1: Confirm the current baseline is green**

Run: `pytest tests/test_eval_vs_stockfish.py -v`
Expected: all tests PASS (this is the safety net for the refactor below).

- [ ] **Step 2: Create `src/imba_chess/eval/position_evaluator.py` with the moved code**

```python
from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

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
```

- [ ] **Step 3: Remove the moved code from `scripts/eval_vs_stockfish.py` and import it instead**

Delete from `scripts/eval_vs_stockfish.py`:
- The `_SequenceHistory` class (original lines 91-178).
- The `_is_power_of_two` function (original lines 321-323).
- The `_load_model` function (original lines 325-389).
- The `_forward_model` function (original lines 435-461).
- The `_value_scalar_from_logits` function (original lines 463-466).
- The `_project_legal_logits` function (original lines 468-494).
- The `_CachedNode` class (original lines 496-513).
- The `CachedPositionEvaluator` class (original lines 515-691).

Add to the top-level imports (near the existing `from imba_chess...` imports):

```python
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _CachedNode,
    _SequenceHistory,
    _forward_model,
    _is_power_of_two,
    _project_legal_logits,
    _value_scalar_from_logits,
    load_hstu_checkpoint,
)
```

Rename the single call site that used `_load_model(...)` (originally around line 1542) to `load_hstu_checkpoint(...)` — same keyword arguments, no other changes.

- [ ] **Step 4: Re-run the baseline suite to confirm the refactor is behavior-preserving**

Run: `pytest tests/test_eval_vs_stockfish.py -v`
Expected: all tests PASS, identical to Step 1 (the tests access `module._SequenceHistory` / `module.CachedPositionEvaluator` on the dynamically-loaded script module — these still resolve because the import in Step 3 binds those exact names at module level).

- [ ] **Step 5: Add the new module to the eval package's public exports**

Edit `src/imba_chess/eval/__init__.py`, add:

```python
from .position_evaluator import CachedPositionEvaluator, load_hstu_checkpoint
```

and add `"CachedPositionEvaluator"` and `"load_hstu_checkpoint"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/eval/position_evaluator.py src/imba_chess/eval/__init__.py scripts/eval_vs_stockfish.py
git commit -m "refactor: extract shared position-evaluator infra for reuse by rollout generation"
```

---

## Task 2: Add the `[expert_iteration]` config section

**Files:**
- Modify: `src/imba_chess/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `imba_chess.config.ExpertIterationConfig` (fields `rollout_path: Optional[str] = None`, `beta: float = 0.0`), accessible as `RepoConfig().expert_iteration`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_expert_iteration_config_defaults():
    config = RepoConfig()
    assert config.expert_iteration.rollout_path is None
    assert config.expert_iteration.beta == 0.0


def test_load_repo_config_reads_expert_iteration_section(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[expert_iteration]
rollout_path = "artifacts/rollouts/ckpt23.parquet"
beta = 0.3
"""
    )
    config = load_repo_config(config_path)
    assert config.expert_iteration.rollout_path == "artifacts/rollouts/ckpt23.parquet"
    assert config.expert_iteration.beta == 0.3
```

(Check the top of `tests/test_config.py` for its existing `RepoConfig`/`load_repo_config` import line and add to it rather than re-importing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -k expert_iteration -v`
Expected: FAIL with `AttributeError: 'RepoConfig' object has no attribute 'expert_iteration'`

- [ ] **Step 3: Implement the config section**

In `src/imba_chess/config.py`, add after `ValueNetSection`:

```python
@dataclass(frozen=True)
class ExpertIterationConfig:
    # Path to a rollout parquet written by scripts/generate_search_rollouts.py.
    # Unset (default) => training is byte-identical to today.
    rollout_path: Optional[str] = None
    # blend(real_outcome, searched_value; beta). 0 = today's exact target,
    # 1 = pure searched estimate.
    beta: float = 0.0
```

Add `expert_iteration: ExpertIterationConfig = field(default_factory=ExpertIterationConfig)` to `RepoConfig`, and in `load_repo_config`:

```python
        expert_iteration=_read_section(
            ExpertIterationConfig, payload.get("expert_iteration", {}), "expert_iteration"
        ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: all PASS, including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/config.py tests/test_config.py
git commit -m "feat: add [expert_iteration] config section (rollout_path, beta)"
```

---

## Task 3: Value-target blend helper (spec Part 3)

Pure function, no torch dependency — the math that turns `(root_wdl_unsearched, backed_value, real_outcome_stm, beta)` into a blended 3-vector.

**Note on the spec's formula:** the raw formula (`p_win = (1 - p_draw0 + backed_value) / 2`, `p_loss = (1 - p_draw0 - backed_value) / 2`) can produce a negative `p_win` or `p_loss` when `backed_value` (the search's own, differently-sourced estimate) is extreme and `p_draw0` is large — e.g. `root_wdl_unsearched=(0.05, 0.9, 0.05)` with `backed_value=-0.8` gives `p_win_raw = -0.175`. This wasn't caught in the design doc. The implementation below clamps both to `>= 0` and renormalizes the 3-vector to sum to 1, so it's always a valid distribution; in that edge case the draw mass no longer exactly equals `p_draw0` (documented and tested as its own case, distinct from the common case where it does).

**Files:**
- Create: `src/imba_chess/data/value_target_blend.py`
- Test: `tests/test_value_target_blend.py`

**Interfaces:**
- Produces: `imba_chess.data.value_target_blend.compute_blended_value_target(*, root_wdl_unsearched: tuple[float, float, float], backed_value: float, real_outcome_stm: int, beta: float) -> list[float]` — returns `[p_loss, p_draw, p_win]`, summing to 1.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_value_target_blend.py`:

```python
import pytest

from imba_chess.data.value_target_blend import compute_blended_value_target


def test_beta_zero_reproduces_real_outcome_one_hot():
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.9,
        real_outcome_stm=1,
        beta=0.0,
    )
    assert result == pytest.approx([0.0, 0.0, 1.0])


def test_beta_one_reproduces_searched_vec_when_not_clamped():
    # p_win_raw = (1 - 0.3 + 0.2) / 2 = 0.45, p_loss_raw = (1 - 0.3 - 0.2) / 2 = 0.25
    # both non-negative -> no clamping, sums to 1 exactly.
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.2,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert result == pytest.approx([0.25, 0.3, 0.45])


def test_draw_mass_preserved_when_not_clamped():
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.2,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert result[1] == pytest.approx(0.3)


def test_sums_to_one_for_intermediate_beta():
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.2,
        real_outcome_stm=-1,
        beta=0.4,
    )
    assert sum(result) == pytest.approx(1.0)


def test_monotone_in_backed_value():
    low = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=-0.1,
        real_outcome_stm=0,
        beta=1.0,
    )
    high = compute_blended_value_target(
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        backed_value=0.4,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert high[2] > low[2]  # p_win increases
    assert high[0] < low[0]  # p_loss decreases


def test_clamps_and_renormalizes_when_backed_value_is_extreme():
    # p_win_raw = (1 - 0.9 - 0.8) / 2 = -0.35 -> clamped to 0.
    result = compute_blended_value_target(
        root_wdl_unsearched=(0.05, 0.9, 0.05),
        backed_value=-0.8,
        real_outcome_stm=0,
        beta=1.0,
    )
    assert result[2] == pytest.approx(0.0)
    assert all(value >= 0.0 for value in result)
    assert sum(result) == pytest.approx(1.0)
    # Draw mass is NOT preserved in the clamped case (documents the deviation
    # from the common-case invariant tested above).
    assert result[1] != pytest.approx(0.9)


def test_beta_out_of_range_raises():
    with pytest.raises(ValueError, match="beta"):
        compute_blended_value_target(
            root_wdl_unsearched=(0.2, 0.3, 0.5),
            backed_value=0.0,
            real_outcome_stm=0,
            beta=1.5,
        )


def test_invalid_real_outcome_raises():
    with pytest.raises(ValueError, match="real_outcome_stm"):
        compute_blended_value_target(
            root_wdl_unsearched=(0.2, 0.3, 0.5),
            backed_value=0.0,
            real_outcome_stm=2,
            beta=0.5,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_value_target_blend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.data.value_target_blend'`

- [ ] **Step 3: Implement `src/imba_chess/data/value_target_blend.py`**

```python
from __future__ import annotations


def compute_blended_value_target(
    *,
    root_wdl_unsearched: tuple[float, float, float],
    backed_value: float,
    real_outcome_stm: int,
    beta: float,
) -> list[float]:
    """blend(real_outcome, searched_value; beta) from spec Part 3.

    root_wdl_unsearched is (p_loss0, p_draw0, p_win0) from the trunk's own
    un-searched value head at the root position (a frozen snapshot recorded
    at rollout-generation time). backed_value is the searched, side-to-move
    POV scalar in [-1, 1] for the best arm. real_outcome_stm is the game's
    actual outcome from this token's side-to-move POV, in {-1, 0, 1}.

    Returns [p_loss, p_draw, p_win], summing to 1. beta=0 reproduces the
    one-hot real-outcome vector exactly; beta=1 is the pure searched
    estimate (draw mass equal to p_draw0, except in the rare case where an
    extreme backed_value combined with a high p_draw0 would otherwise drive
    p_win or p_loss negative -- there both are clamped to 0 and the vector
    is renormalized, which changes the draw share).
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if real_outcome_stm not in (-1, 0, 1):
        raise ValueError(f"real_outcome_stm must be in {{-1, 0, 1}}, got {real_outcome_stm}")

    _, p_draw0, _ = root_wdl_unsearched
    p_win_raw = (1.0 - p_draw0 + backed_value) / 2.0
    p_loss_raw = (1.0 - p_draw0 - backed_value) / 2.0
    p_win = max(0.0, p_win_raw)
    p_loss = max(0.0, p_loss_raw)
    p_draw = p_draw0
    total = p_win + p_loss + p_draw
    if total <= 0.0:
        raise ValueError(
            "root_wdl_unsearched and backed_value produced a degenerate "
            "(all-zero) searched value vector"
        )
    searched_vec = [p_loss / total, p_draw / total, p_win / total]

    real_outcome_vec = [0.0, 0.0, 0.0]
    real_outcome_vec[real_outcome_stm + 1] = 1.0

    return [
        (1.0 - beta) * real_value + beta * searched_value
        for real_value, searched_value in zip(real_outcome_vec, searched_vec)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_value_target_blend.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/data/value_target_blend.py tests/test_value_target_blend.py
git commit -m "feat: add search-backed value target blend helper (ExIt Phase 1a Part 3)"
```

---

## Task 4: Rollout data structure (spec Part 2)

`RolloutRow` schema plus a parquet writer/loader, keyed by `(game_id, ply)` for training-time lookup. Flattens the spec's `search_config: struct` into separate scalar columns (`search_budget`, `search_top_m`, `search_max_depth`) — consistent with the spec's own stated preference for flat columns over nested structs everywhere else in the same schema.

**Files:**
- Create: `src/imba_chess/data/rollout_store.py`
- Test: `tests/test_rollout_store.py`

**Interfaces:**
- Produces: `imba_chess.data.rollout_store.RolloutRow` (frozen dataclass, fields listed below), `write_rollout_parquet(rows: list[RolloutRow], path: str | Path) -> None`, `load_rollout_lookup(path: str | Path) -> dict[tuple[str, int], RolloutRow]`.
- `RolloutRow` fields: `game_id: str`, `ply: int`, `human_move_uci: str`, `human_move_backed_value: float | None`, `real_outcome_stm: int`, `best_arm_move_uci: str`, `best_arm_backed_value: float`, `root_wdl_unsearched: tuple[float, float, float]`, `arm_move_uci: tuple[str, ...]`, `arm_backed_value: tuple[float, ...]`, `arm_evals_spent: tuple[int, ...]`, `arm_log_prior: tuple[float, ...]`, `search_budget: int`, `search_top_m: int`, `search_max_depth: int`, `checkpoint: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rollout_store.py`:

```python
import pytest

pytest.importorskip("pyarrow")

from imba_chess.data.rollout_store import (
    RolloutRow,
    load_rollout_lookup,
    write_rollout_parquet,
)


def _row(game_id: str, ply: int) -> RolloutRow:
    return RolloutRow(
        game_id=game_id,
        ply=ply,
        human_move_uci="e2e4",
        human_move_backed_value=0.1,
        real_outcome_stm=1,
        best_arm_move_uci="d2d4",
        best_arm_backed_value=0.3,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=("d2d4", "e2e4", "", ""),
        arm_backed_value=(0.3, 0.1, 0.0, 0.0),
        arm_evals_spent=(120, 80, 0, 0),
        arm_log_prior=(-0.5, -0.7, 0.0, 0.0),
        search_budget=2048,
        search_top_m=4,
        search_max_depth=8,
        checkpoint="artifacts/checkpoints/best_hr10_checkpoint_23.pt",
    )


def test_write_then_load_round_trips(tmp_path):
    path = tmp_path / "rollouts.parquet"
    rows = [_row("g1", 3), _row("g1", 7), _row("g2", 0)]

    write_rollout_parquet(rows, path)
    lookup = load_rollout_lookup(path)

    assert set(lookup.keys()) == {("g1", 3), ("g1", 7), ("g2", 0)}
    restored = lookup[("g1", 3)]
    assert restored == rows[0]


def test_load_handles_null_human_move_backed_value(tmp_path):
    path = tmp_path / "rollouts.parquet"
    row = _row("g1", 0)
    row_with_null = RolloutRow(**{**row.__dict__, "human_move_backed_value": None})

    write_rollout_parquet([row_with_null], path)
    lookup = load_rollout_lookup(path)

    assert lookup[("g1", 0)].human_move_backed_value is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rollout_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.data.rollout_store'`

- [ ] **Step 3: Implement `src/imba_chess/data/rollout_store.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class RolloutRow:
    game_id: str
    ply: int
    human_move_uci: str
    human_move_backed_value: float | None
    real_outcome_stm: int
    best_arm_move_uci: str
    best_arm_backed_value: float
    root_wdl_unsearched: tuple[float, float, float]
    arm_move_uci: tuple[str, ...]
    arm_backed_value: tuple[float, ...]
    arm_evals_spent: tuple[int, ...]
    arm_log_prior: tuple[float, ...]
    search_budget: int
    search_top_m: int
    search_max_depth: int
    checkpoint: str


_ROLLOUT_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string()),
        pa.field("ply", pa.int64()),
        pa.field("human_move_uci", pa.string()),
        pa.field("human_move_backed_value", pa.float64()),
        pa.field("real_outcome_stm", pa.int64()),
        pa.field("best_arm_move_uci", pa.string()),
        pa.field("best_arm_backed_value", pa.float64()),
        pa.field("root_wdl_unsearched", pa.list_(pa.float64())),
        pa.field("arm_move_uci", pa.list_(pa.string())),
        pa.field("arm_backed_value", pa.list_(pa.float64())),
        pa.field("arm_evals_spent", pa.list_(pa.int64())),
        pa.field("arm_log_prior", pa.list_(pa.float64())),
        pa.field("search_budget", pa.int64()),
        pa.field("search_top_m", pa.int64()),
        pa.field("search_max_depth", pa.int64()),
        pa.field("checkpoint", pa.string()),
    ]
)


def _row_to_record(row: RolloutRow) -> dict:
    return {
        "game_id": row.game_id,
        "ply": row.ply,
        "human_move_uci": row.human_move_uci,
        "human_move_backed_value": row.human_move_backed_value,
        "real_outcome_stm": row.real_outcome_stm,
        "best_arm_move_uci": row.best_arm_move_uci,
        "best_arm_backed_value": row.best_arm_backed_value,
        "root_wdl_unsearched": list(row.root_wdl_unsearched),
        "arm_move_uci": list(row.arm_move_uci),
        "arm_backed_value": list(row.arm_backed_value),
        "arm_evals_spent": list(row.arm_evals_spent),
        "arm_log_prior": list(row.arm_log_prior),
        "search_budget": row.search_budget,
        "search_top_m": row.search_top_m,
        "search_max_depth": row.search_max_depth,
        "checkpoint": row.checkpoint,
    }


def write_rollout_parquet(rows: list[RolloutRow], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_row_to_record(row) for row in rows], schema=_ROLLOUT_SCHEMA)
    pq.write_table(table, output_path)


def load_rollout_lookup(path: str | Path) -> dict[tuple[str, int], RolloutRow]:
    table = pq.read_table(path)
    lookup: dict[tuple[str, int], RolloutRow] = {}
    for record in table.to_pylist():
        row = RolloutRow(
            game_id=record["game_id"],
            ply=int(record["ply"]),
            human_move_uci=record["human_move_uci"],
            human_move_backed_value=(
                float(record["human_move_backed_value"])
                if record["human_move_backed_value"] is not None
                else None
            ),
            real_outcome_stm=int(record["real_outcome_stm"]),
            best_arm_move_uci=record["best_arm_move_uci"],
            best_arm_backed_value=float(record["best_arm_backed_value"]),
            root_wdl_unsearched=tuple(float(v) for v in record["root_wdl_unsearched"]),
            arm_move_uci=tuple(record["arm_move_uci"]),
            arm_backed_value=tuple(float(v) for v in record["arm_backed_value"]),
            arm_evals_spent=tuple(int(v) for v in record["arm_evals_spent"]),
            arm_log_prior=tuple(float(v) for v in record["arm_log_prior"]),
            search_budget=int(record["search_budget"]),
            search_top_m=int(record["search_top_m"]),
            search_max_depth=int(record["search_max_depth"]),
            checkpoint=record["checkpoint"],
        )
        lookup[(row.game_id, row.ply)] = row
    return lookup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rollout_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Add to the data package's public exports**

Edit `src/imba_chess/data/__init__.py`, add:

```python
from .rollout_store import RolloutRow, load_rollout_lookup, write_rollout_parquet
from .value_target_blend import compute_blended_value_target
```

and add `"RolloutRow"`, `"load_rollout_lookup"`, `"write_rollout_parquet"`, `"compute_blended_value_target"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/data/rollout_store.py tests/test_rollout_store.py src/imba_chess/data/__init__.py
git commit -m "feat: add rollout parquet schema, writer, and loader (ExIt Phase 1a Part 2)"
```

---

## Task 5: Rollout generation script (spec Part 1)

**Files:**
- Create: `scripts/generate_search_rollouts.py`
- Test: `tests/test_generate_search_rollouts.py`

**Interfaces:**
- Consumes: `imba_chess.eval.position_evaluator.{load_hstu_checkpoint, _SequenceHistory, CachedPositionEvaluator, _forward_model, _project_legal_logits}` (Task 1), `imba_chess.eval.search.{HalvingConfig, select_value_search_halving}`, `imba_chess.data.rollout_store.{RolloutRow, write_rollout_parquet}` (Task 4), `imba_chess.data.LichessDataset`, `imba_chess.data.move_vocab.{MoveVocab, load_or_create_static_move_vocab}`, `imba_chess.data.board_state.BoardStateEncoder`.
- Produces: a parquet file at `--output-path`, plus two pure, independently-testable helper functions: `_sample_ply_indices(num_plies: int, *, every_n: int, seed: int, game_id: str) -> list[int]` and `_pad_or_truncate_arms(rows: list[dict], *, top_m: int) -> list[dict]`.

This script processes one sampled root position at a time (search itself already batches evaluator calls internally per wave); cross-position batching is an explicit non-goal here since this is offline, throughput-insensitive generation, matching how `eval_vs_stockfish.py` also processes one game at a time.

- [ ] **Step 1: Write the failing tests for the two pure helpers**

Create `tests/test_generate_search_rollouts.py`:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_search_rollouts.py"
    spec = importlib.util.spec_from_file_location("generate_search_rollouts_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load generate_search_rollouts.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_ply_indices_is_deterministic_and_bounded():
    module = _load_script_module()

    first = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g1")
    second = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g1")
    assert first == second
    assert all(0 <= idx < 40 for idx in first)
    assert len(first) >= 1


def test_sample_ply_indices_differs_by_game_id():
    module = _load_script_module()

    a = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g1")
    b = module._sample_ply_indices(40, every_n=8, seed=42, game_id="g2")
    assert a != b or len(a) <= 1  # near-certain to differ with 40 plies / every_n=8


def test_sample_ply_indices_empty_game():
    module = _load_script_module()
    assert module._sample_ply_indices(0, every_n=8, seed=42, game_id="g1") == []


def test_pad_or_truncate_arms_pads_short_lists():
    module = _load_script_module()
    rows = [
        {"move_uci": "e2e4", "backed_value": 0.3, "evals_spent": 100, "policy_log_prob": -0.2},
    ]
    padded = module._pad_or_truncate_arms(rows, top_m=3)
    assert len(padded) == 3
    assert padded[0]["move_uci"] == "e2e4"
    assert padded[1]["move_uci"] == ""
    assert padded[1]["backed_value"] == 0.0
    assert padded[1]["evals_spent"] == 0


def test_pad_or_truncate_arms_truncates_long_lists():
    module = _load_script_module()
    rows = [
        {"move_uci": f"m{i}", "backed_value": float(i), "evals_spent": i, "policy_log_prob": -float(i)}
        for i in range(5)
    ]
    truncated = module._pad_or_truncate_arms(rows, top_m=3)
    assert len(truncated) == 3
    assert [r["move_uci"] for r in truncated] == ["m0", "m1", "m2"]


def test_pad_or_truncate_arms_maps_none_backed_value_to_zero():
    module = _load_script_module()
    rows = [
        {"move_uci": "e2e4", "backed_value": None, "evals_spent": 0, "policy_log_prob": -0.1},
    ]
    padded = module._pad_or_truncate_arms(rows, top_m=1)
    assert padded[0]["backed_value"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_search_rollouts.py -v`
Expected: FAIL with `FileNotFoundError` / `RuntimeError` from `_load_script_module` (the script doesn't exist yet).

- [ ] **Step 3: Implement `scripts/generate_search_rollouts.py`**

```python
#!/usr/bin/env python3
"""Generate search-backed value-target rollouts from the training split.

Replays games from LichessDataset(split="train") and, at a sampled subset of
plies per game, calls the same value_search_halving search used at inference
to record a position-resolved value estimate. Writes one row per sampled
position to a rollout parquet consumed by scripts/train.py via
[expert_iteration].rollout_path. Generates data only; trains nothing (mirrors
scripts/train_value_net.py's role as a small, focused, non-Ignite script).

Usage: python scripts/generate_search_rollouts.py --checkpoint <path> --output-path <path>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import torch
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import load_or_create_static_move_vocab
from imba_chess.data.rollout_store import RolloutRow, write_rollout_parquet
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _SequenceHistory,
    _forward_model,
    _project_legal_logits,
    load_hstu_checkpoint,
)
from imba_chess.eval.search import HalvingConfig, select_value_search_halving

_RESULT_TO_WHITE_OUTCOME = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _result_to_white_outcome(result: str) -> int:
    outcome = _RESULT_TO_WHITE_OUTCOME.get(result)
    if outcome is None:
        raise ValueError(f"Unsupported game result: {result!r}")
    return outcome


def _sample_ply_indices(num_plies: int, *, every_n: int, seed: int, game_id: str) -> list[int]:
    if num_plies <= 0:
        return []
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    rng = random.Random(f"{seed}:{game_id}")
    offset = rng.randrange(0, every_n)
    return list(range(offset, num_plies, every_n))


def _pad_or_truncate_arms(rows: list[dict], *, top_m: int) -> list[dict]:
    selected = list(rows[:top_m])
    while len(selected) < top_m:
        selected.append(
            {"move_uci": "", "backed_value": 0.0, "evals_spent": 0, "policy_log_prob": 0.0}
        )
    return [
        {
            "move_uci": row["move_uci"],
            "backed_value": (
                float(row["backed_value"]) if row["backed_value"] is not None else 0.0
            ),
            "evals_spent": int(row["evals_spent"]),
            "policy_log_prob": float(row["policy_log_prob"]),
        }
        for row in selected
    ]


def _generate_rollout_row(
    *,
    model,
    board: chess.Board,
    history: _SequenceHistory,
    move_vocab,
    board_state_encoder: BoardStateEncoder,
    device: torch.device,
    dtype: torch.dtype,
    halving_config: HalvingConfig,
    game_id: str,
    ply: int,
    human_move_uci: str,
    real_outcome_stm: int,
    checkpoint_path: str,
) -> RolloutRow | None:
    batch = history.build_batch_for_current_position(board)
    output = _forward_model(model=model, batch=batch, device=device, dtype=dtype, return_kv=True)

    logits = output["logits"][-1]
    try:
        legal_logits, legal_moves, _, mapped_legal = _project_legal_logits(
            logits=logits, board=board, move_vocab=move_vocab
        )
    except RuntimeError:
        return None
    if mapped_legal == 0:
        return None
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    root_wdl_unsearched = tuple(
        float(v) for v in torch.softmax(output["value_logits"][-1].float(), dim=-1).tolist()
    )

    evaluator = CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        device=device,
        dtype=dtype,
        prefix_kv=output["kv_caches"],
        prefix_len=int(batch["total_tokens"]),
    )
    best_local_idx, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=halving_config,
    )
    best_move_uci = legal_moves[best_local_idx].uci()
    best_row = next((row for row in rows if row["move_uci"] == best_move_uci), None)
    if best_row is None or best_row["backed_value"] is None:
        return None  # budget-starved fallback with no scored arm; skip this position

    human_row = next((row for row in rows if row["move_uci"] == human_move_uci), None)
    human_move_backed_value = (
        float(human_row["backed_value"])
        if human_row is not None and human_row["backed_value"] is not None
        else None
    )

    arms = _pad_or_truncate_arms(rows, top_m=halving_config.top_m)
    return RolloutRow(
        game_id=game_id,
        ply=ply,
        human_move_uci=human_move_uci,
        human_move_backed_value=human_move_backed_value,
        real_outcome_stm=real_outcome_stm,
        best_arm_move_uci=best_move_uci,
        best_arm_backed_value=float(best_row["backed_value"]),
        root_wdl_unsearched=root_wdl_unsearched,
        arm_move_uci=tuple(arm["move_uci"] for arm in arms),
        arm_backed_value=tuple(arm["backed_value"] for arm in arms),
        arm_evals_spent=tuple(arm["evals_spent"] for arm in arms),
        arm_log_prior=tuple(arm["policy_log_prob"] for arm in arms),
        search_budget=halving_config.budget,
        search_top_m=halving_config.top_m,
        search_max_depth=halving_config.max_depth,
        checkpoint=checkpoint_path,
    )


def _process_game(
    game: dict,
    *,
    model,
    move_vocab,
    board_state_encoder: BoardStateEncoder,
    device: torch.device,
    dtype: torch.dtype,
    halving_config: HalvingConfig,
    every_n_plies: int,
    sample_seed: int,
    checkpoint_path: str,
) -> list[RolloutRow]:
    plays = game["plays"]
    sampled = set(
        _sample_ply_indices(
            len(plays), every_n=every_n_plies, seed=sample_seed, game_id=game["game_id"]
        )
    )
    if not sampled:
        return []

    game_result_white = _result_to_white_outcome(game["result"])
    board = chess.Board()
    history = _SequenceHistory(move_vocab=move_vocab, board_state_encoder=board_state_encoder)
    rows: list[RolloutRow] = []

    for ply_idx, play in enumerate(plays):
        if ply_idx in sampled:
            real_outcome_stm = game_result_white if board.turn == chess.WHITE else -game_result_white
            row = _generate_rollout_row(
                model=model,
                board=board,
                history=history,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                device=device,
                dtype=dtype,
                halving_config=halving_config,
                game_id=game["game_id"],
                ply=ply_idx,
                human_move_uci=play["move_uci"],
                real_outcome_stm=real_outcome_stm,
                checkpoint_path=checkpoint_path,
            )
            if row is not None:
                rows.append(row)
        move = chess.Move.from_uci(play["move_uci"])
        history.append_observed_position(board)
        history.record_played_move(play["move_uci"])
        board.push(move)

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--sample-every-n-plies", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    # Search knobs default to [eval_vs_stockfish] so a rollout run matches
    # whatever the live eval protocol currently uses; override per-run if needed.
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--search-top-m", type=int, default=None)
    parser.add_argument("--search-max-depth", type=int, default=None)
    parser.add_argument("--halving-rounds", type=int, default=None)
    parser.add_argument("--search-refutation-top-r", type=int, default=None)
    parser.add_argument("--search-expand-top", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_config = load_repo_config(args.config)
    eval_cfg = repo_config.eval_vs_stockfish

    device_arg = args.device or eval_cfg.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype or eval_cfg.dtype]

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path, include_unk=repo_config.vocab.include_unk
    )
    model, _ = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=False,
        require_value_head=True,
    )
    board_state_encoder = BoardStateEncoder(repo_config.board_state)

    halving_config = HalvingConfig(
        budget=int(args.search_budget if args.search_budget is not None else eval_cfg.search_budget),
        top_m=int(args.search_top_m if args.search_top_m is not None else eval_cfg.search_top_m),
        rounds=int(args.halving_rounds if args.halving_rounds is not None else eval_cfg.halving_rounds),
        refutation_top_r=int(
            args.search_refutation_top_r
            if args.search_refutation_top_r is not None
            else eval_cfg.search_refutation_top_r
        ),
        expand_top=int(
            args.search_expand_top if args.search_expand_top is not None else eval_cfg.search_expand_top
        ),
        max_depth=int(
            args.search_max_depth if args.search_max_depth is not None else eval_cfg.search_max_depth
        ),
    )

    dataset_cfg = repo_config.dataset
    lichess_dataset = LichessDataset(
        min_avg_elo=dataset_cfg.min_avg_elo,
        min_time_control_sec=dataset_cfg.min_time_control_sec,
        split="train",
        dataset_name=dataset_cfg.dataset_name,
        train_start_month=dataset_cfg.train_start_month,
        train_end_month=dataset_cfg.train_end_month,
        cache_dir=dataset_cfg.cache_dir,
        parquet_batch_size=dataset_cfg.parquet_batch_size,
        max_seq_len=dataset_cfg.max_seq_len,
        board_state_config=repo_config.board_state,
    )

    all_rows: list[RolloutRow] = []
    games_processed = 0
    for game in tqdm(lichess_dataset.stream(), desc="rollout-generation", unit="game"):
        rows = _process_game(
            game,
            model=model,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            device=device,
            dtype=dtype,
            halving_config=halving_config,
            every_n_plies=args.sample_every_n_plies,
            sample_seed=args.sample_seed,
            checkpoint_path=str(args.checkpoint),
        )
        all_rows.extend(rows)
        games_processed += 1
        if args.max_games is not None and games_processed >= args.max_games:
            break

    write_rollout_parquet(all_rows, args.output_path)
    print(
        f"wrote {len(all_rows)} rollout rows from {games_processed} games to {args.output_path}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_search_rollouts.py -v`
Expected: all PASS.

- [ ] **Step 5: Add an end-to-end smoke test using a tiny real model**

Append to `tests/test_generate_search_rollouts.py`:

```python
def test_process_game_end_to_end_with_tiny_model(tmp_path):
    import torch as torch_module

    pytest.importorskip("torch")
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.move_vocab import MoveVocab
    from imba_chess.model import HSTUChessModel, build_hstu_chess_config
    from imba_chess.config import ModelConfig

    module = _load_script_module()

    move_vocab = MoveVocab.build_static()
    model_cfg = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=16,
            attention_dim=16,
            num_heads=2,
            num_layers=1,
            max_position_embeddings=64,
            enable_value_head=True,
        ),
        move_vocab_size=len(move_vocab),
    )
    model = HSTUChessModel(model_cfg)
    model.eval()

    game = {
        "game_id": "https://lichess.org/smoketest",
        "result": "1-0",
        "plays": [
            {"move_uci": "e2e4"},
            {"move_uci": "e7e5"},
            {"move_uci": "g1f3"},
            {"move_uci": "b8c6"},
        ],
    }

    rows = module._process_game(
        game,
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch_module.device("cpu"),
        dtype=torch_module.float32,
        halving_config=module.HalvingConfig(budget=32, top_m=4, max_depth=2),
        every_n_plies=1,
        sample_seed=42,
        checkpoint_path="dummy.pt",
    )

    assert len(rows) >= 1
    for row in rows:
        assert row.game_id == "https://lichess.org/smoketest"
        assert 0 <= row.ply < 4
        assert len(row.arm_move_uci) == 4
        assert len(row.root_wdl_unsearched) == 3
        assert abs(sum(row.root_wdl_unsearched) - 1.0) < 1e-4
```

Run: `pytest tests/test_generate_search_rollouts.py -v`
Expected: all PASS (the tiny untrained model produces valid, if arbitrary, values — this test checks shapes/ranges/plumbing, not accuracy).

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_search_rollouts.py tests/test_generate_search_rollouts.py
git commit -m "feat: add scripts/generate_search_rollouts.py (ExIt Phase 1a Part 1)"
```

---

## Task 6: `EventSequence`/`JaggedBatch` optional rollout fields + `EventBuilder` wiring (spec Part 4, dataset side)

**Files:**
- Modify: `src/imba_chess/data/types.py`
- Modify: `src/imba_chess/data/event_builder.py`
- Test: `tests/test_event_builder.py`

**Interfaces:**
- Consumes: `imba_chess.data.value_target_blend.compute_blended_value_target` (Task 3), `imba_chess.data.rollout_store.RolloutRow` (Task 4).
- Produces: `EventBuilder(move_vocab, *, rollout_lookup: dict[tuple[str, int], RolloutRow] | None = None, beta: float = 0.0)`. `build_game(...)` output gains optional keys `value_target_soft: list[list[float]]` and `has_rollout_value_target: list[int]` (present iff `rollout_lookup is not None`, one entry per token including the BOS token at index 0, which is always `[0.0, 0.0, 0.0]` / `0`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_event_builder.py` (reuse the existing `_row()` helper):

```python
from imba_chess.data.rollout_store import RolloutRow


def test_event_builder_without_rollout_lookup_omits_new_keys():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])

    builder = EventBuilder(vocab)
    sample = builder.build_game(game)

    assert "value_target_soft" not in sample
    assert "has_rollout_value_target" not in sample


def test_event_builder_with_rollout_lookup_blends_value_target():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])
    game_id = game["game_id"]

    # Rollout for ply 1 (the second play, token index 2) only.
    rollout_row = RolloutRow(
        game_id=game_id,
        ply=1,
        human_move_uci=game["plays"][1]["move_uci"],
        human_move_backed_value=0.2,
        real_outcome_stm=1,
        best_arm_move_uci=game["plays"][1]["move_uci"],
        best_arm_backed_value=0.6,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=(game["plays"][1]["move_uci"],),
        arm_backed_value=(0.6,),
        arm_evals_spent=(100,),
        arm_log_prior=(-0.1,),
        search_budget=256,
        search_top_m=1,
        search_max_depth=4,
        checkpoint="dummy.pt",
    )
    lookup = {(game_id, 1): rollout_row}

    builder = EventBuilder(vocab, rollout_lookup=lookup, beta=1.0)
    sample = builder.build_game(game)

    assert len(sample["value_target_soft"]) == len(sample["seq_token_id"])
    assert len(sample["has_rollout_value_target"]) == len(sample["seq_token_id"])
    # Only token 2 (== ply 1) has a rollout row; every other token (BOS at 0,
    # ply 0 at 1, ply 2 at 3, ply 3 at 4) must be untouched.
    for token_idx in range(len(sample["seq_token_id"])):
        if token_idx == 2:
            continue
        assert sample["has_rollout_value_target"][token_idx] == 0
        assert sample["value_target_soft"][token_idx] == [0.0, 0.0, 0.0]
    # Token 2 == ply 1 gets the blended target (beta=1.0 -> pure searched_vec).
    assert sample["has_rollout_value_target"][2] == 1
    assert abs(sum(sample["value_target_soft"][2]) - 1.0) < 1e-9
    assert sample["value_target_soft"][2][1] == pytest.approx(0.3)
```

(Add `import pytest` at the top of `tests/test_event_builder.py` if not already present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_event_builder.py -v`
Expected: FAIL — `EventBuilder(vocab, rollout_lookup=lookup, beta=1.0)` raises `TypeError: __init__() got an unexpected keyword argument 'rollout_lookup'`.

- [ ] **Step 3: Add the optional fields to `src/imba_chess/data/types.py`**

Replace the `EventSequence` class with:

```python
class _RolloutEventFields(TypedDict, total=False):
    value_target_soft: list[list[float]]
    has_rollout_value_target: list[int]


class EventSequence(_RolloutEventFields):
    game_id: str
    game_result_white: int
    seq_token_id: list[int]
    piece_ids: list[list[int]]
    turn_id: list[int]
    castle_id: list[int]
    ep_file_id: list[int]
    halfmove_bucket_id: list[int]
    fullmove_bucket_id: list[int]
    prev_move_id: list[int]
    target_move_id: list[int]
    played_by_elo: list[int]
```

Apply the same `total=False` mixin pattern to `JaggedBatch`:

```python
class _RolloutJaggedFields(TypedDict, total=False):
    value_target_soft: Any
    has_rollout_value_target: Any


class JaggedBatch(_RolloutJaggedFields):
    game_id: list[str]
    game_result_white: Any
    num_games: int
    total_tokens: int
    seq_lens: Any
    seq_offsets: Any
    piece_ids: Any
    seq_token_id: Any
    turn_id: Any
    castle_id: Any
    ep_file_id: Any
    halfmove_bucket_id: Any
    fullmove_bucket_id: Any
    prev_move_id: Any
    target_move_id: Any
    played_by_elo: Any
```

- [ ] **Step 4: Wire `rollout_lookup`/`beta` into `src/imba_chess/data/event_builder.py`**

Modify the `EventBuilder` class:

```python
from __future__ import annotations

from typing import Any, Dict

from .move_vocab import MoveVocab
from .rollout_store import RolloutRow
from .types import EventSequence
from .value_target_blend import compute_blended_value_target

EVENT_TOKEN_ID = 0
BOS_TOKEN_ID = 1
TARGET_IGNORE_INDEX = -100


def _result_to_game_result_white(value: Any) -> int:
    text = str(value).strip()
    if text == "1-0":
        return 1
    if text == "0-1":
        return -1
    if text == "1/2-1/2":
        return 0
    raise ValueError(f"Unsupported game result for value target: {text!r}")


class EventBuilder:
    """Build BOS+ply event sequences for next-move prediction."""

    def __init__(
        self,
        move_vocab: MoveVocab,
        *,
        rollout_lookup: Dict[tuple[str, int], RolloutRow] | None = None,
        beta: float = 0.0,
    ) -> None:
        self.move_vocab = move_vocab
        self.rollout_lookup = rollout_lookup
        self.beta = beta

    def build_game(self, game: Dict[str, Any]) -> EventSequence:
        game_result_white = _result_to_game_result_white(game["result"])

        seq_token_id = [BOS_TOKEN_ID]
        piece_ids = [[0] * 64]
        turn_id = [0]
        castle_id = [0]
        ep_file_id = [0]
        halfmove_bucket_id = [0]
        fullmove_bucket_id = [0]
        prev_move_id = [self.move_vocab.start_id]
        target_move_id = [TARGET_IGNORE_INDEX]
        played_by_elo = [0]

        previous_move = self.move_vocab.start_id
        for play in game["plays"]:
            state = play["state"]
            current_move = self.move_vocab.encode(play["move_uci"])
            current_played_by_elo = int(play.get("played_by_elo", 0))

            seq_token_id.append(EVENT_TOKEN_ID)
            piece_ids.append(list(state["piece_ids"]))
            turn_id.append(int(state["turn_id"]))
            castle_id.append(int(state["castle_id"]))
            ep_file_id.append(int(state["ep_file_id"]))
            halfmove_bucket_id.append(int(state["halfmove_bucket_id"]))
            fullmove_bucket_id.append(int(state["fullmove_bucket_id"]))
            prev_move_id.append(previous_move)
            target_move_id.append(current_move)
            played_by_elo.append(current_played_by_elo)

            previous_move = current_move

        result: EventSequence = {
            "game_id": game["game_id"],
            "game_result_white": game_result_white,
            "seq_token_id": seq_token_id,
            "piece_ids": piece_ids,
            "turn_id": turn_id,
            "castle_id": castle_id,
            "ep_file_id": ep_file_id,
            "halfmove_bucket_id": halfmove_bucket_id,
            "fullmove_bucket_id": fullmove_bucket_id,
            "prev_move_id": prev_move_id,
            "target_move_id": target_move_id,
            "played_by_elo": played_by_elo,
        }
        if self.rollout_lookup is not None:
            value_target_soft, has_rollout_value_target = self._build_rollout_value_targets(game)
            result["value_target_soft"] = value_target_soft
            result["has_rollout_value_target"] = has_rollout_value_target
        return result

    def _build_rollout_value_targets(
        self, game: Dict[str, Any]
    ) -> tuple[list[list[float]], list[int]]:
        assert self.rollout_lookup is not None
        num_tokens = len(game["plays"]) + 1
        value_target_soft: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(num_tokens)]
        has_rollout_value_target = [0] * num_tokens
        game_id = game["game_id"]

        for ply_idx in range(len(game["plays"])):
            row = self.rollout_lookup.get((game_id, ply_idx))
            if row is None:
                continue
            token_idx = ply_idx + 1
            value_target_soft[token_idx] = compute_blended_value_target(
                root_wdl_unsearched=row.root_wdl_unsearched,
                backed_value=row.best_arm_backed_value,
                real_outcome_stm=row.real_outcome_stm,
                beta=self.beta,
            )
            has_rollout_value_target[token_idx] = 1

        return value_target_soft, has_rollout_value_target
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_event_builder.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/data/types.py src/imba_chess/data/event_builder.py tests/test_event_builder.py
git commit -m "feat: wire optional rollout value-target lookup into EventBuilder"
```

---

## Task 7: `collate_jagged_batch` optional rollout fields (spec Part 4, collate side)

**Files:**
- Modify: `src/imba_chess/data/collate.py`
- Test: `tests/test_collate.py`

**Interfaces:**
- Consumes: `EventSequence`'s optional `value_target_soft`/`has_rollout_value_target` keys (Task 6).
- Produces: `JaggedBatch["value_target_soft"]` (`torch.float32`, shape `[total_tokens, 3]`) and `JaggedBatch["has_rollout_value_target"]` (`torch.bool`, shape `[total_tokens]`) — present iff every sample in the batch has them.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_collate.py`:

```python
def _sample_with_rollout(game_id, seq_len, value_target_soft, has_rollout):
    return {
        "game_id": game_id,
        "game_result_white": 1,
        "seq_token_id": [1] + [0] * (seq_len - 1),
        "piece_ids": [[0] * 64 for _ in range(seq_len)],
        "turn_id": [0] * seq_len,
        "castle_id": [15] * seq_len,
        "ep_file_id": [0] * seq_len,
        "halfmove_bucket_id": [0] * seq_len,
        "fullmove_bucket_id": [0] * seq_len,
        "prev_move_id": [1] * seq_len,
        "target_move_id": [-100] + [4] * (seq_len - 1),
        "played_by_elo": [0] * seq_len,
        "value_target_soft": value_target_soft,
        "has_rollout_value_target": has_rollout,
    }


def test_collate_includes_rollout_fields_when_present_on_every_sample():
    batch = [
        _sample_with_rollout("g1", 2, [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [0, 1]),
        _sample_with_rollout("g2", 2, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0, 0]),
    ]
    out = collate_jagged_batch(batch)
    assert out["value_target_soft"].shape == (4, 3)
    assert out["has_rollout_value_target"].shape == (4,)
    assert out["has_rollout_value_target"].dtype == torch.bool
    assert out["has_rollout_value_target"].tolist() == [False, True, False, False]
    assert out["value_target_soft"][1].tolist() == pytest.approx([0.2, 0.3, 0.5])


def test_collate_raises_on_mixed_rollout_key_presence():
    batch = [
        _sample_with_rollout("g1", 2, [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [0, 1]),
        {
            "game_id": "g2",
            "game_result_white": -1,
            "seq_token_id": [1, 0],
            "piece_ids": [[0] * 64, [1] * 64],
            "turn_id": [0, 0],
            "castle_id": [0, 15],
            "ep_file_id": [0, 0],
            "halfmove_bucket_id": [0, 0],
            "fullmove_bucket_id": [0, 0],
            "prev_move_id": [1, 1],
            "target_move_id": [-100, 3],
            "played_by_elo": [0, 2320],
        },
    ]
    with pytest.raises(ValueError, match="Mixed presence"):
        collate_jagged_batch(batch)


def test_collate_without_rollout_fields_unchanged():
    # Existing test_collate_jagged_batch_shapes_and_offsets already covers
    # this; this test only asserts the new keys are absent.
    batch = [
        _sample_with_rollout("g1", 2, [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [0, 1])
    ]
    del batch[0]["value_target_soft"]
    del batch[0]["has_rollout_value_target"]
    out = collate_jagged_batch(batch)
    assert "value_target_soft" not in out
    assert "has_rollout_value_target" not in out
```

(Add `import pytest` to `tests/test_collate.py` if not already present — it currently only imports `torch` via `pytest.importorskip`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collate.py -v`
Expected: the three new tests FAIL — `KeyError`/`AssertionError` since `collate_jagged_batch` doesn't yet look at the new keys.

- [ ] **Step 3: Implement the change in `src/imba_chess/data/collate.py`**

```python
from __future__ import annotations

from typing import List

from .types import EventSequence, JaggedBatch

def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required for collate_jagged_batch") from exc
    return torch


def collate_jagged_batch(batch: List[EventSequence]) -> JaggedBatch:
    """Flatten event sequences into jagged tensors with seq_lens/seq_offsets."""
    if not batch:
        raise ValueError("collate_jagged_batch received an empty batch")

    torch = _require_torch()

    scalar_keys = [
        "seq_token_id",
        "turn_id",
        "castle_id",
        "ep_file_id",
        "halfmove_bucket_id",
        "fullmove_bucket_id",
        "prev_move_id",
        "target_move_id",
        "played_by_elo",
    ]
    per_game_scalar_keys = ["game_result_white"]
    rollout_keys = ("value_target_soft", "has_rollout_value_target")

    sample_has_rollout = ["value_target_soft" in sample for sample in batch]
    if any(sample_has_rollout) and not all(sample_has_rollout):
        raise ValueError(
            "Mixed presence of rollout value-target keys across a collated batch: "
            "every sample from one EventBuilder must consistently include or "
            "omit value_target_soft/has_rollout_value_target."
        )
    include_rollout = all(sample_has_rollout)

    flat_scalars = {key: [] for key in scalar_keys}
    flat_piece_ids: list[list[int]] = []
    flat_value_target_soft: list[list[float]] = []
    flat_has_rollout_value_target: list[int] = []
    seq_lens: list[int] = []
    per_game_scalars = {key: [] for key in per_game_scalar_keys}

    for sample in batch:
        seq_len = len(sample["seq_token_id"])
        game_id = sample.get("game_id", "<unknown>")
        piece_ids = sample["piece_ids"]
        if len(piece_ids) != seq_len:
            raise ValueError(
                f"Sample {game_id} has piece_ids length {len(piece_ids)} "
                f"but seq_token_id length {seq_len}"
            )
        for key in scalar_keys:
            values = sample[key]
            if len(values) != seq_len:
                raise ValueError(
                    f"Sample {game_id} has {key} length {len(values)} "
                    f"but seq_token_id length {seq_len}"
                )
        if include_rollout:
            for key in rollout_keys:
                values = sample[key]
                if len(values) != seq_len:
                    raise ValueError(
                        f"Sample {game_id} has {key} length {len(values)} "
                        f"but seq_token_id length {seq_len}"
                    )
        seq_lens.append(seq_len)
        flat_piece_ids.extend(piece_ids)
        for key in scalar_keys:
            flat_scalars[key].extend(sample[key])
        for key in per_game_scalar_keys:
            per_game_scalars[key].append(sample[key])
        if include_rollout:
            flat_value_target_soft.extend(sample["value_target_soft"])
            flat_has_rollout_value_target.extend(sample["has_rollout_value_target"])

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)

    output: JaggedBatch = {
        "game_id": [sample["game_id"] for sample in batch],
        "num_games": len(batch),
        "total_tokens": offsets[-1],
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "seq_offsets": torch.tensor(offsets, dtype=torch.long),
        "piece_ids": torch.tensor(flat_piece_ids, dtype=torch.long),
    }

    for key in scalar_keys:
        output[key] = torch.tensor(flat_scalars[key], dtype=torch.long)
    for key in per_game_scalar_keys:
        output[key] = torch.tensor(per_game_scalars[key], dtype=torch.long)
    if include_rollout:
        output["value_target_soft"] = torch.tensor(flat_value_target_soft, dtype=torch.float32)
        output["has_rollout_value_target"] = torch.tensor(
            flat_has_rollout_value_target, dtype=torch.bool
        )

    return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collate.py -v`
Expected: all PASS, including the two pre-existing tests (regression check).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/data/collate.py tests/test_collate.py
git commit -m "feat: collate optional rollout value-target fields into JaggedBatch"
```

---

## Task 8: `build_event_dataloader` + `scripts/train.py` wiring (spec Part 4, config wiring)

Only the **train** loader in `scripts/train.py` gets the rollout lookup; the fast-val/full-val/fast-test/test loaders never see it — matches the spec's "Out of scope: rollout sampling from val/test splits."

**Files:**
- Modify: `src/imba_chess/data/dataloader.py`
- Modify: `scripts/train.py`
- Test: `tests/test_dataloader.py`

**Interfaces:**
- Consumes: `imba_chess.data.rollout_store.load_rollout_lookup` (Task 4).
- Produces: `build_event_dataloader(*, lichess_dataset, config=None, move_vocab=None, rollout_lookup=None, rollout_beta=0.0)` — two new keyword-only params, both defaulting to the byte-identical case.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dataloader.py`:

```python
def test_build_event_dataloader_threads_rollout_lookup_into_event_builder():
    from imba_chess.data.rollout_store import RolloutRow

    games = [_game("g1", "e2e4", "e7e5")]
    vocab = MoveVocab.build_from_games(games)
    dataset = DummyLichessDataset(games)
    rollout_row = RolloutRow(
        game_id="g1",
        ply=0,
        human_move_uci="e2e4",
        human_move_backed_value=0.2,
        real_outcome_stm=1,
        best_arm_move_uci="e2e4",
        best_arm_backed_value=0.6,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=("e2e4",),
        arm_backed_value=(0.6,),
        arm_evals_spent=(50,),
        arm_log_prior=(-0.1,),
        search_budget=256,
        search_top_m=1,
        search_max_depth=4,
        checkpoint="dummy.pt",
    )

    loader = build_event_dataloader(
        lichess_dataset=dataset,
        config=RepoConfig(dataloader=DataloaderConfig(max_tokens_per_batch=1024)),
        move_vocab=vocab,
        rollout_lookup={("g1", 0): rollout_row},
        rollout_beta=1.0,
    )

    batch = next(iter(loader))
    assert "value_target_soft" in batch
    assert "has_rollout_value_target" in batch
    assert bool(batch["has_rollout_value_target"][1].item()) is True


def test_build_event_dataloader_without_rollout_lookup_omits_fields():
    games = [_game("g1", "e2e4", "e7e5")]
    vocab = MoveVocab.build_from_games(games)
    dataset = DummyLichessDataset(games)

    loader = build_event_dataloader(
        lichess_dataset=dataset,
        config=RepoConfig(dataloader=DataloaderConfig(max_tokens_per_batch=1024)),
        move_vocab=vocab,
    )

    batch = next(iter(loader))
    assert "value_target_soft" not in batch
    assert "has_rollout_value_target" not in batch
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataloader.py -v`
Expected: the first new test FAILS with `TypeError: build_event_dataloader() got an unexpected keyword argument 'rollout_lookup'`.

- [ ] **Step 3: Implement the change in `src/imba_chess/data/dataloader.py`**

```python
def build_event_dataloader(
    *,
    lichess_dataset: Any,
    config: Optional[RepoConfig] = None,
    move_vocab: Optional[MoveVocab] = None,
    rollout_lookup: Optional[dict] = None,
    rollout_beta: float = 0.0,
) -> Any:
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("torch is required to build DataLoader")

    runtime = config or RepoConfig()
    num_workers = int(runtime.dataloader.num_workers)
    prefetch_factor = runtime.dataloader.prefetch_factor
    persistent_workers = bool(runtime.dataloader.persistent_workers)

    if num_workers < 0:
        raise ValueError("dataloader.num_workers must be >= 0")
    if prefetch_factor is not None and prefetch_factor < 1:
        raise ValueError("dataloader.prefetch_factor must be >= 1 when set")
    if prefetch_factor is not None and num_workers == 0:
        raise ValueError(
            "dataloader.prefetch_factor requires dataloader.num_workers > 0"
        )
    if persistent_workers and num_workers == 0:
        raise ValueError(
            "dataloader.persistent_workers=true requires dataloader.num_workers > 0"
        )

    resolved_move_vocab = move_vocab or load_or_create_static_move_vocab(
        path=runtime.vocab.path,
        include_unk=runtime.vocab.include_unk,
    )
    game_iterable_dataset = lichess_dataset.as_torch_iterable(
        rank=runtime.dataloader.rank,
        world_size=runtime.dataloader.world_size,
    )
    event_builder = EventBuilder(
        resolved_move_vocab, rollout_lookup=rollout_lookup, beta=rollout_beta
    )
    event_dataset = ChessEventIterableDataset(game_iterable_dataset, event_builder)
    packed_dataset = MaxTokensJaggedBatchDataset(
        event_dataset=event_dataset,
        max_tokens_per_batch=runtime.dataloader.max_tokens_per_batch,
    )

    dataloader_kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": runtime.dataloader.pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(
        packed_dataset,
        **dataloader_kwargs,
    )
```

(Only the signature and the `event_builder = EventBuilder(...)` line change; everything else in the function is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataloader.py -v`
Expected: all PASS.

- [ ] **Step 5: Wire it into `scripts/train.py` (train loader only)**

Add the import near the top of `scripts/train.py`:

```python
from imba_chess.data.rollout_store import load_rollout_lookup
```

Replace the `train_loader = build_event_dataloader(...)` call (currently right after the `if args.eval_only: ... return` block) with:

```python
    rollout_lookup = None
    if repo_config.expert_iteration.rollout_path:
        rollout_lookup = load_rollout_lookup(repo_config.expert_iteration.rollout_path)
        print(
            f"Loaded {len(rollout_lookup)} rollout value targets from "
            f"{repo_config.expert_iteration.rollout_path} "
            f"(beta={repo_config.expert_iteration.beta})"
        )

    train_loader = build_event_dataloader(
        lichess_dataset=_make_dataset(repo_config, split="train"),
        config=repo_config,
        move_vocab=move_vocab,
        rollout_lookup=rollout_lookup,
        rollout_beta=float(repo_config.expert_iteration.beta),
    )
```

Do **not** touch the `fast_val_loader`, `full_val_loader`, `fast_test_loader`, or `test_loader` construction calls — they must keep calling `build_event_dataloader` without `rollout_lookup`/`rollout_beta`.

- [ ] **Step 6: Commit**

```bash
git add src/imba_chess/data/dataloader.py scripts/train.py tests/test_dataloader.py
git commit -m "feat: thread rollout value-target lookup into the train loader only"
```

---

## Task 9: Per-token conditional value loss in `HSTUChessModel` (spec Part 4, model side)

**Files:**
- Modify: `src/imba_chess/model/hstu_model.py`
- Test: `tests/test_hstu_model.py`

**Interfaces:**
- Consumes: `batch.get("has_rollout_value_target")` (`torch.bool[N]` or absent), `batch.get("value_target_soft")` (`torch.float32[N, 3]` or absent).
- Behavior: for tokens where `has_rollout_value_target` is `True`, the per-token value loss is soft cross-entropy against `value_target_soft`; for every other token (including all tokens when the batch has no rollout keys at all), it is the existing hard cross-entropy against `value_target`. The existing `value_weights` (progress-based, Elo-scaled) apply identically to both — only the per-token loss *formula* changes, not the weighting.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hstu_model.py`:

```python
def test_hstu_chess_model_value_loss_uses_soft_ce_for_gated_tokens():
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
    )
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()

    value_target_soft = torch.zeros((total_tokens, 3), dtype=torch.float32)
    value_target_soft[1] = torch.tensor([0.1, 0.2, 0.7])
    has_rollout_value_target = torch.zeros(total_tokens, dtype=torch.bool)
    has_rollout_value_target[1] = True
    batch["value_target_soft"] = value_target_soft
    batch["has_rollout_value_target"] = has_rollout_value_target

    out = model(batch, return_loss=True)
    assert torch.isfinite(out["value_loss"])

    # Manually recompute expected per-token loss: soft CE at token 1, hard CE elsewhere.
    value_logits = out["value_logits"]
    seq_offsets = batch["seq_offsets"].to(dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(torch.arange(batch["num_games"]), counts)
    z_token = batch["game_result_white"].to(dtype=torch.long)[token_game_id]
    turn_id = batch["turn_id"].to(dtype=torch.long)
    value_target = (torch.where(turn_id == 0, z_token, -z_token) + 1).clamp(min=0, max=2)
    hard_loss = F.cross_entropy(value_logits.float(), value_target, reduction="none")
    soft_loss = -(value_target_soft * F.log_softmax(value_logits.float(), dim=-1)).sum(dim=-1)
    expected_per_token = torch.where(has_rollout_value_target, soft_loss, hard_loss)

    token_pos = torch.arange(value_logits.shape[0]) - seq_offsets[token_game_id]
    seq_len = counts[token_game_id].clamp_min(1)
    progress = token_pos.to(torch.float32) / (seq_len.to(torch.float32) - 1.0).clamp_min(1.0)
    valid_mask = batch["target_move_id"].to(dtype=torch.long) != config.ignore_index
    value_weights = progress.pow(config.value_weight_alpha) * valid_mask.to(torch.float32)
    expected = (expected_per_token * value_weights).sum() / value_weights.sum().clamp_min(1.0)

    assert torch.allclose(out["value_loss"], expected, atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_value_loss_beta_zero_soft_target_matches_hard_ce():
    # A one-hot soft target must produce numerically the same per-token loss
    # as the existing hard-CE path (this is what beta=0 relies on).
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
    )
    torch.manual_seed(0)
    model = HSTUChessModel(config)
    batch = _batch()
    total_tokens = batch["seq_token_id"].numel()

    out_baseline = model(batch, return_loss=True)

    # one_hot(value_target) as the soft target, gated everywhere valid.
    seq_offsets = batch["seq_offsets"].to(dtype=torch.long)
    counts = seq_offsets[1:] - seq_offsets[:-1]
    token_game_id = torch.repeat_interleave(torch.arange(batch["num_games"]), counts)
    z_token = batch["game_result_white"].to(dtype=torch.long)[token_game_id]
    turn_id = batch["turn_id"].to(dtype=torch.long)
    value_target = (torch.where(turn_id == 0, z_token, -z_token) + 1).clamp(min=0, max=2)
    batch["value_target_soft"] = F.one_hot(value_target, num_classes=3).to(torch.float32)
    batch["has_rollout_value_target"] = torch.ones(total_tokens, dtype=torch.bool)

    torch.manual_seed(0)
    model_gated = HSTUChessModel(config)
    out_gated = model_gated(batch, return_loss=True)

    assert torch.allclose(out_gated["value_loss"], out_baseline["value_loss"], atol=1e-6, rtol=1e-6)


def test_hstu_chess_model_value_loss_without_rollout_keys_unchanged():
    # Regression: a batch entirely lacking the two optional keys must behave
    # exactly as before this change.
    config = HSTUChessConfig(
        move_vocab_size=128,
        model_dim=64,
        linear_hidden_dim=16,
        attention_dim=16,
        num_heads=2,
        num_layers=0,
        max_position_embeddings=32,
        enable_value_head=True,
        value_loss_weight=0.1,
    )
    torch.manual_seed(0)
    model_a = HSTUChessModel(config)
    torch.manual_seed(0)
    model_b = HSTUChessModel(config)
    batch = _batch()

    out_a = model_a(batch, return_loss=True)
    out_b = model_b(batch, return_loss=True)
    assert torch.allclose(out_a["value_loss"], out_b["value_loss"], atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hstu_model.py -k "soft_ce or beta_zero" -v`
Expected: the first two new tests FAIL — the model currently ignores `has_rollout_value_target`/`value_target_soft` entirely, so `out["value_loss"]` won't match the gated-expected computation (the third test should already PASS since it doesn't touch new behavior; confirm it does as an early regression check).

- [ ] **Step 3: Implement the change in `src/imba_chess/model/hstu_model.py`**

Replace the value-loss block inside `forward()` (currently the `if value_logits is not None:` block) with:

```python
            if value_logits is not None:
                game_result_white = batch["game_result_white"].to(
                    device=policy_logits.device, dtype=torch.long, non_blocking=True
                )
                if game_result_white.ndim != 1 or int(game_result_white.shape[0]) != batch_games:
                    raise ValueError(
                        "game_result_white must have shape [B] where B == num_games"
                    )
                z_token = game_result_white.index_select(0, token_game_id)
                turn_id = batch["turn_id"].to(
                    device=policy_logits.device, dtype=torch.long, non_blocking=True
                )
                y = torch.where(turn_id == 0, z_token, -z_token)
                value_target = (y + 1).clamp(min=0, max=2)

                progress = token_pos_in_game.to(torch.float32) / (
                    seq_len_for_token.to(torch.float32) - 1.0
                ).clamp_min(1.0)
                value_weights = progress.pow(self.config.value_weight_alpha)
                value_weights = value_weights * valid_mask.to(value_weights.dtype)
                if elo_scale is not None:
                    value_weights = value_weights * elo_scale.to(value_weights.dtype)

                per_token_value_loss = F.cross_entropy(
                    value_logits.float(),
                    value_target,
                    reduction="none",
                    label_smoothing=self.config.value_label_smoothing,
                )

                has_rollout_value_target = batch.get("has_rollout_value_target")
                value_target_soft = batch.get("value_target_soft")
                if has_rollout_value_target is not None and value_target_soft is not None:
                    rollout_mask = has_rollout_value_target.to(
                        device=policy_logits.device, dtype=torch.bool
                    )
                    if bool(rollout_mask.any()):
                        soft_targets = value_target_soft.to(
                            device=policy_logits.device, dtype=torch.float32
                        )
                        per_token_soft_loss = -(
                            soft_targets * F.log_softmax(value_logits.float(), dim=-1)
                        ).sum(dim=-1)
                        per_token_value_loss = torch.where(
                            rollout_mask, per_token_soft_loss, per_token_value_loss
                        )

                value_loss_sum = (per_token_value_loss * value_weights).sum()
                value_weight_sum = value_weights.sum().clamp_min(1.0)
                value_loss = value_loss_sum / value_weight_sum
                output["value_loss"] = value_loss
                total_loss = total_loss + self.config.value_loss_weight * value_loss
```

(Only the addition of `has_rollout_value_target`/`value_target_soft` reading and the `torch.where` blend is new; every other line in this block is unchanged from the current file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hstu_model.py -v`
Expected: all PASS, including every pre-existing value-loss test (regression check for byte-identical behavior when the new keys are absent).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/model/hstu_model.py tests/test_hstu_model.py
git commit -m "feat: per-token conditional soft-CE value loss for rollout-gated tokens"
```

---

## Task 10: End-to-end smoke test (spec Testing section, last bullet)

Ties Tasks 3-9 together: a tiny synthetic rollout parquet, run through `EventBuilder` -> `collate_jagged_batch` -> `HSTUChessModel`, confirming the soft-CE path is actually exercised (not silently falling back to hard CE) and the loss is finite across a few steps.

**Files:**
- Test: `tests/test_expert_iteration_end_to_end.py`

**Interfaces:**
- Consumes everything from Tasks 3, 4, 6, 7, 9 together; no new production code.

- [ ] **Step 1: Write the test**

Create `tests/test_expert_iteration_end_to_end.py`:

```python
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from imba_chess.config import ModelConfig
from imba_chess.data.collate import collate_jagged_batch
from imba_chess.data.event_builder import EventBuilder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.data.rollout_store import RolloutRow, load_rollout_lookup, write_rollout_parquet
from imba_chess.model import HSTUChessModel, build_hstu_chess_config


def _row():
    return {
        "Event": "Rated Blitz game",
        "Site": "https://lichess.org/e2e-example",
        "UTCDate": "2026-01-01",
        "UTCTime": "12:00:00",
        "White": "Alice",
        "Black": "Bob",
        "WhiteElo": "2200",
        "BlackElo": "2200",
        "Result": "1-0",
        "TimeControl": "300+0",
        "Termination": "Normal",
        "ECO": "C20",
        "Opening": "King's Pawn Game",
        "movetext": "1. e4 e5 2. Nf3 Nc6 1-0",
    }


def test_end_to_end_training_step_with_rollout_targets(tmp_path):
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])
    game_id = game["game_id"]

    rollout_row = RolloutRow(
        game_id=game_id,
        ply=1,
        human_move_uci=game["plays"][1]["move_uci"],
        human_move_backed_value=0.1,
        real_outcome_stm=-1,
        best_arm_move_uci=game["plays"][1]["move_uci"],
        best_arm_backed_value=0.4,
        root_wdl_unsearched=(0.3, 0.3, 0.4),
        arm_move_uci=(game["plays"][1]["move_uci"],),
        arm_backed_value=(0.4,),
        arm_evals_spent=(64,),
        arm_log_prior=(-0.2,),
        search_budget=128,
        search_top_m=1,
        search_max_depth=4,
        checkpoint="dummy.pt",
    )
    rollout_path = tmp_path / "rollouts.parquet"
    write_rollout_parquet([rollout_row], rollout_path)
    lookup = load_rollout_lookup(rollout_path)

    builder = EventBuilder(vocab, rollout_lookup=lookup, beta=0.7)
    sample = builder.build_game(game)
    assert sum(sample["has_rollout_value_target"]) == 1

    batch = collate_jagged_batch([sample])
    assert "value_target_soft" in batch
    assert "has_rollout_value_target" in batch

    model_cfg = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=16,
            attention_dim=16,
            num_heads=2,
            num_layers=1,
            max_position_embeddings=32,
            enable_value_head=True,
            value_loss_weight=0.2,
        ),
        move_vocab_size=len(vocab),
    )
    model = HSTUChessModel(model_cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(3):
        optimizer.zero_grad()
        out = model(batch, return_loss=True)
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["value_loss"])
        out["loss"].backward()
        optimizer.step()

    # The value head must actually have received gradients through the
    # soft-CE path (not silently skipped): check at least one of its
    # parameters moved.
    grad_norms = [
        p.grad.norm().item() for p in model.value_head.parameters() if p.grad is not None
    ]
    assert any(norm > 0.0 for norm in grad_norms)
```

- [ ] **Step 2: Run the test**

This task has no new production code — it only exercises the integration of Tasks 3-9, all of which are already implemented by this point in the plan, so unlike the other tasks there is no red phase here.

Run: `pytest tests/test_expert_iteration_end_to_end.py -v`
Expected: PASS. If it fails, that indicates a real integration gap between two tasks that the per-task unit tests didn't catch — re-check the `(game_id, ply)` join-key convention first (`ply` is 0-indexed into `game["plays"]`, `token_idx = ply + 1`), since that convention is shared across Tasks 4, 5, and 6 without a single test that spans all three.

- [ ] **Step 3: Run the full test suite as a final regression check**

Run: `pytest tests/ -v`
Expected: all PASS. This is the final check that nothing in Tasks 1-9 broke any pre-existing test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_expert_iteration_end_to_end.py
git commit -m "test: end-to-end smoke test for search-backed value distillation (Phase 1a)"
```

---

## Explicitly out of scope for this plan

Matches the spec's "Out of scope (v1 / Phase 1a)" section — do not implement any of these here:

- Phase 1b (policy target from `evals_spent`, confidence gating margin `m`).
- Phase 2 (self-play generation, GRPO/policy-gradient updates).
- Iterative rounds (regenerating rollouts from an improved checkpoint and retraining again).
- Rollout sampling from val/test splits.
- Any change to the standalone value net or `value_net_alpha` inference blend.
- Actually running the tuning-methodology funnel (label-level `β` sweep, validation-loss-proxy fine-tunes, live SF2200 eval) — that's an experiment to run once this plan's code exists, not a coding task in this plan.
