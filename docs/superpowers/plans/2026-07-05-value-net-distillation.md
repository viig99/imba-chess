# Standalone Value Network (Stockfish Distillation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A position-only WDL value network trained on `Lichess/chess-position-evaluations`, blended into search via `value_net_alpha` in `CachedPositionEvaluator`, per `docs/superpowers/specs/2026-07-05-value-net-distillation-design.md`.

**Architecture:** `ValueNet` reuses `BoardSquareEncoder` (bidirectional attention over 64 squares) with scalar-feature embeddings broadcast-added to square tokens; a streaming dataset converts eval-DB rows to side-to-move-POV soft WDL targets via Stockfish 17's vendored `win_rate_model`; a lean standalone trainer; the eval script loads the net optionally and blends `value_stm = (1−α)·model + α·net` inside the evaluator's existing wave.

**Tech Stack:** PyTorch, python-chess, HF `datasets` streaming, `torch_optimi.StableAdamW`, pytest. No new dependencies.

## Global Constraints

- No new third-party dependencies.
- `src/imba_chess/eval/search.py` and `tests/test_search.py` untouched.
- With `value_net_checkpoint` unset (default) the eval script's behavior is **byte-identical to today** — all existing eval tests pass unchanged.
- WDL index convention everywhere: index 0 = loss, 1 = draw, 2 = win, side-to-move POV (matches `_value_scalar_from_logits`: `p[2] − p[0]`).
- Lichess `cp`/`mate` are **White-POV**; the side-to-move flip is a tested invariant.
- ValueNet is position-only and clock-blind: consumes `piece_ids [B,64]`, `turn_id`, `castle_id`, `ep_file_id` only; ignores unknown batch keys.
- SF `win_rate_model` constants vendored verbatim from Stockfish 17 `src/uci.cpp` (cited in a comment): `as = {-37.45051876, 121.19101539, -132.78783573, 420.70576692}`, `bs = {90.26261072, -137.26549898, 71.10130540, 51.35259597}`, material clamp `[17, 78] / 58.0`, value clamp `±4000` internal units; UCI cp maps to internal units via `v = cp * a(material) / 100` (SF's normalized-cp convention: +100 cp ≡ 50% win).
- Mate-row target: 0.995 mass on the winning class, 0.0025 on each other class, side-to-move POV.
- Verified parquet types (trust these, not the dataset card's prose): `fen` str, `line` str, `depth` uint8, `knodes` int32, `cp` int16 nullable, `mate` int8 nullable.
- Test command: `.venv/bin/python -m pytest ...` (in a worktree: prefix `PYTHONPATH=src`). No test may hit the network — dataset tests go through `samples_from_rows` with scripted rows.

---

### Task 1: `cp_to_wdl` + `ValueNet` model

**Files:**
- Create: `src/imba_chess/model/value_net.py`
- Test: `tests/test_value_net.py` (new)

**Interfaces:**
- Consumes: `BoardSquareEncoder` from `imba_chess.model.hstu_model` (constructor `(*, dim, num_heads, num_layers, out_dim)`).
- Produces (used by Tasks 2–4):
  - `ValueNetConfig(dim=256, num_heads=4, num_layers=6)` frozen dataclass.
  - `ValueNet(config).forward(batch: dict) -> Tensor [B, 3]` — keys used: `piece_ids [B,64]`, `turn_id [B]`, `castle_id [B]`, `ep_file_id [B]`; extra keys ignored; tensors moved to the model's device internally.
  - `cp_to_wdl(cp: int, material: int) -> tuple[float, float, float]` — `(p_loss, p_draw, p_win)`, side-independent pure function.
  - `board_material_count(board: chess.Board) -> int` — SF material formula over both sides.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_value_net.py`:

```python
from __future__ import annotations

import chess
import pytest

torch = pytest.importorskip("torch")

from imba_chess.model.value_net import (
    ValueNet,
    ValueNetConfig,
    board_material_count,
    cp_to_wdl,
)


def test_cp_to_wdl_sums_to_one_and_is_symmetric():
    for cp in [-4000, -300, -50, 0, 50, 300, 4000]:
        for material in [17, 40, 58, 78]:
            p_loss, p_draw, p_win = cp_to_wdl(cp, material)
            assert abs(p_loss + p_draw + p_win - 1.0) < 1e-9
            assert 0.0 <= p_draw <= 1.0
            # Symmetry: negating cp swaps win and loss exactly.
            q_loss, q_draw, q_win = cp_to_wdl(-cp, material)
            assert abs(p_win - q_loss) < 1e-9
            assert abs(p_draw - q_draw) < 1e-9


def test_cp_to_wdl_monotone_in_cp():
    for material in [20, 58, 78]:
        wins = [cp_to_wdl(cp, material)[2] for cp in range(-500, 501, 50)]
        assert all(b >= a for a, b in zip(wins, wins[1:]))


def test_cp_to_wdl_anchor_and_extremes():
    # SF normalized-cp convention: +100 cp == 50% win probability.
    _, _, p_win = cp_to_wdl(100, 58)
    assert abs(p_win - 0.5) < 1e-6
    # Equal position: win and loss mass are equal and small vs draw.
    p_loss, p_draw, p_win = cp_to_wdl(0, 58)
    assert abs(p_win - p_loss) < 1e-9
    assert p_draw > p_win
    # Huge advantage saturates.
    assert cp_to_wdl(4000, 58)[2] > 0.99
    # Extreme cp values beyond the clamp do not blow up.
    assert cp_to_wdl(100000, 58)[2] == pytest.approx(cp_to_wdl(20000, 58)[2], abs=1e-6)


def test_board_material_count():
    assert board_material_count(chess.Board()) == 78  # 16P + 4N*3 + 4B*3 + 4R*5 + 2Q*9
    assert board_material_count(chess.Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")) == 0


def test_value_net_forward_shapes_and_determinism():
    torch.manual_seed(0)
    net = ValueNet(ValueNetConfig(dim=32, num_heads=2, num_layers=2)).eval()
    batch = {
        "piece_ids": torch.randint(0, 13, (5, 64)),
        "turn_id": torch.randint(0, 2, (5,)),
        "castle_id": torch.randint(0, 16, (5,)),
        "ep_file_id": torch.randint(0, 9, (5,)),
        # Extra keys must be ignored (the eval wave batch carries them).
        "prev_move_id": torch.zeros(5, dtype=torch.long),
        "seq_token_id": torch.zeros(5, dtype=torch.long),
    }
    with torch.no_grad():
        out1 = net(batch)
        out2 = net(batch)
    assert out1.shape == (5, 3)
    torch.testing.assert_close(out1, out2)


def test_value_net_turn_changes_output():
    torch.manual_seed(1)
    net = ValueNet(ValueNetConfig(dim=32, num_heads=2, num_layers=2)).eval()
    base = {
        "piece_ids": torch.randint(0, 13, (1, 64)),
        "castle_id": torch.zeros(1, dtype=torch.long),
        "ep_file_id": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        white = net({**base, "turn_id": torch.tensor([0])})
        black = net({**base, "turn_id": torch.tensor([1])})
    assert not torch.allclose(white, black)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_value_net.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.model.value_net'`.

- [ ] **Step 3: Implement `src/imba_chess/model/value_net.py`**

```python
"""Position-only WDL value network distilled from Stockfish evaluations.

Consumes a single board state (no game history, no clocks) and predicts
win/draw/loss from the side-to-move POV. The body reuses the big model's
BoardSquareEncoder; scalar state features are broadcast-added to the 64
square tokens so side-to-move/castling can interact with square content.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import chess
import torch
import torch.nn as nn

from .hstu_model import BoardSquareEncoder

# Stockfish 17 win_rate_model (src/uci.cpp): polynomial coefficients for the
# logistic's midpoint (a) and slope (b) as functions of normalized material.
_SF17_AS = (-37.45051876, 121.19101539, -132.78783573, 420.70576692)
_SF17_BS = (90.26261072, -137.26549898, 71.10130540, 51.35259597)


def board_material_count(board: chess.Board) -> int:
    """Stockfish's material count over both sides: P + 3N + 3B + 5R + 9Q."""
    return (
        len(board.pieces(chess.PAWN, chess.WHITE))
        + len(board.pieces(chess.PAWN, chess.BLACK))
        + 3 * (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)))
        + 3 * (len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
        + 5 * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
        + 9 * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
    )


def _win_rate(v: float, a: float, b: float) -> float:
    return 1.0 / (1.0 + math.exp((a - v) / b))


def cp_to_wdl(cp: int, material: int) -> tuple[float, float, float]:
    """(p_loss, p_draw, p_win) from a UCI centipawn eval, given board material.

    Uses Stockfish 17's win_rate_model polynomial. Lichess cp values follow
    SF's normalized-cp convention (+100 cp == 50% win probability), so cp is
    mapped back to internal units via v = cp * a(material) / 100 before the
    logistic; p_loss is the same model evaluated at -v (symmetry), and draw
    mass is the remainder.
    """
    m = min(max(material, 17), 78) / 58.0
    a = ((_SF17_AS[0] * m + _SF17_AS[1]) * m + _SF17_AS[2]) * m + _SF17_AS[3]
    b = ((_SF17_BS[0] * m + _SF17_BS[1]) * m + _SF17_BS[2]) * m + _SF17_BS[3]
    v = cp * a / 100.0
    v = min(max(v, -4000.0), 4000.0)
    p_win = _win_rate(v, a, b)
    p_loss = _win_rate(-v, a, b)
    p_draw = max(0.0, 1.0 - p_win - p_loss)
    return (p_loss, p_draw, p_win)


@dataclass(frozen=True)
class ValueNetConfig:
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 6


class ValueNet(nn.Module):
    def __init__(self, config: ValueNetConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        self.piece_square_embedding = nn.Embedding(13 * 64, dim)
        self.turn_embedding = nn.Embedding(2, dim)
        self.castle_embedding = nn.Embedding(16, dim)
        self.ep_embedding = nn.Embedding(9, dim)
        self.encoder = BoardSquareEncoder(
            dim=dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            out_dim=dim,
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 3),
        )
        self.register_buffer(
            "square_ids", torch.arange(64, dtype=torch.long), persistent=False
        )

    def _clamp_ids(self, ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        return ids.clamp(min=0, max=num_embeddings - 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.piece_square_embedding.weight.device
        piece_ids = batch["piece_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        turn_id = self._clamp_ids(
            batch["turn_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.turn_embedding.num_embeddings,
        )
        castle_id = self._clamp_ids(
            batch["castle_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.castle_embedding.num_embeddings,
        )
        ep_file_id = self._clamp_ids(
            batch["ep_file_id"].to(device=device, dtype=torch.long, non_blocking=True),
            self.ep_embedding.num_embeddings,
        )

        pair_ids = piece_ids * 64 + self.square_ids
        squares = self.piece_square_embedding(pair_ids)  # [B, 64, dim]
        features = (
            self.turn_embedding(turn_id)
            + self.castle_embedding(castle_id)
            + self.ep_embedding(ep_file_id)
        )  # [B, dim]
        squares = squares + features.unsqueeze(1)
        pooled = self.encoder(squares)  # [B, dim]
        return self.head(pooled)  # [B, 3] WDL logits (0=loss, 1=draw, 2=win)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_value_net.py -v`
Expected: PASS (6 tests). If the anchor test fails, check the `v = cp * a / 100` mapping (the +100cp ≡ 50% convention) before touching constants.

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/model/value_net.py tests/test_value_net.py
git commit -m "feat: position-only ValueNet + vendored SF17 cp->WDL model"
```

---

### Task 2: Streaming position-eval dataset

**Files:**
- Create: `src/imba_chess/data/position_eval_dataset.py`
- Test: `tests/test_position_eval_dataset.py` (new)

**Interfaces:**
- Consumes (Task 1): `cp_to_wdl`, `board_material_count` from `imba_chess.model.value_net`; existing `BoardStateEncoder`.
- Produces (used by Task 3): `PositionEvalDataset(split="train"|"val", depth_min=12, dataset_name="Lichess/chess-position-evaluations", shuffle_buffer_size=10000, seed=0, val_permille=5)` — a torch `IterableDataset` yielding `{"piece_ids" [64] long, "turn_id" long, "castle_id" long, "ep_file_id" long, "wdl_target" [3] float32}` per sample; plus the testable core `samples_from_rows(rows: Iterable[dict]) -> Iterator[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_position_eval_dataset.py`:

```python
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imba_chess.data.position_eval_dataset import PositionEvalDataset
from imba_chess.model.value_net import cp_to_wdl

# 4-field FENs exactly as the eval DB provides them.
_START_W = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
_START_B = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"


def _row(**overrides):
    base = {"fen": _START_W, "line": "e2e4", "depth": 30, "knodes": 1000, "cp": 100, "mate": None}
    base.update(overrides)
    return base


def _dataset(**kwargs):
    defaults = dict(split="train", depth_min=12, val_permille=0)
    defaults.update(kwargs)
    return PositionEvalDataset(**defaults)


def test_pov_flip_black_to_move():
    # Same +100 White-POV eval; for Black to move that is a -100 stm eval.
    ds = _dataset()
    w = list(ds.samples_from_rows([_row(fen=_START_W)]))[0]
    b = list(ds.samples_from_rows([_row(fen=_START_B)]))[0]
    assert w["turn_id"].item() == 0 and b["turn_id"].item() == 1
    expected_w = torch.tensor(cp_to_wdl(100, 78))
    expected_b = torch.tensor(cp_to_wdl(-100, 78))
    torch.testing.assert_close(w["wdl_target"], expected_w.float())
    torch.testing.assert_close(b["wdl_target"], expected_b.float())
    # Flip symmetry: black's target is white's reversed.
    torch.testing.assert_close(b["wdl_target"], w["wdl_target"].flip(0))


def test_mate_rows_saturate_correct_side():
    ds = _dataset()
    # mate +3 (White mates). White to move: winning. Black to move: losing.
    w = list(ds.samples_from_rows([_row(cp=None, mate=3)]))[0]
    b = list(ds.samples_from_rows([_row(fen=_START_B, cp=None, mate=3)]))[0]
    torch.testing.assert_close(
        w["wdl_target"], torch.tensor([0.0025, 0.0025, 0.995])
    )
    torch.testing.assert_close(
        b["wdl_target"], torch.tensor([0.995, 0.0025, 0.0025])
    )


def test_filters_shallow_missing_and_invalid_rows():
    ds = _dataset(depth_min=12)
    rows = [
        _row(depth=5),                      # too shallow
        _row(cp=None, mate=None),           # no label
        _row(fen="not a fen"),              # unparseable
        _row(cp=None, mate=0),              # mate 0 is not a usable label
        _row(),                             # valid
    ]
    samples = list(ds.samples_from_rows(rows))
    assert len(samples) == 1


def test_holdout_split_is_deterministic_and_disjoint():
    train = _dataset(val_permille=200, split="train")
    val = _dataset(val_permille=200, split="val")
    # Distinct FENs via the castling-rights field (all parse to legal boards).
    fens = [_START_W, _START_W.replace("KQkq", "KQ"), _START_W.replace("KQkq", "kq"),
            _START_W.replace("KQkq", "K"), _START_W.replace("KQkq", "-")]
    rows = [_row(fen=f) for f in fens]
    n_train = len(list(train.samples_from_rows(rows)))
    n_val = len(list(val.samples_from_rows(rows)))
    assert n_train + n_val == len(fens)
    # Determinism: same result on a second pass.
    assert n_train == len(list(train.samples_from_rows(rows)))


def test_sample_tensor_shapes():
    sample = list(_dataset().samples_from_rows([_row()]))[0]
    assert sample["piece_ids"].shape == (64,)
    assert sample["wdl_target"].shape == (3,)
    assert sample["wdl_target"].dtype == torch.float32
    assert abs(float(sample["wdl_target"].sum()) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_position_eval_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imba_chess.data.position_eval_dataset'`.

- [ ] **Step 3: Implement `src/imba_chess/data/position_eval_dataset.py`**

```python
"""Streaming dataset over Lichess/chess-position-evaluations.

Yields flat per-position samples with soft WDL targets (side-to-move POV)
for training the standalone ValueNet. Verified parquet schema: fen str,
line str, depth uint8, knodes int32, cp int16 nullable, mate int8 nullable
(cp/mate mutually exclusive; the dataset card's prose is partly wrong).
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, Iterable, Iterator

import chess
import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info

from ..model.value_net import board_material_count, cp_to_wdl
from .board_state import BoardStateEncoder

_MATE_TARGET_WIN = (0.0025, 0.0025, 0.995)
_MATE_TARGET_LOSS = (0.995, 0.0025, 0.0025)


class PositionEvalDataset(IterableDataset):
    def __init__(
        self,
        *,
        split: str = "train",
        depth_min: int = 12,
        dataset_name: str = "Lichess/chess-position-evaluations",
        shuffle_buffer_size: int = 10_000,
        seed: int = 0,
        val_permille: int = 5,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0 <= val_permille <= 1000:
            raise ValueError("val_permille must be in [0, 1000]")
        self.split = split
        self.depth_min = int(depth_min)
        self.dataset_name = dataset_name
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.val_permille = int(val_permille)
        self._encoder = BoardStateEncoder()

    def _in_val(self, fen: str) -> bool:
        return zlib.crc32(fen.encode("utf-8")) % 1000 < self.val_permille

    def _row_to_sample(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor] | None:
        depth = row.get("depth")
        cp = row.get("cp")
        mate = row.get("mate")
        if depth is None or int(depth) < self.depth_min:
            return None
        if cp is None and mate is None:
            return None
        fen = row.get("fen")
        if not fen:
            return None
        if self._in_val(fen) != (self.split == "val"):
            return None
        try:
            board = chess.Board(fen)
        except ValueError:
            return None

        stm_sign = 1 if board.turn == chess.WHITE else -1
        if mate is not None:
            mate_stm = int(mate) * stm_sign
            if mate_stm == 0:
                return None
            target = _MATE_TARGET_WIN if mate_stm > 0 else _MATE_TARGET_LOSS
        else:
            target = cp_to_wdl(int(cp) * stm_sign, board_material_count(board))

        state = self._encoder.encode(board)
        return {
            "piece_ids": torch.tensor(state.piece_ids, dtype=torch.long),
            "turn_id": torch.tensor(state.turn_id, dtype=torch.long),
            "castle_id": torch.tensor(state.castle_id, dtype=torch.long),
            "ep_file_id": torch.tensor(state.ep_file_id, dtype=torch.long),
            "wdl_target": torch.tensor(target, dtype=torch.float32),
        }

    def samples_from_rows(
        self, rows: Iterable[Dict[str, Any]]
    ) -> Iterator[Dict[str, torch.Tensor]]:
        for row in rows:
            sample = self._row_to_sample(row)
            if sample is not None:
                yield sample

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rows = load_dataset(self.dataset_name, split="train", streaming=True)
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            rows = rows.shard(num_shards=worker.num_workers, index=worker.id)
        if self.split == "train" and self.shuffle_buffer_size > 0:
            rows = rows.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        yield from self.samples_from_rows(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_position_eval_dataset.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/imba_chess/data/position_eval_dataset.py tests/test_position_eval_dataset.py
git commit -m "feat: streaming position-eval dataset with stm-POV soft WDL targets"
```

---

### Task 3: Trainer script + `[value_net]` config

**Files:**
- Create: `scripts/train_value_net.py`
- Modify: `src/imba_chess/config.py` (new section dataclass + loader wiring)
- Modify: `config/imba_chess.toml` (new `[value_net]` section)
- Test: `tests/test_config.py` (append), `tests/test_value_net.py` (append trainer smoke)

**Interfaces:**
- Consumes: `ValueNet`, `ValueNetConfig` (Task 1); `PositionEvalDataset` (Task 2).
- Produces: `ValueNetSection` config dataclass (fields below) exposed as `RepoConfig.value_net`; checkpoints at `{checkpoint_dir}/value_net_best.pt` / `value_net_last.pt` containing `{"model": state_dict, "config": {"dim":…, "num_heads":…, "num_layers":…}, "step": int, "val_loss": float}`; `soft_cross_entropy(logits, targets) -> Tensor` importable from the script module.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_value_net_config_defaults():
    from imba_chess.config import ValueNetSection

    section = ValueNetSection()
    assert section.dim == 256
    assert section.num_heads == 4
    assert section.num_layers == 6
    assert section.depth_min == 12
    assert section.batch_size == 1024
    assert section.checkpoint_dir == "artifacts/value_net"


def test_load_repo_config_reads_value_net_section(tmp_path):
    config_path = tmp_path / "imba_chess.toml"
    config_path.write_text(
        """
[value_net]
dim = 128
train_steps = 500
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_repo_config(config_path)
    assert config.value_net.dim == 128
    assert config.value_net.train_steps == 500
    assert config.value_net.num_layers == 6
```

Append to `tests/test_value_net.py`:

```python
def test_trainer_smoke_one_step():
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_value_net.py"
    spec = importlib.util.spec_from_file_location("train_value_net_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    net = ValueNet(ValueNetConfig(dim=32, num_heads=2, num_layers=1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    batch = {
        "piece_ids": torch.randint(0, 13, (8, 64)),
        "turn_id": torch.randint(0, 2, (8,)),
        "castle_id": torch.randint(0, 16, (8,)),
        "ep_file_id": torch.randint(0, 9, (8,)),
        "wdl_target": torch.softmax(torch.randn(8, 3), dim=-1),
    }
    before = [p.detach().clone() for p in net.parameters()]
    loss = module.train_step(net, batch, optimizer, grad_clip_norm=1.0)
    assert torch.isfinite(torch.tensor(loss))
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(before, net.parameters())
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_value_net.py -v -k "value_net or trainer"`
Expected: FAIL — `ImportError` (no `ValueNetSection`), `FileNotFoundError`/`AttributeError` for the trainer.

- [ ] **Step 3: Add the config section**

In `src/imba_chess/config.py`, after `EvalVsStockfishConfig`, add:

```python
@dataclass(frozen=True)
class ValueNetSection:
    # Model (ValueNetConfig fields)
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    # Data
    dataset_name: str = "Lichess/chess-position-evaluations"
    depth_min: int = 12
    shuffle_buffer_size: int = 10_000
    val_permille: int = 5
    # Training
    batch_size: int = 1024
    num_workers: int = 2
    max_lr: float = 3e-4
    weight_decay: float = 0.01
    train_steps: int = 200_000
    log_every_steps: int = 100
    val_every_steps: int = 5_000
    val_batches: int = 50
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    checkpoint_dir: str = "artifacts/value_net"
```

Add `value_net: ValueNetSection = field(default_factory=ValueNetSection)` to
`RepoConfig`, and in `load_repo_config` add:

```python
        value_net=_read_section(
            ValueNetSection, payload.get("value_net", {}), "value_net"
        ),
```

In `config/imba_chess.toml`, append at the end:

```toml
[value_net]
# Standalone position-only WDL net distilled from Stockfish evals
# (Lichess/chess-position-evaluations). Trained by scripts/train_value_net.py;
# consumed at eval time via [eval_vs_stockfish] value_net_checkpoint.
dim = 256
num_heads = 4
num_layers = 6
depth_min = 12
batch_size = 1024
num_workers = 2
max_lr = 3e-4
train_steps = 200000
val_every_steps = 5000
checkpoint_dir = "artifacts/value_net"
```

- [ ] **Step 4: Implement `scripts/train_value_net.py`**

```python
#!/usr/bin/env python3
"""Train the standalone position-only WDL value net on Stockfish evals.

Lean flat-batch supervised loop: no jagged packing, no game parsing.
Usage: python scripts/train_value_net.py [--config config/imba_chess.toml]
       [--steps N] [--device cuda|cpu|auto]
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.position_eval_dataset import PositionEvalDataset
from imba_chess.model.value_net import ValueNet, ValueNetConfig


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def train_step(model, batch, optimizer, *, grad_clip_norm: float, autocast_ctx=None) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    with ctx:
        logits = model(batch)
        loss = soft_cross_entropy(logits, batch["wdl_target"].to(logits.device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def validate(model, loader, *, max_batches: int, device) -> tuple[float, float]:
    model.eval()
    losses, correct, total = [], 0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(batch)
        targets = batch["wdl_target"].to(logits.device)
        losses.append(float(soft_cross_entropy(logits, targets)))
        correct += int((logits.argmax(-1) == targets.argmax(-1)).sum())
        total += int(targets.size(0))
    mean_loss = sum(losses) / max(1, len(losses))
    accuracy = correct / max(1, total)
    return mean_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()

    cfg = load_repo_config(args.config).value_net
    steps = int(args.steps if args.steps is not None else cfg.train_steps)
    device_arg = args.device or cfg.device
    device = torch.device(
        "cuda" if (device_arg == "auto" and torch.cuda.is_available()) else
        device_arg if device_arg != "auto" else "cpu"
    )
    torch.manual_seed(cfg.seed)

    model = ValueNet(
        ValueNetConfig(dim=cfg.dim, num_heads=cfg.num_heads, num_layers=cfg.num_layers)
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ValueNet params: {num_params / 1e6:.2f}M | device: {device}")

    def make_loader(split: str) -> DataLoader:
        dataset = PositionEvalDataset(
            split=split,
            depth_min=cfg.depth_min,
            dataset_name=cfg.dataset_name,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            seed=cfg.seed,
            val_permille=cfg.val_permille,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers if split == "train" else 0,
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader("train")
    val_loader = make_loader("val")

    try:
        from optimi import StableAdamW

        optimizer = StableAdamW(
            model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay,
            kahan_sum=True,
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, total_steps=steps, pct_start=0.05
    )

    use_amp = device.type == "cuda" and cfg.dtype in {"bfloat16", "float16"}
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=getattr(torch, cfg.dtype))
        if use_amp
        else None
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tb"))
    model_config_payload = {
        "dim": cfg.dim, "num_heads": cfg.num_heads, "num_layers": cfg.num_layers
    }

    def save(name: str, step: int, val_loss: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "config": model_config_payload,
                "step": step,
                "val_loss": val_loss,
            },
            checkpoint_dir / name,
        )

    best_val = float("inf")
    step = 0
    t0 = time.time()
    while step < steps:
        for batch in train_loader:
            loss = train_step(
                model, batch, optimizer,
                grad_clip_norm=cfg.grad_clip_norm, autocast_ctx=autocast_ctx,
            )
            scheduler.step()
            step += 1
            if step % cfg.log_every_steps == 0:
                rate = step / (time.time() - t0)
                print(f"step {step}/{steps} loss {loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} ({rate:.1f} it/s)")
                writer.add_scalar("train/loss", loss, step)
            if step % cfg.val_every_steps == 0 or step == steps:
                val_loss, val_acc = validate(
                    model, val_loader, max_batches=cfg.val_batches, device=device
                )
                print(f"  val loss {val_loss:.4f} acc {val_acc:.3f}")
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/acc", val_acc, step)
                save("value_net_last.pt", step, val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    save("value_net_best.pt", step, val_loss)
            if step >= steps:
                break
    writer.close()
    print(f"done: best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_value_net.py -v`
Expected: PASS (all, including the two new config tests and the trainer smoke).

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_value_net.py src/imba_chess/config.py config/imba_chess.toml tests/test_config.py tests/test_value_net.py
git commit -m "feat: standalone value-net trainer + [value_net] config section"
```

---

### Task 4: Eval integration — blend in `CachedPositionEvaluator`

**Files:**
- Modify: `scripts/eval_vs_stockfish.py`
- Modify: `src/imba_chess/config.py` (+2 fields on `EvalVsStockfishConfig`), `config/imba_chess.toml`
- Test: `tests/test_eval_vs_stockfish.py` (append), `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `ValueNet`, `ValueNetConfig` (Task 1); checkpoint payload format (Task 3).
- Produces: `CachedPositionEvaluator(..., value_net=None, value_net_alpha=1.0)`; `_load_value_net(path, repo_config, device) -> ValueNet`; CLI `--value-net-checkpoint`, `--value-net-alpha`; `run_config["value_net"] = {"checkpoint": str|None, "alpha": float}` in the output JSON.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_eval_vs_stockfish_value_net_defaults():
    config = EvalVsStockfishConfig()
    assert config.value_net_checkpoint is None
    assert config.value_net_alpha == 1.0
```

Append to `tests/test_eval_vs_stockfish.py`:

```python
class _StubValueNet(torch.nn.Module):
    """Constant WDL logits: every position looks equally winning (stm POV)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch):  # type: ignore[no-untyped-def]
        self.calls += 1
        batch_size = int(batch["turn_id"].numel())
        logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        logits[:, 2] = 10.0  # p(win) ~ 1 for the side to move, always
        return logits


def _rerank_move_with_net(value_net, alpha):
    module = _load_eval_script_module()
    move_vocab = _mini_vocab()
    model = _DummyValueRerankModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)
    move, _ = module._select_model_move(
        model=model,
        batch=batch,
        board=board,
        move_vocab=move_vocab,
        board_state_encoder=BoardStateEncoder(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        policy="value_rerank",
        value_rerank_top_k=2,
        value_rerank_lambda=0.2,
        debug_topk=0,
        value_net=value_net,
        value_net_alpha=alpha,
    )
    return move


def test_value_net_alpha_zero_reproduces_model_head():
    net = _StubValueNet()
    # alpha=0: model head decides (d2d4 per the dummy's value scheme), and
    # the net must not even be called.
    move = _rerank_move_with_net(net, alpha=0.0)
    assert move.uci() == "d2d4"
    assert net.calls == 0


def test_value_net_alpha_one_uses_net_values():
    net = _StubValueNet()
    # alpha=1: the constant net makes all candidates equal; the policy-prior
    # tiebreak picks the higher-prior move e2e4 instead of the model head's
    # d2d4 preference.
    move = _rerank_move_with_net(net, alpha=1.0)
    assert move.uci() == "e2e4"
    assert net.calls == 1


def test_value_net_alpha_half_blends():
    module = _load_eval_script_module()
    # Blend math is checked directly on the evaluator with a stub decode
    # model: model scalar -1 (loss), net scalar ~+1 (win) -> ~0 at alpha=.5.
    move_vocab = _mini_vocab()
    model = _DummyValueRerankModel(move_vocab)
    encoder = BoardStateEncoder()
    board = chess.Board()
    history = module._SequenceHistory(move_vocab=move_vocab, board_state_encoder=encoder)
    root_batch = history.build_batch_for_current_position(board)
    prefill = model(root_batch, return_loss=False, return_kv=True)
    net = _StubValueNet()
    evaluator = module.CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefix_kv=prefill["kv_caches"],
        prefix_len=int(root_batch["total_tokens"]),
        value_net=net,
        value_net_alpha=0.5,
    )
    move = chess.Move.from_uci("d2d4")
    handle = evaluator.extend(None, board, move)
    board1 = board.copy()
    board1.push(move)
    (result,) = evaluator.evaluate([(handle, board1)])
    # Dummy model head: d2d4 node -> value_logits [4,0,0] -> scalar ~ -0.96.
    # Stub net: scalar ~ +1. Blend at 0.5 ~ 0.02.
    assert -0.2 < result.value_stm < 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py tests/test_config.py -v -k value_net`
Expected: FAIL — `AttributeError` (config fields), `TypeError` (unexpected kwargs).

- [ ] **Step 3: Config fields**

`src/imba_chess/config.py`, `EvalVsStockfishConfig`, after `search_max_depth`:

```python
    value_net_checkpoint: Optional[str] = None
    value_net_alpha: float = 1.0
```

`config/imba_chess.toml`, after the halving knobs in `[eval_vs_stockfish]`:

```toml
# Standalone value-net blend: value = (1-alpha)*model_head + alpha*net.
# Unset checkpoint = current behavior (model head only).
# value_net_checkpoint = "artifacts/value_net/value_net_best.pt"
value_net_alpha = 1.0
```

- [ ] **Step 4: Wire the eval script**

4a. Import: extend the model imports with

```python
from imba_chess.model.value_net import ValueNet, ValueNetConfig
```

4b. Add loader after `_load_model`:

```python
def _load_value_net(path: Path, repo_config, device: torch.device) -> ValueNet:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    saved = payload.get("config", {}) if isinstance(payload, dict) else {}
    vn_cfg = repo_config.value_net
    net = ValueNet(
        ValueNetConfig(
            dim=int(saved.get("dim", vn_cfg.dim)),
            num_heads=int(saved.get("num_heads", vn_cfg.num_heads)),
            num_layers=int(saved.get("num_layers", vn_cfg.num_layers)),
        )
    ).to(device)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net
```

4c. `CachedPositionEvaluator.__init__` gains `value_net=None, value_net_alpha: float = 1.0` (stored as `self._value_net`, `self._value_net_alpha`). In `evaluate`, inside the existing `with torch.inference_mode(), autocast_ctx:` block, after the logits/value offload lines, add:

```python
            net_scalars = None
            if self._value_net is not None and self._value_net_alpha > 0.0:
                net_logits = self._value_net(new_token_batch).float().cpu()
                net_probs = torch.softmax(net_logits, dim=-1)
                net_scalars = net_probs[:, 2] - net_probs[:, 0]
```

and in the results loop replace

```python
            value_stm = _value_scalar_from_logits(value_logits[row])
```

with

```python
            value_stm = _value_scalar_from_logits(value_logits[row])
            if net_scalars is not None:
                alpha = self._value_net_alpha
                value_stm = (1.0 - alpha) * value_stm + alpha * float(net_scalars[row])
```

4d. `_select_model_move` gains `value_net=None, value_net_alpha: float = 1.0` (after `halving_config`), passed into the `CachedPositionEvaluator(...)` construction. `_run_segment` gains and forwards the same two parameters into its `_select_model_move(...)` call.

4e. CLI: after the `--search-max-depth` argument:

```python
    parser.add_argument("--value-net-checkpoint", type=Path, default=None)
    parser.add_argument("--value-net-alpha", type=float, default=None)
```

Resolution in `main()` after the `args.search_max_depth` block:

```python
    args.value_net_checkpoint = (
        eval_cfg.value_net_checkpoint
        if args.value_net_checkpoint is None
        else args.value_net_checkpoint
    )
    args.value_net_alpha = float(
        eval_cfg.value_net_alpha
        if args.value_net_alpha is None
        else args.value_net_alpha
    )
```

Validation block addition:

```python
    if not 0.0 <= args.value_net_alpha <= 1.0:
        raise ValueError("--value-net-alpha must be in [0, 1]")
```

Loading in `main()` after `_load_model`:

```python
    value_net = (
        _load_value_net(Path(args.value_net_checkpoint), repo_config, device)
        if args.value_net_checkpoint
        else None
    )
    if value_net is not None:
        print(f"  value_net={args.value_net_checkpoint} alpha={args.value_net_alpha}")
```

Call-site: pass `value_net=value_net, value_net_alpha=float(args.value_net_alpha),` into `_run_segment(...)`. In `_summary_to_payload`'s `run_config`, next to `"search"`, add:

```python
        "value_net": {
            "checkpoint": str(value_net_checkpoint) if value_net_checkpoint else None,
            "alpha": value_net_alpha,
        },
```

threading `value_net_checkpoint`/`value_net_alpha` as keyword-only params from both call sites (mirror how `search_knobs` is passed).

- [ ] **Step 5: Run the new tests, then the full eval suite**

Run: `.venv/bin/python -m pytest tests/test_eval_vs_stockfish.py tests/test_config.py -v`
Expected: PASS — including every pre-existing test unchanged (the no-net default path must be byte-identical).

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_vs_stockfish.py src/imba_chess/config.py config/imba_chess.toml tests/test_eval_vs_stockfish.py tests/test_config.py
git commit -m "feat: optional value-net blend in search evaluation (value_net_alpha)"
```

---

## Post-implementation (manual, not part of the plan)

Train on the GPU box (`python scripts/train_value_net.py`, watch val loss),
then acceptance: `POLICIES="value_search_halving" ELO=1800 TAG=vnet
./eval_best_checkpoint.sh --search-budget 1024 --search-max-depth 6
--value-net-checkpoint artifacts/value_net/value_net_best.pt` vs the 0.595
baseline; one follow-up at `--value-net-alpha 0.5`. README results update
follows the numbers.
