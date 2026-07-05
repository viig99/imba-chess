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
