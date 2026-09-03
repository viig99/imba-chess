from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from imba_chess.data.stockfish_evals import CpToWdlCalibration, eval_from_comment


def _calibration_payload() -> dict:
    return {
        "cp_bins": [
            {"cp_center": -200.0, "n": 10, "p_loss": 0.7, "p_draw": 0.2, "p_win": 0.1},
            {"cp_center": -20.0, "n": 10, "p_loss": 0.4, "p_draw": 0.3, "p_win": 0.3},
            {"cp_center": 50.0, "n": 10, "p_loss": 0.2, "p_draw": 0.3, "p_win": 0.5},
            {"cp_center": 300.0, "n": 10, "p_loss": 0.1, "p_draw": 0.1, "p_win": 0.8},
        ],
        "mate": {
            "-1": {"n": 10, "p_loss": 0.9, "p_draw": 0.05, "p_win": 0.05},
            "1": {"n": 10, "p_loss": 0.04, "p_draw": 0.06, "p_win": 0.9},
        },
        "source": "synthetic",
    }


def _write_calibration(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload or _calibration_payload()), encoding="utf-8")
    return path


def _load_vectorised_calibrator():
    path = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_evals_to_wdl.py"
    spec = importlib.util.spec_from_file_location("calibrate_evals_to_wdl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_calibration


def test_scalar_calibration_matches_vectorised_reference(tmp_path):
    payload = _calibration_payload()
    calibration = CpToWdlCalibration.load(_write_calibration(tmp_path, payload))
    cases = [
        (-500.0, None),
        (-200.0, None),
        (-75.0, None),
        (-20.0, None),
        (15.0, None),
        (300.0, None),
        (800.0, None),
        (None, -4),
        (None, 2),
    ]
    cp = np.array([np.nan if c is None else c for c, _ in cases])
    mate = np.array([np.nan if m is None else m for _, m in cases])
    expected = _load_vectorised_calibrator()(payload, cp, mate)
    actual = np.array([calibration.wdl(c, m) for c, m in cases])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_exact_zero_bucket_overrides_interpolation(tmp_path):
    payload = _calibration_payload()
    payload["cp_zero"] = {"n": 10, "p_loss": 0.3, "p_draw": 0.6, "p_win": 0.1}
    calibration = CpToWdlCalibration.load(_write_calibration(tmp_path, payload))

    assert calibration.wdl(0.0, None) == pytest.approx((0.3, 0.6, 0.1))
    # Neighbours still interpolate the continuous bins, untouched by the
    # point mass.
    without_zero = CpToWdlCalibration.load(
        _write_calibration(tmp_path, _calibration_payload())
    )
    assert calibration.wdl(1.0, None) == pytest.approx(without_zero.wdl(1.0, None))
    assert calibration.wdl(-1.0, None) == pytest.approx(without_zero.wdl(-1.0, None))
    # Without the bucket, 0.0 interpolates as before (old JSONs still load).
    assert without_zero.zero_probs is None
    assert without_zero.wdl(0.0, None) != pytest.approx((0.3, 0.6, 0.1))

    cp = np.array([0.0, 1.0])
    mate = np.array([np.nan, np.nan])
    vectorised = _load_vectorised_calibrator()(payload, cp, mate)
    np.testing.assert_allclose(vectorised[0], (0.3, 0.6, 0.1), atol=1e-12)
    np.testing.assert_allclose(vectorised[1], calibration.wdl(1.0, None), atol=1e-12)


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("[%eval 0.14]", (14.0, None)),
        ("[%eval #3]", (None, 3)),
        ("[%eval #-2]", (None, -2)),
        ("[%clk 0:01:00]", (None, None)),
        ("[%eval nope]", (None, None)),
        ("ordinary comment", (None, None)),
    ],
)
def test_eval_from_comment(comment, expected):
    actual = eval_from_comment(comment)
    if expected[0] is not None:
        assert actual[0] == pytest.approx(expected[0])
        assert actual[1] is None
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("cp_bins"),
        lambda payload: payload.update(cp_bins=payload["cp_bins"][:1]),
        lambda payload: payload["mate"].pop("1"),
        lambda payload: payload["cp_bins"].__setitem__(
            1, {**payload["cp_bins"][1], "cp_center": -200.0}
        ),
        lambda payload: payload["cp_bins"][0].update(p_loss=-0.1),
        lambda payload: payload["cp_bins"][0].update(p_draw=float("nan")),
    ],
)
def test_load_rejects_malformed_calibration(tmp_path, mutate):
    payload = _calibration_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        CpToWdlCalibration.load(_write_calibration(tmp_path, payload))


def test_wdl_rejects_invalid_inputs(tmp_path):
    calibration = CpToWdlCalibration.load(_write_calibration(tmp_path))
    with pytest.raises(ValueError, match="Exactly one"):
        calibration.wdl(None, None)
    with pytest.raises(ValueError, match="Exactly one"):
        calibration.wdl(10.0, 2)
    with pytest.raises(ValueError, match="non-zero"):
        calibration.wdl(None, 0)
    with pytest.raises(ValueError, match="finite"):
        calibration.wdl(float("nan"), None)
