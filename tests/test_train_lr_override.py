import importlib.util
from pathlib import Path

import pytest
import torch

from imba_chess.config import DatasetConfig, ModelConfig, RepoConfig

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("_train_for_lr_test", _TRAIN_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def _make_optimizer(lr: float = 7e-4) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD([param], lr=lr)


def test_constant_lr_scheduler_pins_lr_across_steps():
    optimizer = _make_optimizer()
    scheduler = train._build_constant_lr_scheduler(optimizer, 5e-5)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    for _ in range(50):
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_constant_lr_scheduler_sets_initial_lr():
    # LambdaLR reads base_lrs from "initial_lr"; if only "lr" were set, the
    # first step() would snap the rate back to the pre-override value.
    optimizer = _make_optimizer(lr=7e-4)
    train._build_constant_lr_scheduler(optimizer, 1e-5)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(1e-5)


def test_override_beats_a_restored_onecycle_state():
    """The real scenario: --resume restores OneCycleLR state, then we override.

    Reproduces the ordering in main() -- build OneCycle, load its state back
    (as Checkpoint.load_objects does), then apply the override -- and asserts
    the override wins and stays won.
    """
    optimizer = _make_optimizer(lr=7e-4)
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=7e-4, total_steps=1_000_000, pct_start=0.1
    )
    for _ in range(230_000):
        onecycle.step()
    resumed_state = onecycle.state_dict()

    # Simulate resume: scheduler state comes back from the checkpoint.
    onecycle.load_state_dict(resumed_state)
    resumed_lr = optimizer.param_groups[0]["lr"]
    assert resumed_lr > 1e-4, "sanity: resumed schedule lr should still be high"

    scheduler = train._build_constant_lr_scheduler(optimizer, 5e-5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    for _ in range(100):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


@pytest.mark.parametrize("bad_lr", [0.0, -1e-5])
def test_non_positive_override_rejected(bad_lr):
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="must be > 0"):
        train._build_constant_lr_scheduler(optimizer, bad_lr)


def test_lr_override_flag_defaults_to_none_and_parses(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--config", "c.toml"])
    assert train.parse_args().lr_override is None

    monkeypatch.setattr(
        "sys.argv", ["train.py", "--config", "c.toml", "--lr-override", "5e-5"]
    )
    assert train.parse_args().lr_override == pytest.approx(5e-5)


def test_run_limit_flags_parse(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "c.toml",
            "--max-steps",
            "34000",
            "--max-games",
            "2000000",
        ],
    )
    args = train.parse_args()
    assert args.max_steps == 34_000
    assert args.max_games == 2_000_000


def test_make_dataset_parses_evals_on_all_splits_when_value_head_enabled():
    config = RepoConfig(model=ModelConfig(enable_value_head=True))
    train_dataset = train._make_dataset(config, split="train")
    val_dataset = train._make_dataset(config, split="val")
    assert train_dataset.parse_stockfish_evals is True
    assert val_dataset.parse_stockfish_evals is True

    no_value_head = RepoConfig()
    assert train._make_dataset(no_value_head, split="val").parse_stockfish_evals is False


def test_make_dataset_routes_distinct_local_train_and_validation_corpora():
    config = RepoConfig(
        dataset=DatasetConfig(
            local_corpus_path="artifacts/corpus/train.parquet",
            val_local_corpus_path="artifacts/corpus/val.parquet",
        )
    )

    assert (
        train._make_dataset(config, split="train").local_corpus_path
        == "artifacts/corpus/train.parquet"
    )
    assert (
        train._make_dataset(config, split="val").local_corpus_path
        == "artifacts/corpus/val.parquet"
    )
    assert train._make_dataset(config, split="test").local_corpus_path is None
