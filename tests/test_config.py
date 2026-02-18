import pytest

from imba_chess.config import load_repo_config


def test_load_repo_config_reads_sections(tmp_path):
    config_path = tmp_path / "imba_chess.toml"
    config_path.write_text(
        """
[dataset]
min_avg_elo = 2200
train_start_month = "2018-01"
train_end_month = "2025-07"
val_start_month = "2025-08"
val_end_month = "2025-08"
test_start_month = "2025-09"
test_end_month = "2025-09"
val_max_games = 100000
test_max_games = 100000
max_seq_len = 256

[vocab]
path = "tmp_vocab.json"

[dataloader]
max_tokens_per_batch = 4096

[model]
num_layers = 8

[training]
epochs = 5
steps_per_epoch = 1234
eval_every_steps = 200
max_lr = 0.0003
lr_start_factor = 0.2
lr_end_factor = 0.5
onecycle_pct_start = 0.3
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_repo_config(config_path)
    assert config.dataset.min_avg_elo == 2200
    assert config.dataset.train_start_month == "2018-01"
    assert config.dataset.train_end_month == "2025-07"
    assert config.dataset.val_start_month == "2025-08"
    assert config.dataset.val_end_month == "2025-08"
    assert config.dataset.test_start_month == "2025-09"
    assert config.dataset.test_end_month == "2025-09"
    assert config.dataset.val_max_games == 100000
    assert config.dataset.test_max_games == 100000
    assert config.dataset.max_seq_len == 256
    assert config.vocab.path == "tmp_vocab.json"
    assert config.dataloader.max_tokens_per_batch == 4096
    assert config.model.num_layers == 8
    assert config.training.epochs == 5
    assert config.training.steps_per_epoch == 1234
    assert config.training.eval_every_steps == 200
    assert config.training.max_lr == 0.0003
    assert config.training.lr_start_factor == 0.2
    assert config.training.lr_end_factor == 0.5
    assert config.training.onecycle_pct_start == 0.3


def test_load_repo_config_unknown_key_raises(tmp_path):
    config_path = tmp_path / "imba_chess.toml"
    config_path.write_text(
        """
[dataset]
bad_key = 1
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown keys in \[dataset\]"):
        load_repo_config(config_path)
