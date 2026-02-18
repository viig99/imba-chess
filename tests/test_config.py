import pytest

from imba_chess.config import load_repo_config


def test_load_repo_config_reads_sections(tmp_path):
    config_path = tmp_path / "imba_chess.toml"
    config_path.write_text(
        """
[dataset]
min_avg_elo = 2200
max_seq_len = 256

[vocab]
path = "tmp_vocab.json"

[dataloader]
max_tokens_per_batch = 4096

[model]
num_layers = 8
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_repo_config(config_path)
    assert config.dataset.min_avg_elo == 2200
    assert config.dataset.max_seq_len == 256
    assert config.vocab.path == "tmp_vocab.json"
    assert config.dataloader.max_tokens_per_batch == 4096
    assert config.model.num_layers == 8


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
