"""Evaluation-only config edits preserve existing training resume identities."""

import hashlib
from pathlib import Path
import pytest
from imba_chess.self_play.config import _base_config_identity, _BASE_CONFIG_IDENTITIES


@pytest.mark.parametrize("path", sorted(Path("config").glob("imba_chess*.toml")))
def test_shipped_config_identity_is_preserved_and_changes_are_rejected(path, tmp_path):
    data = path.read_bytes()
    current = hashlib.sha256(data).hexdigest()
    assert current in _BASE_CONFIG_IDENTITIES
    assert _base_config_identity(path) == _BASE_CONFIG_IDENTITIES[current]
    changed = tmp_path / path.name
    changed.write_bytes(data + b"\n# user modification\n")
    assert _base_config_identity(changed) != _base_config_identity(path)
