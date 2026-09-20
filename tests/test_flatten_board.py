import pytest
import torch

from imba_chess.config import load_repo_config
from imba_chess.model import HSTUChessConfig, HSTUChessModel, build_hstu_chess_config
from imba_chess.model.checkpoint import load_initial_weights, normalized_model_state
from test_hstu_model import _batch


@pytest.mark.parametrize("prefix", ["", "_orig_mod.", "module.", "module._orig_mod."])
def test_native_checkpoint_loading_is_strict(prefix):
    cfg = HSTUChessConfig(move_vocab_size=128, model_dim=64, num_layers=0)
    original = HSTUChessModel(cfg)
    loaded = HSTUChessModel(cfg)
    checkpoint = {"model": {prefix + k: v for k, v in original.state_dict().items()}}
    load_initial_weights(loaded, checkpoint)
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    load_initial_weights(loaded, original.state_dict())
    incompatible = dict(original.state_dict())
    incompatible["board_encoder.out_proj.weight"] = torch.zeros(64, 64)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_initial_weights(loaded, incompatible)


def test_checkpoint_key_collisions_are_rejected():
    with pytest.raises(ValueError, match="colliding"):
        normalized_model_state({"x": torch.zeros(1), "_orig_mod.x": torch.zeros(1)})


def test_square_readout_learns_distinct_weights():
    torch.manual_seed(42)
    cfg = HSTUChessConfig(move_vocab_size=128, model_dim=64, num_layers=0, dropout=0)
    model = HSTUChessModel(cfg)
    batch = _batch()
    batch["piece_ids"] = torch.randint(0, 13, (5, 64))
    model(batch)["loss"].backward()
    grad = model.board_encoder.out_proj.weight.grad.reshape(64, 64, 64)
    assert torch.isfinite(grad).all()
    assert not torch.allclose(grad[:, 0], grad[:, 1])


def test_flatten_is_the_only_default():
    config = load_repo_config('config/imba_chess_v4_laptop.toml')
    cfg = build_hstu_chess_config(config.model, move_vocab_size=1970)
    model = HSTUChessModel(cfg)
    assert model.board_encoder.out_proj.in_features == 4096
    assert not hasattr(cfg, 'board_pooling')
    assert sum(p.numel() for p in model.parameters()) == 52388996


def test_weights_only_cli_is_explicit(monkeypatch):
    from scripts import train
    monkeypatch.setattr('sys.argv', ['train.py', '--init-weights', 'ckpt34.pt'])
    assert str(train.parse_args().init_weights) == 'ckpt34.pt'
    monkeypatch.setattr('sys.argv', ['train.py', '--init-weights', 'ckpt34.pt', '--resume', 'old.pt'])
    with pytest.raises(ValueError, match='cannot be combined'):
        train.main()
