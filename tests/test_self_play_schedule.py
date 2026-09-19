import copy

import pytest
import torch

from imba_chess.data.self_play_store import SelfPlayStore
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.trainer import Stage2Trainer


def trainer(model, lr):
    return Stage2Trainer(model=model, config=LearningConfig(lr=lr), move_vocab=None,
                         encoder=None, device=torch.device('cpu'), max_positions=128)


def update(t):
    t.optimizer.zero_grad()
    t.model(torch.ones(2, 3)).square().mean().backward()
    t.optimizer.step()
    t.scheduler.step()


def test_supervised_schedule_and_optimizer_continue_exactly(tmp_path):
    torch.set_num_threads(1)
    source = trainer(torch.nn.Linear(3, 2), 0.0005)
    source.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        source.optimizer, max_lr=0.0005, total_steps=100, pct_start=0.1,
        anneal_strategy='linear', cycle_momentum=False,
        div_factor=1, final_div_factor=2)
    for _ in range(34):
        update(source)
    path = tmp_path / 'supervised.pt'
    torch.save(dict(model=source.model.state_dict(), optimizer=source.optimizer.state_dict(),
                    scheduler=source.scheduler.state_dict()), path)
    lr = source.optimizer.param_groups[0]['lr']
    target = trainer(copy.deepcopy(source.model), lr)
    target.initialize_optimization(path)
    assert target.steps == target.exposures == 0
    assert target.scheduler.last_epoch == 34
    for _ in range(4):
        update(source)
        update(target)
        assert source.scheduler.get_last_lr() == target.scheduler.get_last_lr()
        for a, b in zip(source.model.parameters(), target.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    store = SelfPlayStore(tmp_path / 'replay')
    target.checkpoint(tmp_path / 'resume.pt', progress={'phase': 'collect'},
                      store=store, config_id='test')
    restored = trainer(torch.nn.Linear(3, 2), lr)
    restored.resume(tmp_path / 'resume.pt', store=store, config_id='test')
    update(target)
    update(restored)
    assert target.scheduler.get_last_lr() == restored.scheduler.get_last_lr()
    for a, b in zip(target.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    wrong = trainer(copy.deepcopy(source.model), lr)
    with pytest.raises(ValueError, match='match initialized model'):
        wrong.initialize_optimization(path)
