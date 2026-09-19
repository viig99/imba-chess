import hashlib
import json
import math
from dataclasses import asdict, replace

import pytest
import torch

from imba_chess.self_play.config import LearningConfig, SelfPlayConfig, _base_config_identity
from imba_chess.self_play.dataset import policy_weights, reconstruct, collate_self_play
from imba_chess.self_play.losses import self_play_loss
from tests.test_self_play import mate_game, VOCAB, ENCODER, tiny_model
from imba_chess.self_play.trainer import Stage2Trainer
from imba_chess.data.self_play_store import SelfPlayStore, validate_game

ON = LearningConfig(policy_surprise_enabled=True)


def target(d, base=1):
    if d == 0:
        return dict(policy=[.5, .5], root_log_priors=[math.log(.5)] * 2, policy_training_weight=base)
    return dict(policy=[1., 0.], root_log_priors=[-d, math.log1p(-math.exp(-d))], policy_training_weight=base)


def test_hand_weights_and_edge_cases():
    result = policy_weights([target(0), target(2)], learning=ON)
    assert result['policy_surprise'] == [0, 2]
    assert result['policy_training_weight'] == pytest.approx([.5, 1.5])
    for ds in ([0, 0], [2, 2]):
        assert policy_weights([target(d) for d in ds], learning=ON)['policy_training_weight'] == [1, 1]
    ts = [target(0), target(2), target(100, 0), dict(policy=[1.])]
    assert policy_weights(ts, learning=ON)['policy_training_weight'] == pytest.approx([.5, 1.5, 0, 1])
    # Cap is before normalization: final weights can exceed it.
    result = policy_weights([target(0)] * 19 + [target(100)], learning=ON)
    assert result['policy_surprise_weight'][-1] > 3
    assert sum(result['policy_surprise_clipped']) == 1
    ts = [dict(policy=[.25, .75], root_log_priors=[math.log(.5)] * 2)]
    assert policy_weights(ts)['policy_surprise'][0] == pytest.approx(.25 * math.log(.5) + .75 * math.log(1.5))


def sample(game, learning=None):
    return reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=learning)


def test_game_normalization_and_legacy_replay(tmp_path):
    g = mate_game()
    for i, t in enumerate(g['targets']):
        n = len(t['policy'])
        t['policy'] = [1.] + [0.] * (n - 1)
        t['root_log_priors'] = [-(i + 1.)] + [math.log1p(-math.exp(-(i + 1.))) - math.log(n - 1)] * (n - 1)
    a = sample(g, ON)
    b = sample(mate_game('other', prefix=['f2f3', 'e7e5']), ON)
    one, two = collate_self_play([a]), collate_self_play([b, a])
    assert torch.equal(one['policy_training_weight'], two['policy_training_weight'][-4:])
    assert one['policy_weight_metrics']['eligible_surprise_mean'] == pytest.approx(2.5)
    for t in g['targets']:
        t.pop('root_log_priors')
    store = SelfPlayStore(tmp_path, flush_games=1)
    store.add(g)
    assert sample(store.read_game('g'), ON)['policy_training_weight'] == [1.] * 4


def test_loss_gradient_parity_zero_weights_padding_and_detach():
    batch = collate_self_play([sample(mate_game())])
    n = batch['total_tokens']
    torch.manual_seed(2)
    logits = torch.randn(n, len(VOCAB), requires_grad=True)
    values = torch.randn(n, 3, requires_grad=True)
    out = dict(logits=logits, value_logits=values)
    original = dict(batch)
    original.pop('policy_training_weight')
    old = self_play_loss(out, original)
    new = self_play_loss(out, batch)
    assert torch.equal(old['loss'], new['loss'])
    old_grads = torch.autograd.grad(old['loss'], (logits, values), retain_graph=True)
    new_grads = torch.autograd.grad(new['loss'], (logits, values), retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(old_grads, new_grads))
    weights = torch.tensor([0., .5, 1.5, 2.], requires_grad=True)
    batch['policy_training_weight'] = weights
    weighted = self_play_loss(out, batch)
    indices = batch['supervised_indices']
    logs = logits[indices].gather(1, batch['legal_ids']).masked_fill(~batch['legal_mask'], -torch.inf).log_softmax(-1).masked_fill(~batch['legal_mask'], 0)
    ce = -(batch['policy'] * logs).sum(-1)
    assert weighted['weighted_policy_loss'].item() == pytest.approx(((weights * ce).sum() / weights.sum()).item())
    grads = torch.autograd.grad(weighted['loss'], (logits, values, weights), allow_unused=True, retain_graph=True)
    assert torch.equal(grads[1], old_grads[1]) and grads[2] is None
    assert grads[0][indices[0]].count_nonzero() == 0
    batch['policy_training_weight'] = torch.zeros(4)
    zero = self_play_loss(out, batch)
    assert zero['weighted_policy_loss'] == 0
    assert torch.autograd.grad(zero['loss'], logits)[0].count_nonzero() == 0


def test_old_identity_and_validation():
    cfg = SelfPlayConfig()
    old = asdict(cfg)
    old.pop('streaming')
    for k in list(old['learning']):
        if k.startswith('policy_surprise_'):
            old['learning'].pop(k)
    expected = hashlib.sha256(json.dumps(dict(settings=old, base_sha256=_base_config_identity(cfg.base_config)), sort_keys=True).encode()).hexdigest()
    assert cfg.identifier == expected
    assert replace(cfg, learning=ON).identifier != expected
    for kwargs in (dict(policy_surprise_fraction=-.1), dict(policy_surprise_fraction=float('nan')), dict(policy_surprise_cap=.5), dict(policy_surprise_enabled=1)):
        with pytest.raises(ValueError):
            LearningConfig(**kwargs)
    g = mate_game()
    g['targets'][0]['policy_training_weight'] = -1
    with pytest.raises(ValueError):
        validate_game(g)


def test_legacy_checkpoint_and_incompatible_resume(tmp_path):
    store = SelfPlayStore(tmp_path / 'replay', flush_games=1)
    store.add(mate_game())
    def trainer(config):
        return Stage2Trainer(model=tiny_model(), config=config, move_vocab=VOCAB,
                             encoder=ENCODER, device=torch.device('cpu'), max_positions=128)
    a = trainer(LearningConfig())
    a.begin_phase(store)
    path = tmp_path / 'old.pt'
    a.checkpoint(path, progress={}, store=store, config_id='same')
    state = torch.load(path, weights_only=False)
    for k in list(state['learning_config']):
        if k.startswith('policy_surprise_'):
            state['learning_config'].pop(k)
    torch.save(state, path)
    trainer(LearningConfig()).resume(path, store=store, config_id='same')
    for cfg in (ON, LearningConfig(policy_surprise_cap=4), LearningConfig(policy_surprise_fraction=.2)):
        with pytest.raises(ValueError, match='configuration changed'):
            trainer(cfg).resume(path, store=store, config_id='same')
