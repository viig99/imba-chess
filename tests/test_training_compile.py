"""CUDA integration: fused attention must preserve the actual learning update."""

from dataclasses import replace

import pytest
import torch

from imba_chess.model import HSTUChessModel
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.dataset import collate_self_play, reconstruct
from imba_chess.self_play.trainer import Stage2Trainer
from tests.test_self_play import ENCODER, VOCAB, mate_game, tiny_model


@pytest.mark.extended
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward needs CUDA")
@pytest.mark.parametrize("auxiliary", [False, True])
def test_compiled_learning_matches_eager_across_batch_shapes(auxiliary):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = replace(tiny_model().config, model_dim=64, attention_dim=16,
                     linear_hidden_dim=16, num_heads=4, num_layers=2, dropout=0.0,
                     enable_auxiliary_value_head=auxiliary, auxiliary_value_heads=3)
    eager = HSTUChessModel(config).cuda().train()
    compiled = HSTUChessModel(config).cuda().train()
    compiled.load_state_dict(eager.state_dict())

    learning = LearningConfig(
        value_weight=.25 if auxiliary else 1.,
        value_search_mix=.25 if auxiliary else 0.,
        auxiliary_value_weight=.25 if auxiliary else 0.,
        auxiliary_value_lambda=(.8, .9, .95),
    )

    def trainer(model, **kwargs):
        return Stage2Trainer(model=model, config=learning, move_vocab=VOCAB,
                             encoder=ENCODER, device=torch.device("cuda"), max_positions=128,
                             **kwargs)

    reference, candidate = trainer(eager, compile_model=False), trainer(compiled)
    assert candidate.compile_enabled and not reference.compile_enabled
    game = mate_game()
    game["targets"][0]["policy_training_weight"] = 0.
    for target in game["targets"]:
        target["search_wdl"] = [.2, .3, .5]
    sample = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128,
                         learning=learning)
    for count in (1, 3, 2):
        batch = collate_self_play([sample] * count)
        results = []
        for t in (reference, candidate):
            t.optimizer.zero_grad(set_to_none=True)
            losses = t._loss_fn(t.model, batch, t.config.value_weight, t.config.auxiliary_value_weight)
            losses["loss"].backward()
            results.append(losses)
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], atol=2e-5, rtol=2e-4)
        for (name, a), (_, b) in zip(eager.named_parameters(), compiled.named_parameters()):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4, msg=name)
        assert eager.layers[0]._ps_w.grad is not None
        for t in (reference, candidate):
            t._clip_gradients()
            t.optimizer.step()
        for a, b in zip(eager.parameters(), compiled.parameters()):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    assert candidate.model is compiled
    assert candidate.model.state_dict().keys() == eager.state_dict().keys()
