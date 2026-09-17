"""CUDA integration: fused attention must preserve the actual learning update."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from imba_chess.model import HSTUChessModel
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.dataset import collate_self_play, reconstruct
from imba_chess.self_play.trainer import Stage2Trainer, _model_loss, _training_loss
from tests.test_self_play import ENCODER, VOCAB, mate_game, tiny_model


@pytest.mark.extended
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward needs CUDA")
def test_compiled_learning_matches_eager_across_batch_shapes():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = replace(tiny_model().config, model_dim=64, attention_dim=16,
                     linear_hidden_dim=16, num_heads=4, num_layers=2, dropout=0.0)
    eager = HSTUChessModel(config).cuda().train()
    compiled = HSTUChessModel(config).cuda().train()
    compiled.load_state_dict(eager.state_dict())

    def trainer(model):
        return Stage2Trainer(model=model, config=LearningConfig(), move_vocab=VOCAB,
                             encoder=ENCODER, device=torch.device("cuda"), max_positions=128)

    reference, candidate = trainer(eager), trainer(compiled)
    candidate._loss_fn = partial(_training_loss, model_loss=torch.compile(
        _model_loss, fullgraph=True, dynamic=True))
    sample = reconstruct(mate_game(), move_vocab=VOCAB, encoder=ENCODER, max_positions=128)
    for count in (1, 3, 2):
        batch = collate_self_play([sample] * count)
        results = []
        for t in (reference, candidate):
            t.optimizer.zero_grad(set_to_none=True)
            losses = t._loss_fn(t.model, batch, t.config.value_weight)
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
            torch.nn.utils.clip_grad_norm_(t.model.parameters(), t.config.grad_clip,
                                           error_if_nonfinite=True)
            t.optimizer.step()
        for a, b in zip(eager.parameters(), compiled.parameters()):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    assert candidate.model is compiled
    assert candidate.model.state_dict().keys() == eager.state_dict().keys()
