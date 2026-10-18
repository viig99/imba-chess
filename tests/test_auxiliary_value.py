"""Search WDL is measurement-only; auxiliary learning shares main value features."""
from dataclasses import asdict, replace
import math

import chess
import pytest
import torch

from imba_chess.data.self_play_store import SelfPlayStore, validate_game
from imba_chess.eval.gumbel_search import GumbelConfig, select_gumbel
from imba_chess.model import HSTUChessModel, create_batch_dense_mask
from imba_chess.self_play.config import LearningConfig
from imba_chess.self_play.dataset import reconstruct, collate_self_play, smoothed_search_wdl
from imba_chess.self_play.losses import self_play_loss
from imba_chess.self_play.trainer import Stage2Trainer
from tests.test_gumbel_search import FakeEvaluator
from tests.test_self_play import tiny_model, mate_game, VOCAB, ENCODER


class WDLEvaluator(FakeEvaluator):
    def evaluate(self, batch):
        return [ev._replace(wdl=(0.1, 0.3, 0.6)) for ev in super().evaluate(batch)]


@pytest.mark.parametrize("prefix", [[], ["e2e4"]])
@pytest.mark.parametrize("depth", [1, 2, 5])
def test_wdl_backup_preserves_search_and_matches_scalar(prefix, depth):
    board = chess.Board()
    for move in prefix:
        board.push_uci(move)
    args = dict(board=board, noise=[0.] * board.legal_moves.count(),
                config=GumbelConfig(simulations=17, top_m=3, max_depth=depth))
    plain = select_gumbel(evaluator=FakeEvaluator(.5), **args)
    measured = select_gumbel(evaluator=WDLEvaluator(.5), **args)
    wdl_fields = ("search_wdl", "improved_wdl")
    assert {k: v for k, v in asdict(measured).items() if k not in wdl_fields} == {
        k: v for k, v in asdict(plain).items() if k not in wdl_fields}
    assert plain.improved_wdl is None
    loss, draw, win = measured.search_wdl
    assert sum(measured.search_wdl) == pytest.approx(1)
    assert draw == pytest.approx(.3)
    assert win - loss == pytest.approx(sum(n*q for n, q in zip(measured.visits, measured.qvalues))/17)
    if depth == 1:
        assert measured.search_wdl == pytest.approx((.6, .3, .1))


@pytest.mark.parametrize("uci, expected", [("f7g7", (0, 0, 1)), ("f7e6", (0, 1, 0))])
def test_terminal_backups_use_rules(uci, expected):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    from imba_chess.eval.cozy_bridge import board_to_cozy
    evaluator = WDLEvaluator(.5)
    root = evaluator.evaluate([(None, board_to_cozy(board))])[0]
    result = select_gumbel(evaluator=evaluator, board=board, root_eval=root,
                          noise=[100 if m == uci else 0 for m in root.legal_ucis],
                          config=GumbelConfig(simulations=8, top_m=1))
    assert result.terminal_hits == 8
    assert result.search_wdl == expected


def game_with_wdl(gid="g", prefix=None):
    game = mate_game(gid, prefix=prefix)
    for target in game["targets"]:
        target["search_wdl"] = [.2, .3, .5]
    return game


def test_smoothing_terminal_anchor_both_colors_and_packed_alignment():
    game = game_with_wdl(prefix=["f2f3", "e7e5"])
    # Last position is Black to move, who wins. Earlier position is White.
    targets = smoothed_search_wdl(game, .5)
    assert targets[1] == pytest.approx([.1, .15, .75])
    assert targets[0] == pytest.approx([.475, .225, .3])
    assert smoothed_search_wdl(game, 0) == [[.2, .3, .5]] * 2
    assert smoothed_search_wdl(game, 1) == [[1., 0, 0], [0, 0, 1.]]
    sample = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128,
                         learning=LearningConfig(auxiliary_value_weight=.25, auxiliary_value_lambda=.5))
    batch = collate_self_play([sample, sample])
    assert batch["supervised_indices"].tolist() == [3, 4, 8, 9]
    torch.testing.assert_close(batch["auxiliary_value_target"], torch.tensor(targets * 2))


def test_missing_or_invalid_search_targets_rejected():
    with pytest.raises(ValueError, match="requires recorded"):
        smoothed_search_wdl(mate_game(), .95)
    game = game_with_wdl()
    game["targets"][0]["search_wdl"] = [.2, .3, .8]
    with pytest.raises(ValueError, match="normalized"):
        validate_game(game)
    game = game_with_wdl()
    game["status"] = "unfinished"
    with pytest.raises(ValueError, match="completed"):
        smoothed_search_wdl(game, .95)


def auxiliary_model():
    return HSTUChessModel(replace(tiny_model().config, enable_auxiliary_value_head=True,
                                  value_head_blocks=2, dropout=0.))


def test_auxiliary_loss_shares_features_but_search_output_unchanged():
    torch.set_num_threads(1)
    model = auxiliary_model()
    sample = reconstruct(game_with_wdl(), move_vocab=VOCAB, encoder=ENCODER, max_positions=128,
                         learning=LearningConfig(auxiliary_value_weight=.25))
    batch = collate_self_play([sample])
    mask = create_batch_dense_mask(batch["seq_offsets"], total_tokens=batch["total_tokens"], device="cpu")
    model.eval()
    initial = model(batch, block_mask=mask, return_loss=False)
    with torch.no_grad():
        model.auxiliary_value_head.weight.normal_(0, .1)
        model.auxiliary_value_head.bias.fill_(100)
    changed = model(batch, block_mask=mask, return_loss=False)
    assert "auxiliary_value_logits" not in changed
    for key in ("logits", "value_logits"):
        torch.testing.assert_close(initial[key], changed[key], rtol=0, atol=0)
    model.train()
    output = model(batch, block_mask=mask, return_loss=False)
    torch.testing.assert_close(output["value_logits"], initial["value_logits"], rtol=0, atol=0)
    losses = self_play_loss(output, batch, auxiliary_value_weight=.25)
    torch.testing.assert_close(losses["loss"], losses["policy_loss"] + losses["value_loss"] + .25 * losses["auxiliary_value_loss"])
    losses["auxiliary_value_loss"].backward()
    assert model.value_head[0].weight.grad.abs().sum() > 0
    assert model.board_encoder.out_proj.weight.grad.abs().sum() > 0
    assert model.value_head[-1].weight.grad is None  # Separate final readout.


@pytest.mark.parametrize("weight", [None, .25, 0.0])
def test_auxiliary_cross_entropy_hand_calculation(weight):
    output = dict(logits=torch.zeros(2, 2), value_logits=torch.zeros(2, 3),
                  auxiliary_value_logits=torch.tensor([[0., 0., 0.], [math.log(.2), math.log(.3), math.log(.5)]]))
    batch = dict(supervised_indices=torch.tensor([1]), legal_ids=torch.tensor([[0, 1]]),
                 legal_mask=torch.tensor([[True, True]]), policy=torch.tensor([[.5, .5]]),
                 value_target=torch.tensor([[0., 0., 0.], [0., 0., 1.]]),
                 auxiliary_value_target=torch.tensor([[.1, .2, .7]], requires_grad=True))
    loss = self_play_loss(output, batch, **({} if weight is None else {"auxiliary_value_weight": weight}))
    expected_aux = -(.1*math.log(.2)+.2*math.log(.3)+.7*math.log(.5))
    assert loss["loss"].item() == pytest.approx(math.log(2) + math.log(3) + (1.0 if weight is None else weight) * expected_aux)
    if weight != 0.0:
        assert loss["auxiliary_value_loss"].item() == pytest.approx(expected_aux)
        assert not loss["auxiliary_value_loss"].requires_grad


def test_auxiliary_trainer_exact_resume_and_config_rejection(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = LearningConfig(lr=.001)
    store = SelfPlayStore(tmp_path / "replay", flush_games=1)
    store.add(game_with_wdl())
    def trainer(config=cfg):
        return Stage2Trainer(model=auxiliary_model(), config=config, move_vocab=VOCAB,
                             encoder=ENCODER, device=torch.device("cpu"), max_positions=128)
    a = trainer()
    a.begin_phase(store)
    a.train(store, exposure_budget=8)
    a.checkpoint(tmp_path / "state.pt", progress={}, store=store, config_id="aux")
    a.train(store, exposure_budget=12)
    b = trainer()
    b.resume(tmp_path / "state.pt", store=store, config_id="aux")
    b.train(store, exposure_budget=12)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration changed"):
        trainer(replace(cfg, auxiliary_value_lambda=.8)).resume(tmp_path / "state.pt", store=store, config_id="aux")
    with pytest.raises(ValueError, match="configuration changed"):
        trainer(replace(cfg, auxiliary_value_weight=.25)).resume(tmp_path / "state.pt", store=store, config_id="aux")


def test_multi_horizon_auxiliary_heads(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    lambdas = (.8, .9, .95)
    cfg = LearningConfig(lr=.001, value_weight=.25, auxiliary_value_weight=.25,
                         auxiliary_value_lambda=list(lambdas))
    assert cfg.auxiliary_value_lambda == lambdas == cfg.auxiliary_value_lambdas
    game = game_with_wdl(prefix=["f2f3", "e7e5"])
    sample = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=cfg)
    batch = collate_self_play([sample])
    per_horizon = [smoothed_search_wdl(game, decay) for decay in lambdas]
    assert batch["auxiliary_value_target"].shape == (2, 3, 3)
    for h, rows in enumerate(per_horizon):
        torch.testing.assert_close(batch["auxiliary_value_target"][:, h], torch.tensor(rows))

    def model(heads=3):
        return HSTUChessModel(replace(auxiliary_model().config, auxiliary_value_heads=heads))
    m = model()
    assert m.auxiliary_value_head.out_features == 9
    m.train()
    mask = create_batch_dense_mask(batch["seq_offsets"], total_tokens=batch["total_tokens"], device="cpu")
    with torch.no_grad():
        m.auxiliary_value_head.weight.normal_(0, .1)
    output = m(batch, block_mask=mask, return_loss=False)
    losses = self_play_loss(output, batch, value_weight=.25, auxiliary_value_weight=.25)
    logits = output["auxiliary_value_logits"].index_select(0, batch["supervised_indices"]).view(-1, 3, 3)
    expected = [-(batch["auxiliary_value_target"][:, h] * logits[:, h].log_softmax(-1)).sum(-1).mean()
                for h in range(3)]
    for h in range(3):
        torch.testing.assert_close(losses[f"auxiliary_value_loss_{h}"], expected[h])
    torch.testing.assert_close(
        losses["loss"],
        losses["policy_loss"] + .25 * losses["value_loss"] + .25 * sum(expected))
    torch.testing.assert_close(losses["auxiliary_value_loss"], sum(expected) / 3)

    with pytest.raises(ValueError, match="auxiliary head count"):
        Stage2Trainer(model=model(heads=1), config=cfg, move_vocab=VOCAB, encoder=ENCODER,
                      device=torch.device("cpu"), max_positions=128)
    with pytest.raises(ValueError, match="nonempty"):
        LearningConfig(auxiliary_value_lambda=[])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LearningConfig(auxiliary_value_lambda=[.9, 1.5])

    store = SelfPlayStore(tmp_path / "replay", flush_games=1)
    store.add(game_with_wdl())
    def trainer(config=cfg):
        return Stage2Trainer(model=model(), config=config, move_vocab=VOCAB,
                             encoder=ENCODER, device=torch.device("cpu"), max_positions=128)
    a = trainer()
    a.begin_phase(store)
    a.train(store, exposure_budget=8)
    a.checkpoint(tmp_path / "state.pt", progress={}, store=store, config_id="aux3")
    a.train(store, exposure_budget=12)
    b = trainer()
    b.resume(tmp_path / "state.pt", store=store, config_id="aux3")
    b.train(store, exposure_budget=12)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration changed"):
        trainer(replace(cfg, auxiliary_value_lambda=(.8, .9, .98))).resume(
            tmp_path / "state.pt", store=store, config_id="aux3")


def test_value_search_mix_blends_outcome_with_ply_search_wdl():
    game = game_with_wdl(prefix=["f2f3", "e7e5"])
    game["targets"][0]["search_wdl"] = [.6, .3, .1]  # White to move, will lose.
    plain = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128,
                        learning=LearningConfig())
    mixed = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128,
                        learning=LearningConfig(value_search_mix=.25))
    rows = [i for i, flag in enumerate(plain["has_value_target"]) if flag]
    assert [plain["value_target"][i] for i in rows] == [[1., 0, 0], [0, 0, 1.]]
    assert [v for i in rows for v in mixed["value_target"][i]] == pytest.approx(
        [.75 + .25 * .6, .25 * .3, .25 * .1, .25 * .2, .25 * .3, .75 + .25 * .5])
    assert all(sum(mixed["value_target"][i]) == pytest.approx(1) for i in rows)
    assert mixed["value_target"][:rows[0]] == plain["value_target"][:rows[0]]
    with pytest.raises(ValueError, match="value_search_mix requires"):
        reconstruct(mate_game(), move_vocab=VOCAB, encoder=ENCODER, max_positions=128,
                    learning=LearningConfig(value_search_mix=.25))
    with pytest.raises(ValueError, match=r"value_search_mix must be in \[0, 1\]"):
        LearningConfig(value_search_mix=1.5)


def test_value_mix_diagnostics_use_actual_outcome():
    game = game_with_wdl(prefix=["f2f3", "e7e5"])
    learning = LearningConfig(value_search_mix=.25)
    sample = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=learning)
    assert sample["outcome_target"] == [[1., 0, 0], [0, 0, 1.]]
    batch = collate_self_play([sample])
    n = batch["total_tokens"]
    # Favour the correct side at both positions: White loses, Black wins.
    value_logits = torch.zeros(n, 3)
    value_logits[batch["supervised_indices"][0]] = torch.tensor([2., 0, 0])
    value_logits[batch["supervised_indices"][1]] = torch.tensor([0., 0, 2.])
    output = dict(logits=torch.zeros(n, len(VOCAB)), value_logits=value_logits)
    mixed = self_play_loss(output, batch, auxiliary_value_weight=0.0)
    assert mixed["decisive_positions"] == 2
    assert mixed["conditional_wl_accuracy"] == 1
    assert mixed["observed_draw"] == 0
    labels = batch["value_target"].index_select(0, batch["supervised_indices"])
    expected = -(labels * value_logits.index_select(0, batch["supervised_indices"]).log_softmax(-1)).sum(-1).mean()
    torch.testing.assert_close(mixed["value_loss"], expected)
    assert "outcome_target" not in reconstruct(
        game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=LearningConfig())


def completed_raw_q(result):
    """Completed Q before Gumbel's scaling: visited means, v_mix elsewhere."""
    logs = result.root_log_priors
    top = max(logs)
    weights = [math.exp(x - top) for x in logs]
    probs = [max(w / sum(weights), 1.1754943508222875e-38) for w in weights]
    seen = [i for i, n in enumerate(result.visits) if n]
    mass = sum(probs[i] for i in seen)
    count = sum(result.visits)
    mixed = (result.root_value + count * sum(probs[i] * result.qvalues[i] for i in seen) / mass) / (count + 1)
    return [q if n else mixed for q, n in zip(result.qvalues, result.visits)]


@pytest.mark.parametrize("budget, top_m", [(16, 4), (64, 8), (256, 16)])
def test_improved_wdl_is_the_value_of_the_improved_policy(budget, top_m):
    from tests.test_puct_reference import HashEvaluator

    board = chess.Board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP1QPPP/R3K2R w KQkq - 0 9")
    result = select_gumbel(evaluator=HashEvaluator(), board=board, noise=[0.] * board.legal_moves.count(),
                           config=GumbelConfig(simulations=budget, top_m=top_m, max_depth=4))
    loss, draw, win = result.improved_wdl
    assert min(result.improved_wdl) >= 0 and loss + draw + win == pytest.approx(1)
    q = completed_raw_q(result)
    assert win - loss == pytest.approx(sum(p * v for p, v in zip(result.policy, q)), abs=1e-12)
    # Halving spends most simulations on moves it discards; that average differs.
    assert max(abs(a - b) for a, b in zip(result.search_wdl, result.improved_wdl)) > 1e-3


def test_improved_wdl_of_a_visit_policy_is_search_wdl():
    from imba_chess.eval.puct_search import PuctConfig, select_puct
    from tests.test_puct_reference import HashEvaluator

    board = chess.Board()
    visits = select_puct(evaluator=HashEvaluator(), board=board, noise=0.0,
                         config=PuctConfig(algorithm="puct_variance", simulations=64, max_depth=4))
    assert visits.improved_wdl == pytest.approx(visits.search_wdl, abs=1e-12)
    rpo = select_puct(evaluator=HashEvaluator(), board=board, noise=0.0,
                      config=PuctConfig(algorithm="puct_variance", simulations=64, max_depth=4,
                                        target_policy="rpo"))
    loss, _, win = rpo.improved_wdl
    assert win - loss == pytest.approx(sum(p * v for p, v in zip(rpo.policy, completed_raw_q(rpo))), abs=1e-12)
    # One legal move: every policy is that move, so both labels agree.
    only = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    single = select_gumbel(evaluator=HashEvaluator(), board=only, noise=[0.],
                           config=GumbelConfig(simulations=8, top_m=4, max_depth=3))
    assert single.improved_wdl == pytest.approx(single.search_wdl, abs=1e-12)


def test_value_search_source_selects_the_label_for_mix_and_auxiliary_targets():
    game = game_with_wdl(prefix=["f2f3", "e7e5"])
    for target in game["targets"]:
        target["improved_wdl"] = [.1, .2, .7]
    learning = LearningConfig(value_search_mix=.25, auxiliary_value_weight=.25,
                              auxiliary_value_lambda=.5, value_search_source="improved_wdl")
    sample = reconstruct(game, move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=learning)
    rows = [i for i, flag in enumerate(sample["has_value_target"]) if flag]
    assert [v for i in rows for v in sample["value_target"][i]] == pytest.approx(
        [.75 + .25 * .1, .25 * .2, .25 * .7, .25 * .1, .25 * .2, .75 + .25 * .7])
    assert sample["auxiliary_value_target"] == smoothed_search_wdl(game, .5, "improved_wdl")
    assert smoothed_search_wdl(game, 0, "improved_wdl") == [[.1, .2, .7]] * 2
    assert smoothed_search_wdl(game, .5) != smoothed_search_wdl(game, .5, "improved_wdl")
    with pytest.raises(ValueError, match="requires recorded improved_wdl"):
        reconstruct(game_with_wdl(), move_vocab=VOCAB, encoder=ENCODER, max_positions=128, learning=learning)
    with pytest.raises(ValueError, match="requires recorded improved_wdl"):
        smoothed_search_wdl(game_with_wdl(), .5, "improved_wdl")
    game["targets"][0]["improved_wdl"] = [.2, .3, .8]
    with pytest.raises(ValueError, match="improved_wdl must be a normalized"):
        validate_game(game)
    with pytest.raises(ValueError, match="value_search_source"):
        LearningConfig(value_search_source="played_q")


def test_value_search_source_default_keeps_run_identity():
    from imba_chess.self_play.config import load_config

    config = load_config("config/self_play.toml")
    assert config.learning.value_search_source == "search_wdl"
    # The live Gumbel recipe's identifier before the option existed (main at 97b0fed).
    assert config.identifier == "fa9936a468c3c5fe67f9a9d0068977e238401702167f90dbc1889b81a83cfc65"
    changed = replace(config, learning=replace(config.learning, value_search_source="improved_wdl"))
    assert changed.identifier != config.identifier
