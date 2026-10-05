import math
from dataclasses import asdict, replace
import importlib
import io

import chess
import imba_chess_native as cc
import numpy as np
import pytest
import torch

from imba_chess.config import BoardStateConfig, ModelConfig, RepoConfig
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.collate import collate_jagged_batch
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.position_evaluator import _SequenceHistory, load_hstu_checkpoint
from imba_chess.klent.config import KlentConfig
from imba_chess.klent.engine import SlotEngine
from imba_chess.klent.run import KlentRun, build_model, run
from imba_chess.klent.selfplay import BoardCodec, SelfPlay
from imba_chess.klent.targets import improved_policy, lambda_from_tau, lambda_returns
from imba_chess.klent.train import KlentTrainer, build_batch, klent_loss
from imba_chess.model import HSTUChessConfig, HSTUChessModel, create_batch_dense_mask
from imba_chess.model.checkpoint import load_initial_weights

VOCAB = MoveVocab.load("artifacts/move_vocab_static_uci.json")
BOARD = BoardStateConfig(en_passant="legal")


def tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    config = dict(
        move_vocab_size=len(VOCAB),
        model_dim=16,
        linear_hidden_dim=4,
        attention_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        max_position_embeddings=64,
        enable_value_head=True,
        enable_action_value_head=True,
    )
    model = HSTUChessModel(HSTUChessConfig(**{**config, **overrides}))
    # The head is zero-initialized; randomize it so Q-dependent paths are exercised.
    head = model.action_value_head
    if head is not None:
        readout = head if isinstance(head, torch.nn.Linear) else head[-1]
        torch.nn.init.normal_(readout.weight, std=0.5)
    return model.eval()


def reference_targets(rewards, values, terminated, lam):
    """Line-for-line port of KazukiOhta/klent main.py calculate_targets."""
    values_next = list(values[1:]) + [math.nan]
    carry, out = math.nan, []
    for r, v, t in reversed(list(zip(rewards, values_next, terminated))):
        carry = r if t else r + -1 * ((1 - lam) * v + lam * carry)
        out.append(carry)
    return out[::-1]


@pytest.mark.parametrize("final", [1.0, 0.0, -1.0])
def test_lambda_returns_match_reference(final):
    rng = np.random.default_rng(0)
    values = rng.uniform(-1, 1, 7)
    lam = lambda_from_tau(8.0)
    rewards = [0.0] * 6 + [final]
    expected = reference_targets(rewards, values, [False] * 6 + [True], lam)
    np.testing.assert_allclose(lambda_returns(final, values, lam), expected, rtol=1e-6)


def test_lambda_returns_hand_computed():
    lam = 0.5
    # Two plies, the second mates: G_1 = 1, G_0 = -((1-lam)*V_1 + lam*G_1).
    np.testing.assert_allclose(lambda_returns(1.0, [0.3, 0.2], lam), [-(0.5 * 0.2 + 0.5), 1.0])


def test_improved_policy_matches_reference_formula():
    logits = torch.tensor([[1.0, -2.0, 0.5, 9.0]])
    q = torch.tensor([[0.1, 0.9, -0.3, 5.0]])
    mask = torch.tensor([[True, True, True, False]])
    policy = improved_policy(logits, q, mask, alpha=0.03, beta=0.1)
    expected = torch.softmax((0.1 * logits[0, :3] + q[0, :3]) / 0.13, 0)
    torch.testing.assert_close(policy[0, :3], expected)
    assert policy[0, 3] == 0


def test_mate_reward_sign():
    board = cc.Board.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    ids, moves = BoardCodec(VOCAB, BOARD).legal(board)
    mate = next(m for m, i in zip(moves, ids) if VOCAB.decode(i) == "a1a8")
    _, _, value = cc.push_and_classify(board, mate, [], True)
    # Child side to move is mated; the mover's reward is -value.
    assert value == -1


def _play(model, *, slots, positions, max_plies, bootstrap="q", advantage=False):
    engine = SlotEngine(model, slots=slots, start_id=VOCAB.start_id, device="cpu", dtype=torch.float32)
    selfplay = SelfPlay(
        engine,
        BoardCodec(VOCAB, BOARD),
        alpha=0.03,
        beta=0.1,
        lam=lambda_from_tau(8.0),
        max_plies=max_plies,
        start_id=VOCAB.start_id,
        generator=torch.Generator().manual_seed(0),
        advantage=advantage,
    )
    return selfplay.collect(positions, bootstrap=bootstrap)


@pytest.mark.parametrize("tied", [True, False])
def test_incremental_decode_matches_full_forward(tied):
    """Stored pi' and values come from one-token decodes over slot caches with
    resets; recomputing them from one jagged forward over the finished games
    must agree, which checks the engine, BOS prefill, resets and batch building."""
    torch.set_num_threads(1)
    model = tiny_model(tie_policy_embeddings=tied)
    games, metrics = _play(model, slots=3, positions=3 * 23, max_plies=7)
    assert len(games) >= 6 and metrics["selfplay/positions_kept"] > 0
    batch = build_batch(games, VOCAB.start_id)
    with torch.no_grad():
        out = model(
            batch,
            block_mask=create_batch_dense_mask(batch["seq_offsets"], total_tokens=batch["total_tokens"]),
            return_loss=False,
        )
    index = batch["supervised_indices"]
    legal, mask = batch["legal_ids"], batch["legal_mask"]
    logits = out["logits"][index].gather(1, legal)
    q = out["q"][index].gather(1, legal)
    policy = improved_policy(logits, q, mask, alpha=0.03, beta=0.1)
    torch.testing.assert_close(policy, batch["policy"], atol=1e-5, rtol=1e-4)


def test_batch_tokens_match_python_chess_history():
    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 9, max_plies=8)
    game = games[0]
    history = _SequenceHistory(move_vocab=VOCAB, board_state_encoder=BoardStateEncoder(BOARD))
    board = chess.Board()
    for move_id in game["move_id"]:
        history.append_observed_position(board)
        uci = VOCAB.decode(int(move_id))
        history.record_played_move(uci)
        board.push_uci(uci)
    sample = {key: getattr(history, key) for key in (
        "seq_token_id", "piece_ids", "turn_id", "castle_id", "ep_file_id",
        "halfmove_bucket_id", "fullmove_bucket_id", "prev_move_id",
        "target_move_id", "played_by_elo")}
    sample.update(game_id="g", game_result_white=0, value_target=[[0.0] * 3] * len(sample["seq_token_id"]),
                  has_value_target=[False] * len(sample["seq_token_id"]))
    expected = collate_jagged_batch([sample])
    actual = build_batch([game], VOCAB.start_id)
    for key in ("seq_token_id", "piece_ids", "turn_id", "castle_id", "ep_file_id",
                "halfmove_bucket_id", "fullmove_bucket_id", "prev_move_id", "seq_offsets"):
        assert torch.equal(actual[key].long(), expected[key].long()), key


def test_game_records_are_consistent():
    games, metrics = _play(tiny_model(), slots=2, positions=2 * 30, max_plies=6)
    for game in games:
        plies = len(game["move_id"])
        assert game["piece_ids"].shape == (plies, 64)
        assert len(game["legal_offsets"]) == plies + 1
        sums = np.add.reduceat(game["policy"], game["legal_offsets"][:-1])
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)
        assert abs(game["returns"]).max() <= 1.0 + 1e-6
    assert metrics["selfplay/positions_played"] == 60
    assert metrics["selfplay/ent_1"] > 0


def test_trainer_reduces_loss():
    torch.manual_seed(0)
    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 24, max_plies=6)
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=64, grad_clip=1.0, compile_model=False)
    trainer.set_phase(warmup=False, lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(0)
    first = trainer.train_epoch(games, rng, policy_weight=1.0, value_weight=1.0)
    for _ in range(30):
        last = trainer.train_epoch(games, rng, policy_weight=1.0, value_weight=1.0)
    assert last["train/loss"] < first["train/loss"]
    assert first["train/q_loss"] > 0


def test_warmup_trains_only_the_heads():
    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 12, max_plies=6, bootstrap="value")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=64, grad_clip=1.0, compile_model=False)
    trainer.set_phase(warmup=True, lr=1e-2, weight_decay=0.0)
    trainer.train_epoch(games, np.random.default_rng(0), policy_weight=0.0, value_weight=1.0)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    heads = {k for k in before if k.startswith(("action_value_head.", "value_head."))}
    assert changed and changed <= heads
    assert {"action_value_head.weight", "action_value_head.bias"} <= changed


def test_checkpoint_without_action_head_loads_only_with_allowlist():
    plain = tiny_model(enable_action_value_head=False)
    model = tiny_model()
    state = {"model": plain.state_dict()}
    with pytest.raises(RuntimeError):
        load_initial_weights(model, state)
    load_initial_weights(model, state, allow_missing_prefixes=("action_value_head.",))
    extra = dict(plain.state_dict(), stray=torch.zeros(1))
    with pytest.raises(RuntimeError):
        load_initial_weights(model, {"model": extra}, allow_missing_prefixes=("action_value_head.",))


def test_config_defaults_follow_reference():
    cfg = KlentConfig()
    assert (cfg.alpha, cfg.beta, cfg.tau) == (0.03, 0.1, 8.0)
    assert cfg.positions_per_iteration == 1024 * 2048
    with pytest.raises(ValueError):
        KlentConfig(bootstrap="mc")


@pytest.fixture
def small_run(monkeypatch):
    torch.set_num_threads(1)
    repo = RepoConfig(model=ModelConfig(
        model_dim=16, linear_hidden_dim=4, attention_dim=4, num_heads=2,
        num_layers=2, dropout=0.0, max_position_embeddings=64,
        enable_value_head=True, value_head_width=8,
    ))
    monkeypatch.setattr(importlib.import_module("imba_chess.klent.run"),
                        "load_repo_config", lambda path: repo)
    cfg = KlentConfig(slots=2, positions_per_iteration=24, total_positions=48,
                      batch_tokens=64, max_plies=6, compile=False)
    return cfg, repo


def test_scratch_starts_uniform_preserves_move_embeddings_and_learns(small_run):
    cfg, repo = small_run
    model = build_model(cfg, repo, VOCAB).eval()
    assert model.prediction_head.weight is not model.prev_move_embedding.weight
    assert torch.count_nonzero(model.prev_move_embedding.weight) > 0
    games, _ = _play(model, slots=2, positions=24, max_plies=6)
    for game in games:
        for start, end in zip(game["legal_offsets"][:-1], game["legal_offsets"][1:]):
            np.testing.assert_allclose(game["policy"][start:end], 1.0 / (end - start))
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=64, grad_clip=1.0, compile_model=False)
    trainer.set_phase(warmup=False, lr=1e-3, weight_decay=0.0)
    trainer.train_epoch(games, np.random.default_rng(0), policy_weight=1.0, value_weight=1.0)
    assert torch.count_nonzero(model.prediction_head.weight) > 0


def test_pretrained_initialization_preserves_weights_and_policy_tying(small_run, tmp_path):
    cfg, repo = small_run
    config = replace(build_model(cfg, repo, VOCAB).config,
                     enable_action_value_head=False, tie_policy_embeddings=True)
    plain = HSTUChessModel(config)
    initial = tmp_path / "initial.pt"
    torch.save(dict(model=plain.state_dict()), initial)
    model = build_model(replace(cfg, init=str(initial)), repo, VOCAB)
    assert model.prediction_head.weight is model.prev_move_embedding.weight
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    assert torch.count_nonzero(model.action_value_head.weight) == 0


def test_policy_loss_matches_full_vocabulary_ce_and_suppresses_illegal_logits():
    logits = torch.tensor([[1., 0., 9., -1.], [2., 3., -2., 8.]], requires_grad=True)
    q = torch.tensor([[.1, .2, .3, .4], [.4, .3, .2, .1]], requires_grad=True)
    batch = dict(
        supervised_indices=torch.tensor([0, 1]),
        legal_ids=torch.tensor([[0, 1, 0], [1, 2, 0]]),
        legal_mask=torch.tensor([[True, True, False], [True, True, False]]),
        policy=torch.tensor([[.8, .2, 0.], [.25, .75, 0.]]),
        move_id=torch.tensor([0, 2]), returns=torch.tensor([-.2, .5]),
        outcome=torch.tensor([0, 2]),
    )
    wdl = torch.tensor([[0., 1., 2.], [2., 1., 0.]], requires_grad=True)
    losses = klent_loss(dict(logits=logits, q=q, value_logits=wdl), batch,
                        policy_weight=1.0, value_weight=1.0)
    dense_target = torch.tensor([[.8, .2, 0., 0.], [0., .25, .75, 0.]])
    expected = -(dense_target * torch.log_softmax(logits, -1)).sum(-1).mean()
    torch.testing.assert_close(losses["policy_loss"], expected)
    torch.testing.assert_close(losses["q_loss"], torch.tensor(.09))
    torch.testing.assert_close(losses["loss"], expected + losses["q_loss"]
                               + torch.nn.functional.cross_entropy(wdl, batch["outcome"]))
    losses["loss"].backward()
    assert logits.grad[0, 2] > 0 and logits.grad[1, 3] > 0
    assert q.grad[0, 1:].count_nonzero() == 0
    assert q.grad[1, [0, 1, 3]].count_nonzero() == 0


def _assert_optimizer_state_equal(a, b):
    assert a.keys() == b.keys()
    for index, values in a.items():
        assert values.keys() == b[index].keys()
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, b[index][key], rtol=0, atol=0)
            else:
                assert value == b[index][key]


@pytest.mark.parametrize("warmup_iterations", [0, 1, 2])
def test_resume_preserves_next_update_and_optimizer_state(small_run, warmup_iterations):
    cfg, _ = small_run
    cfg = replace(cfg, q_warmup_iterations=warmup_iterations)
    original = KlentRun(cfg, device="cpu")
    original.run_iteration()
    # Serialize before either model advances: state_dict otherwise aliases tensors.
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=False)
    resumed = KlentRun(cfg, device="cpu", resume_state=state)
    _assert_optimizer_state_equal(original.trainer.optimizer.state_dict()["state"],
                                  resumed.trainer.optimizer.state_dict()["state"])
    original.run_iteration()
    resumed.run_iteration()
    for key, value in original.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    _assert_optimizer_state_equal(original.trainer.optimizer.state_dict()["state"],
                                  resumed.trainer.optimizer.state_dict()["state"])
    assert original.positions == resumed.positions and original.iteration == resumed.iteration


@pytest.mark.parametrize("warmup_iterations", [0, 2])
def test_resume_runtime_overrides_preserve_moments_and_apply_optimizer_settings(small_run, warmup_iterations):
    cfg, _ = small_run
    cfg = replace(cfg, q_warmup_iterations=warmup_iterations)
    original = KlentRun(cfg, device="cpu")
    original.run_iteration()
    state = original.state_dict()
    changed = replace(cfg, init="no-longer-needed.pt", total_positions=96,
                      batch_tokens=32, slots=3, compile=True, inference_dtype="float32",
                      lr=2e-4, warmup_lr=4e-4, weight_decay=.02, grad_clip=.5)
    resumed = KlentRun(changed, device="cpu", resume_state=state)
    resumed._phase()
    groups = resumed.trainer.optimizer.param_groups
    lr = changed.warmup_lr if warmup_iterations else changed.lr
    assert [g["lr"] for g in groups] == [lr, lr]
    assert [g["weight_decay"] for g in groups] == [changed.weight_decay, 0.0]
    _assert_optimizer_state_equal(state["trainer"]["optimizer"]["state"],
                                  resumed.trainer.optimizer.state_dict()["state"])
    for key, value in state["model"].items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert not resumed.model.config.tie_policy_embeddings
    resumed.run_iteration()


@pytest.mark.parametrize("key,value", [("alpha", .1), ("tau", 4.),
                                       ("seed", 1), ("q_warmup_iterations", 1)])
def test_resume_rejects_algorithm_changes(small_run, key, value):
    cfg, _ = small_run
    original = KlentRun(cfg, device="cpu")
    original._phase()
    with pytest.raises(ValueError, match=key):
        KlentRun(replace(cfg, **{key: value}), device="cpu", resume_state=original.state_dict())


def test_legacy_resume_needs_no_initial_checkpoint_and_keeps_weight_tying(small_run, tmp_path):
    cfg, repo = small_run
    cfg = replace(cfg, init=str(tmp_path / "missing-53k.pt"))
    model = HSTUChessModel(replace(build_model(replace(cfg, init="scratch"), repo, VOCAB).config,
                                   tie_policy_embeddings=True))
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=64, grad_clip=1.0, compile_model=False)
    trainer.set_phase(warmup=False, lr=cfg.lr, weight_decay=0.0)
    state = dict(model=model.state_dict(), trainer=trainer.state_dict(), iteration=1,
                 positions=24, config=asdict(cfg))  # Old checkpoint format: no model metadata.
    torch.save(state, tmp_path / "checkpoint.pt")
    run(replace(cfg, total_positions=48), output=tmp_path, device="cpu", save_every=1)
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert saved["iteration"] == 2 and saved["positions"] == 48
    assert saved["model_config"]["tie_policy_embeddings"]
    assert (tmp_path / "actor-0002.pt").exists()


def test_scratch_actor_and_training_checkpoint_load_for_eval_without_retying(small_run, tmp_path):
    cfg, repo = small_run
    run(replace(cfg, total_positions=24), output=tmp_path, device="cpu", save_every=1)
    for name in ("actor-0001.pt", "checkpoint.pt"):
        saved = torch.load(tmp_path / name, weights_only=False)
        model, _ = load_hstu_checkpoint(checkpoint_path=tmp_path / name, repo_config=repo,
                                       move_vocab=VOCAB, device=torch.device("cpu"),
                                       compile_model=False)
        assert model.prediction_head.weight is not model.prev_move_embedding.weight
        for key, value in saved["model"].items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_resume_rejects_changed_model_architecture(small_run):
    cfg, _ = small_run
    original = KlentRun(cfg, device="cpu")
    original._phase()
    state = original.state_dict()
    state["model_config"]["dropout"] = .1
    with pytest.raises(ValueError, match="model architecture"):
        KlentRun(cfg, device="cpu", resume_state=state)


def test_nonfinite_loss_fails_at_epoch_end(monkeypatch):
    """Per-step sync checks are gone; one NaN anywhere still fails the epoch
    loudly, before the caller can write a checkpoint."""
    import imba_chess.klent.train as train_module

    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 24, max_plies=6)
    real = train_module.klent_loss
    calls = []

    def poisoned(*args, **kwargs):
        losses = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            losses["loss"] = losses["loss"] * float("nan")
        return losses

    monkeypatch.setattr(train_module, "klent_loss", poisoned)
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=16, grad_clip=1.0, compile_model=False)
    trainer.set_phase(warmup=False, lr=1e-3, weight_decay=0.0)
    with pytest.raises(FloatingPointError):
        trainer.train_epoch(games, np.random.default_rng(0), policy_weight=1.0, value_weight=1.0)
    assert len(calls) > 1  # the epoch kept running; the check is deferred


def test_prune_snapshots_keeps_newest(tmp_path):
    from imba_chess.klent.run import prune_snapshots

    for iteration in (10, 20, 30, 40, 50, 60, 70):
        (tmp_path / f"actor-{iteration:04d}.pt").write_bytes(b"x")
    (tmp_path / "checkpoint.pt").write_bytes(b"x")
    prune_snapshots(tmp_path, 5)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "actor-0030.pt", "actor-0040.pt", "actor-0050.pt", "actor-0060.pt",
        "actor-0070.pt", "checkpoint.pt",
    ]
    prune_snapshots(tmp_path, None)
    assert len(list(tmp_path.glob("actor-*.pt"))) == 5


def test_frozen_value_head_never_changes():
    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 24, max_plies=6, bootstrap="value")
    before = {k: v.clone() for k, v in model.state_dict().items()
              if k.startswith(("value_head.", "action_value_head."))}
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=64, grad_clip=1.0, compile_model=False,
                           freeze_value_head=True)
    for warmup in (True, False):
        trainer.set_phase(warmup=warmup, lr=1e-2, weight_decay=0.0)
        trainer.train_epoch(games, np.random.default_rng(0), policy_weight=0.0 if warmup else 1.0,
                            value_weight=0.0)
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items() if k.startswith("value_head."))
    assert not torch.equal(after["action_value_head.weight"], before["action_value_head.weight"])


def test_freeze_value_head_config_rules():
    KlentConfig(init="some.pt", freeze_value_head=True, value_weight=0.0)
    with pytest.raises(ValueError, match="value_weight"):
        KlentConfig(init="some.pt", freeze_value_head=True)
    with pytest.raises(ValueError, match="checkpoint"):
        KlentConfig(freeze_value_head=True, value_weight=0.0)


def test_trunk_gradient_probe_logs_per_loss_norms():
    from imba_chess.klent.train import trunk_parameters

    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 24, max_plies=6)
    trunk_ids = {id(p) for p in trunk_parameters(model)}
    assert id(model.prev_move_embedding.weight) not in trunk_ids  # tied policy matrix
    assert not trunk_ids & {id(p) for p in model.action_value_head.parameters()}
    trainer = KlentTrainer(model, device=torch.device("cpu"), start_id=VOCAB.start_id,
                           batch_tokens=16, grad_clip=1.0, compile_model=False, probe_every=2)
    trainer.set_phase(warmup=False, lr=1e-3, weight_decay=0.0)
    metrics = trainer.train_epoch(games, np.random.default_rng(0), policy_weight=1.0, value_weight=1.0)
    for key in ("grad_trunk_policy", "grad_trunk_q", "grad_trunk_value",
                "grad_trunk_cos_policy_q", "grad_trunk_ratio_policy_q"):
        assert np.isfinite(metrics[f"train/{key}"]), key
    assert -1.0 <= metrics["train/grad_trunk_cos_policy_q"] <= 1.0
    # Warm-up freezes the trunk, so there is nothing to probe.
    trainer.set_phase(warmup=True, lr=1e-3, weight_decay=0.0)
    warm = trainer.train_epoch(games, np.random.default_rng(0), policy_weight=0.0, value_weight=0.0)
    assert not any(k.startswith("train/grad_trunk") for k in warm)


def test_private_q_mlp_starts_at_zero_and_matches_incremental_decode():
    model = tiny_model(action_value_head_blocks=2, action_value_head_width=8)  # random readout
    assert isinstance(model.action_value_head, torch.nn.Sequential)
    fresh = HSTUChessModel(model.config).eval()
    games, _ = _play(fresh, slots=2, positions=2 * 6, max_plies=6)
    batch = build_batch(games, VOCAB.start_id)
    with torch.no_grad():
        q = fresh(batch, block_mask=create_batch_dense_mask(batch["seq_offsets"], total_tokens=batch["total_tokens"]),
                  return_loss=False)["q"]
    assert torch.count_nonzero(q) == 0  # zero-initialised readout: Q = 0, as in the reference
    # With a non-zero readout, step-by-step decode still matches one jagged forward.
    games, _ = _play(model, slots=3, positions=3 * 15, max_plies=6)
    batch = build_batch(games, VOCAB.start_id)
    with torch.no_grad():
        out = model(batch, block_mask=create_batch_dense_mask(batch["seq_offsets"], total_tokens=batch["total_tokens"]),
                    return_loss=False)
    index, legal, mask = batch["supervised_indices"], batch["legal_ids"], batch["legal_mask"]
    policy = improved_policy(out["logits"][index].gather(1, legal), out["q"][index].gather(1, legal),
                             mask, alpha=0.03, beta=0.1)
    torch.testing.assert_close(policy, batch["policy"], atol=1e-5, rtol=1e-4)


def test_resume_accepts_checkpoints_written_before_new_model_fields(tmp_path):
    from dataclasses import asdict
    from imba_chess.klent.run import build_model
    from imba_chess.config import load_repo_config

    cfg = KlentConfig(init="unused.pt")
    repo = load_repo_config("config/imba_chess_v4.toml")
    current = asdict(build_model(KlentConfig(), repo, VOCAB).config)
    legacy = {k: v for k, v in current.items()
              if k not in ("action_value_head_blocks", "action_value_head_width")}
    legacy["tie_policy_embeddings"] = True
    model = build_model(cfg, repo, VOCAB, resume_state=dict(model_config=legacy))
    assert model.config.action_value_head_blocks == 0


def test_resume_config_accepts_settings_added_after_the_checkpoint():
    from dataclasses import asdict
    from imba_chess.klent.run import _validate_resume_config

    cfg = KlentConfig(init="some.pt", freeze_value_head=True, value_weight=0.0)
    legacy = {k: v for k, v in asdict(cfg).items() if k not in ("q_head_blocks", "q_head_width")}
    _validate_resume_config(cfg, dict(config=legacy))  # defaults fill the gap
    with pytest.raises(ValueError, match="q_head_blocks"):
        _validate_resume_config(KlentConfig(init="some.pt", freeze_value_head=True, value_weight=0.0,
                                            q_head_blocks=2), dict(config=legacy))


def test_advantage_loss_subtracts_the_value_head():
    torch.manual_seed(0)
    positions, vocab_size = 4, len(VOCAB)
    output = dict(
        logits=torch.zeros(6, vocab_size),
        q=torch.rand(6, vocab_size) - 0.5,
        value_logits=torch.randn(6, 3),
    )
    batch = dict(
        supervised_indices=torch.tensor([1, 2, 4, 5]),
        legal_ids=torch.tensor([[3, 4]] * positions),
        legal_mask=torch.ones(positions, 2, dtype=torch.bool),
        policy=torch.full((positions, 2), 0.5),
        move_id=torch.tensor([3, 4, 3, 4]),
        returns=torch.tensor([1.0, -0.5, 0.0, 0.25]),
        outcome=torch.tensor([2, 0, 1, 1]),
    )
    plain = klent_loss(output, batch, policy_weight=1.0, value_weight=0.0)
    adv = klent_loss(output, batch, policy_weight=1.0, value_weight=0.0, advantage=True)
    idx = batch["supervised_indices"]
    a = output["q"][idx].gather(1, batch["move_id"][:, None]).squeeze(1)
    wdl = torch.softmax(output["value_logits"][idx], -1)
    v = wdl[:, 2] - wdl[:, 0]
    torch.testing.assert_close(adv["q_loss"], ((a - (batch["returns"] - v)) ** 2).mean())
    torch.testing.assert_close(plain["q_loss"], ((a - batch["returns"]) ** 2).mean())


def test_advantage_bootstrap_adds_the_value_head_back():
    """Returns must bootstrap from Q = V + sum(pi' * A), recomputed here from one
    jagged forward over a finished game."""
    model = tiny_model()
    games, _ = _play(model, slots=2, positions=2 * 14, max_plies=7, advantage=True)
    game = games[0]
    batch = build_batch([game], VOCAB.start_id)
    with torch.no_grad():
        out = model(batch, block_mask=create_batch_dense_mask(batch["seq_offsets"], total_tokens=batch["total_tokens"]),
                    return_loss=False)
    idx, legal, mask = batch["supervised_indices"], batch["legal_ids"], batch["legal_mask"]
    a = out["q"][idx].gather(1, legal)
    wdl = torch.softmax(out["value_logits"][idx], -1)
    values = (wdl[:, 2] - wdl[:, 0]) + (batch["policy"] * a.masked_fill(~mask, 0)).sum(-1)
    final = float(game["outcome"][-1])
    expected = lambda_returns(final, values.numpy(), lambda_from_tau(8.0))
    np.testing.assert_allclose(game["returns"], expected, atol=1e-5)


def test_advantage_mode_requires_a_frozen_value_head():
    KlentConfig(init="some.pt", freeze_value_head=True, value_weight=0.0, q_mode="advantage")
    with pytest.raises(ValueError, match="freeze_value_head"):
        KlentConfig(init="some.pt", q_mode="advantage")
