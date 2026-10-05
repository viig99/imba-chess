import math

import chess
import imba_chess_native as cc
import numpy as np
import pytest
import torch

from imba_chess.config import BoardStateConfig
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.collate import collate_jagged_batch
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.klent.config import KlentConfig
from imba_chess.klent.engine import SlotEngine
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
    if model.action_value_head is not None:
        torch.nn.init.normal_(model.action_value_head.weight, std=0.5)
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


def _play(model, *, slots, positions, max_plies, bootstrap="q"):
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
    )
    return selfplay.collect(positions, bootstrap=bootstrap)


def test_incremental_decode_matches_full_forward():
    """Stored pi' and values come from one-token decodes over slot caches with
    resets; recomputing them from one jagged forward over the finished games
    must agree, which checks the engine, BOS prefill, resets and batch building."""
    torch.set_num_threads(1)
    model = tiny_model()
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
