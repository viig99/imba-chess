from dataclasses import asdict

import chess
import pytest
import torch

from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.alphabeta import AlphaBetaConfig, POLICIES
from imba_chess.eval.actor_protocol import GameDone, WaveRequest
from test_actor_worker import _run_two_short_games
from test_eval_vs_stockfish import (
    _load_eval_script_module, _tiny_actor_mode_model, _drive_model_move_stepwise,
)


@pytest.mark.parametrize('policy', sorted(POLICIES))
def test_serial_and_scheduled_real_kv(policy):
    module = _load_eval_script_module()
    vocab = MoveVocab.build_static()
    model = _tiny_actor_mode_model(vocab)
    board = chess.Board()
    batch = module._SequenceHistory(move_vocab=vocab, board_state_encoder=BoardStateEncoder()).build_batch_for_current_position(board)
    kwargs = dict(model=model, batch=batch, board=board, move_vocab=vocab,
                  board_state_encoder=BoardStateEncoder(), device=torch.device('cpu'),
                  dtype=torch.float32, policy=policy, value_rerank_top_k=1,
                  value_rerank_lambda=0.05,
                  halving_config=AlphaBetaConfig(budget=64, max_depth=3, policy=policy))
    move, debug = module._select_model_move(**kwargs)
    other, step_debug = _drive_model_move_stepwise(module._select_model_move_stepwise(**kwargs),
                                                 model=model, device=torch.device('cpu'), dtype=torch.float32)
    assert move == other
    for key in ['score','completed_depth','attempted_depth','pv','stop_reason']:
        assert debug['search_report'][key] == step_debug['search_report'][key]
    assert debug['search_stats']['new_neural_evaluations'] <= 64


@pytest.mark.parametrize('policy', sorted(POLICIES))
def test_actor_protocol_turn_isolation_and_budget(policy):
    messages, _ = _run_two_short_games(model_move_policy=policy,
                                     halving_config=asdict(AlphaBetaConfig(budget=64, max_depth=3, policy=policy)))
    games = [msg for msg in messages if isinstance(msg, GameDone)]
    assert len(games) == 2
    for game in games:
        assert game.summary_fragment['search_stats']['new_neural_evaluations'] <= 64
        assert game.summary_fragment['incomplete_games'] == 1
    seen = set()
    for wave in [msg for msg in messages if isinstance(msg, WaveRequest)]:
        assert len(wave.rows) == 1
        for row in wave.rows:
            key = (wave.turn_id, row.node_id)
            assert key not in seen
            if row.parent_id is not None:
                assert (wave.turn_id, row.parent_id) in seen
            seen.add(key)
