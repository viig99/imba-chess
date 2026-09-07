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


@pytest.mark.parametrize('policy,cache,lmr', [('value_search_alphabeta','off',False),
    ('value_search_pvs','off',False), ('value_search_pvs','context',False),
    ('value_search_pvs','off',True), ('value_search_pvs','context',True)])
@pytest.mark.parametrize('cli', [False, True])
def test_config_roundtrip(tmp_path, monkeypatch, policy, cache, lmr, cli):
    import sys
    import json
    module = _load_eval_script_module()
    config = tmp_path/'config.toml'
    config.write_text('[eval_vs_stockfish]\n'+f'model_move_policy = "{policy}"\n'+
                      f'search_score_cache = "{("off" if cache == "context" else "context") if cli else cache}"\n'+
                      f'search_lmr = {str(not lmr if cli else lmr).lower()}\n'+
                      f'search_iterative_deepening = {str(cli).lower()}\n')
    output = tmp_path/'results.json'
    argv = ['eval', '--config', str(config), '--checkpoint', 'unused', '--stockfish-path', sys.executable,
            '--device', 'cpu', '--no-compile', '--no-save-games', '--concurrent-games', '2',
            '--ladder-elos', '2400', '--ladder-games-per-segment', '1',
            '--no-include-full-strength-segment', '--output-json', str(output)]
    if cli:
        argv += ['--search-score-cache', cache, '--search-lmr' if lmr else '--no-search-lmr', '--no-search-iterative-deepening']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(module, 'load_hstu_checkpoint', lambda **kw: (None, False))
    def segment(**kw):
        cfg = kw['halving_config']
        assert (cfg.policy, cfg.score_cache, cfg.lmr, cfg.iterative_deepening) == (policy, cache, lmr, False)
        return module.EvalSummary(games=1, incomplete_games=1,
            game_records=[dict(game_idx=0, completed=False, result='*')])
    monkeypatch.setattr(module, '_run_segment_actor_mode', segment)
    module.main()
    payload = json.loads(output.read_text())['aggregate']
    assert payload['run_config']['value_rerank_lambda'] == 'not applicable'
    assert payload['run_config']['search']['search_top_m'] == 'not applicable'
    assert payload['game_records'][0]['completed'] is False


@pytest.mark.parametrize('flags', [
    ['--model-move-policy','value_search_alphabeta','--search-lmr'],
    ['--model-move-policy','value_search_alphabeta','--search-max-depth','129'],
    ['--model-move-policy','value_search_pvs','--search-tactical-coverage'],
    ['--model-move-policy','value_search_pvs','--search-quiescence-plies','1'],
    ['--model-move-policy','value_search_halving','--search-score-cache','context'],
])
def test_reject_unsupported_before_model_load(monkeypatch, flags):
    import sys
    module = _load_eval_script_module()
    monkeypatch.setattr(sys, 'argv', ['eval','--checkpoint','unused']+flags)
    monkeypatch.setattr(module, 'load_hstu_checkpoint', lambda **kw: pytest.fail('loaded checkpoint'))
    with pytest.raises(ValueError):
        module.main()


@pytest.mark.parametrize('policy', sorted(POLICIES))
def test_real_actor_server_cleanup_and_reports(policy):
    from pathlib import Path
    import multiprocessing
    from imba_chess.config import BoardStateConfig
    from test_eval_vs_stockfish import STATIC_VOCAB_PATH, _actor_mode_fake_engine_factory
    module = _load_eval_script_module()
    vocab = MoveVocab.build_static()
    summary = module._run_segment_actor_mode(
        stockfish_path=Path('unused'), segment_options={}, segment_name='ab-integration',
        model=_tiny_actor_mode_model(vocab), games=2, max_plies=4,
        engine_limit=chess.engine.Limit(time=.01), device=torch.device('cpu'), dtype=torch.float32,
        model_move_policy=policy, value_rerank_top_k=1, value_rerank_lambda=.05,
        opening_random_plies=0, seed=42, concurrent_games=2, vocab_path=STATIC_VOCAB_PATH,
        vocab_include_unk=False, board_state_config=asdict(BoardStateConfig()),
        halving_config=AlphaBetaConfig(budget=32, max_depth=3, policy=policy, score_cache='context', lmr=policy=='value_search_pvs'),
        fake_engine_factory=_actor_mode_fake_engine_factory)
    assert summary.games == 2 and len(summary.game_records) == 2
    assert len(summary.search_reports) == 4
    assert summary.inference_stats['wave_rows'] == summary.search_stats['new_neural_evaluations']
    assert any(k.startswith('wave_batch_size_') for k in summary.inference_stats)
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize('policy', sorted(POLICIES))
def test_nonfinite_root_value_is_rejected(monkeypatch, policy):
    module = _load_eval_script_module()
    vocab = MoveVocab.build_static()
    monkeypatch.setattr(module, '_forward_model', lambda **kw: dict(
        logits=torch.zeros(1, len(vocab)), value_logits=torch.full((1, 3), float('nan')),
        kv_caches=[]))
    with pytest.raises(ValueError, match='Non-finite root value'):
        module._select_model_move(model=None, batch={'total_tokens': 1}, board=chess.Board(),
            move_vocab=vocab, board_state_encoder=BoardStateEncoder(), device=torch.device('cpu'),
            dtype=torch.float32, policy=policy, value_rerank_top_k=1, value_rerank_lambda=.05,
            halving_config=AlphaBetaConfig(policy=policy))
