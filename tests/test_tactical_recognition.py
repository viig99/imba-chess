import json

import chess
import pytest

from scripts.audit_tactical_recognition import (
    cluster_summary, ensure_manifest, expected_score, mark_resolution, material,
    select_positions, terminal_record,
)


def test_perspective_ldw_and_terminal_rules():
    assert expected_score([.1, .2, .7], True, True) == pytest.approx(.8)
    assert expected_score([.1, .2, .7], False, True) == pytest.approx(.2)
    assert terminal_record(chess.Board(), True) is None
    board = chess.Board()
    for move in ['f2f3', 'e7e5', 'g2g4', 'd8h4']:
        board.push_uci(move)
    assert terminal_record(board, True)['expectation'] == 0
    assert terminal_record(board, False)['expectation'] == 1


def state(ply, balance=-3, **extra):
    return dict(ply=ply, terminal=None, material=balance, capture=False, promotion=None,
                in_check=False, root_white=True, mover_white=False,
                engine=dict(expectation=.1, mate=None), verified=dict(expectation=.12, mate=None),
                **extra)


def test_resolution_requires_actual_loss_quiet_followup_and_verified_disadvantage():
    rows = [state(i) for i in range(4)]
    assert mark_resolution(rows, 0)[1]['resolution'] == ['settled_material_loss']
    # A still-open exchange is unresolved, even with the same material deficit.
    rows[2]['capture'] = True
    assert mark_resolution(rows, 0)[1]['resolution'] == []
    rows[2]['capture'] = False
    rows[1]['verified']['expectation'] = .8
    assert mark_resolution(rows, 0)[1]['resolution'] == []
    # Mere near-term truncation cannot manufacture a resolved endpoint.
    assert mark_resolution([state(1)], 0)[0]['resolution'] == []


def test_mates_are_separate_from_material_and_preserve_sign():
    row = state(1, balance=0)
    row['engine']['mate'] = -4
    row['verified']['mate'] = -3
    assert mark_resolution([row], 0)[0]['resolution'] == ['forced_mate']
    row['verified']['mate'] = 3
    assert mark_resolution([row], 0)[0]['resolution'] == []


def test_material_is_color_relative():
    board = chess.Board('4k3/8/8/8/8/8/8/Q3K3 w - - 0 1')
    assert material(board, True) == 9
    assert material(board, False) == -9


def test_manifest_rejects_incompatible_resume_without_overwriting(tmp_path):
    path = tmp_path / 'manifest.json'
    ensure_manifest(path, {'checkpoint': 'a'})
    ensure_manifest(path, {'checkpoint': 'a'})
    with pytest.raises(ValueError, match='incompatible'):
        ensure_manifest(path, {'checkpoint': 'b'})
    assert json.loads(path.read_text()) == {'checkpoint': 'a'}


def test_bootstrap_resamples_whole_games_and_retains_position_mean():
    rows = [dict(source='a', gain=0)] * 9 + [dict(source='b', gain=1)]
    out = cluster_summary(rows, 'gain')
    assert out['n'] == 10 and out['games'] == 2
    assert out['mean'] == .1
    # Whole-game draws can be all zero or all one despite unequal game sizes.
    assert out['ci95'] == [0, 1]
    assert cluster_summary([], 'gain')['mean'] is None


def test_selection_keeps_all_cases_and_checks_history_and_origin():
    screen = dict(identity=dict(checkpoint='a'), stockfish={
        'p': dict(prefix=[], fen=chess.STARTING_FEN, source='g', scores={'e2e4': {}, 'd2d4': {}})})
    audit = dict(checkpoint='a', records=[dict(position_id='p', group='blunder',
        cause='value_misorder', chosen='e2e4', best_good_move='d2d4')])
    assert len(select_positions(screen, audit)) == 1
    screen['stockfish']['p']['prefix'] = ['e2e4']
    with pytest.raises(ValueError, match='history'):
        select_positions(screen, audit)
    audit['checkpoint'] = 'b'
    with pytest.raises(ValueError, match='checkpoint'):
        select_positions(screen, audit)


def test_interrupted_engine_line_resumes_without_replaying_or_losing_moves(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.audit_tactical_recognition as audit
    class Engine:
        id = {'name': 'test'}
        def configure(self, _): pass
        def quit(self): pass
    monkeypatch.setattr(audit.chess.engine.SimpleEngine, 'popen_uci', lambda _: Engine())
    monkeypatch.setitem(audit.PROTOCOL, 'continuation_plies', 3)
    calls = []
    def score(engine, board, root_color, nodes):
        calls.append(board.fen())
        if len(calls) == 5:
            raise RuntimeError('interrupted')
        return dict(cp=0, mate=None, expectation=.5, pv=[next(iter(board.legal_moves)).uci()])
    monkeypatch.setattr(audit, 'engine_score', score)
    args = SimpleNamespace(output=tmp_path, stockfish='fake')
    pos = dict(position_id='p', source='g', prefix=[], root_white=True, initial_material=0, bad='e2e4')
    with pytest.raises(RuntimeError, match='interrupted'):
        audit.generate_line(pos, 'bad', args)
    saved = json.loads((tmp_path / 'lines/p-bad.json').read_text())
    assert len(saved['states']) == 2 and not saved['complete']
    result = audit.generate_line(pos, 'bad', args)
    assert result['complete'] and len(result['states']) == 5
    board = chess.Board()
    for row in result['states']:
        if row['move']:
            board.push_uci(row['move'])
        assert board.fen() == row['fen']
