from types import SimpleNamespace

import pytest

from scripts.audit_branch_continuations import root_score, rollout_task, summarize, select_positions


def test_outcome_perspective_and_unlabelled_limits():
    game = dict(status='completed', outcome_white=1)
    assert root_score(game, True) == 1
    assert root_score(game, False) == 0
    assert root_score(dict(status='completed', outcome_white=0), False) == .5
    assert root_score(dict(status='unfinished', outcome_white=None), True) is None


def test_forced_reply_can_end_game_without_network_or_losing_history():
    pos = dict(position_id='p', source='source', prefix=['f2f3', 'e7e5'],
               bad='g2g4', good='g2g3', root_white=True)
    manifest = dict(identity={}, reply_labels={'p': dict(best_reply='d8h4')})
    gen = rollout_task(pos, 'bad_forced_reply', 0, manifest, None, None, 513, None)
    with pytest.raises(StopIteration) as done:
        next(gen)
    game = done.value.value
    assert game['outcome_white'] == -1
    assert game['termination'] == 'checkmate'
    assert game['prefix_moves'] == ['f2f3', 'e7e5', 'g2g4', 'd8h4']
    assert game['targets'] == []
    assert root_score(game, True) == 0


def test_summary_compares_complete_position_clusters_not_individual_branches():
    positions = [dict(position_id='a', source='one', root_white=True),
                 dict(position_id='b', source='two', root_white=False)]
    games = []
    for pos in positions:
        for arm, score in [('bad', 0), ('good', 1), ('bad_forced_reply', 0)]:
            for repeat in range(4):
                outcome = (2 * score - 1) * (1 if pos['root_white'] else -1)
                games.append(dict(position_id=pos['position_id'], arm=arm, status='completed',
                                  outcome_white=outcome, termination='checkmate'))
    result = summarize(positions, games, 4, 42)
    assert result['contrasts']['good_minus_bad'] == dict(positions=2, mean=1,
        source_game_bootstrap_95ci=[1, 1], positive=2, tied=0, negative=0)
    games[-5].update(status='unfinished', outcome_white=None, termination='context_limit')
    result = summarize(positions, games, 4, 42)
    assert result['incomplete'] == 1
    assert result['contrasts']['good_minus_bad']['positions'] == 1


def test_selection_is_reproducible_and_one_position_per_source():
    import chess
    screen = {'stockfish': {}}
    records = []
    for i in range(8):
        pid = str(i)
        screen['stockfish'][pid] = dict(position_id=pid, source=str(i//2), prefix=[], fen=chess.STARTING_FEN,
                                        scores={'e2e4': {}, 'd2d4': {}})
        records.append(dict(position_id=pid, cause='value_misorder', chosen='e2e4', best_good_move='d2d4'))
    args = screen, dict(records=records), 3, 42
    selected = select_positions(*args)
    assert selected == select_positions(*args)
    assert len({p['source'] for p in selected}) == 3
