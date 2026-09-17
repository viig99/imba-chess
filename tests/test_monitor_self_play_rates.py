import json
import sys

from scripts import monitor_self_play_rates as monitor


def test_rates_ignore_repeated_collection_reports_and_partial_records(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'metrics.jsonl'
    prior = dict(iteration=0, phase='collect', completed_games=4, searched_positions=100)
    path.write_text(json.dumps(prior) + '\n' + json.dumps(dict(phase='train', steps=10)) + '\n')
    clock = [0.0]
    monkeypatch.setattr(monitor.time, 'monotonic', lambda: clock[0])

    def sleep(seconds):
        with path.open('a') as stream:
            for row in (prior, prior, dict(iteration=1, phase='collect', completed_games=3,
                                           searched_positions=30), dict(phase='train', steps=12)):
                stream.write(json.dumps(row) + '\n')
            stream.write('{"unfinished":')
        clock[0] += seconds

    monkeypatch.setattr(monitor.time, 'sleep', sleep)
    monkeypatch.setattr(sys, 'argv', ['monitor', '--run', str(tmp_path), '--seconds', '60'])
    monitor.main()
    result = json.loads(capsys.readouterr().out)
    assert result['completed_games'] == 3
    assert result['searched_positions'] == 30
    assert result['optimizer_steps'] == 2
    assert result['games_per_hour'] == 180
    assert result['positions_per_hour'] == 1800
    assert result['optimizer_steps_per_hour'] == 120
