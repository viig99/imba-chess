import json

import pytest

from imba_chess.self_play.config import SelfPlayConfig
from imba_chess.self_play.evaluation import evaluate_pair_checkpoints
from imba_chess.self_play.seeds import Seed, source_split
from tests.test_self_play import ScriptRuntime


@pytest.mark.parametrize('noisy', [True, False, None])
def test_evaluation_noise_reaches_both_actors_and_changes_resume_identity(tmp_path, noisy):
    class Runtime(ScriptRuntime):
        executors = {}
        def __init__(self):
            self.noises = []
        def search(self, **kwargs):
            self.noises.append(kwargs.get('noise'))
            gen = super().search(**kwargs)
            try:
                next(gen)
                gen.send(None)
            except StopIteration as stop:
                result = stop.value
            yield from ()
            return result

    a, b = Runtime(), Runtime()
    source = "monitor-source"
    while source_split(source) != "monitor":
        source += "x"
    args = dict(candidate=a, best=b, candidate_id='a', best_id='b',
                seeds=[Seed('s', source, [], 0, 'monitor', 'c')],
                config=SelfPlayConfig(), max_positions=128,
                output=tmp_path / 'result.json', pairs=1)
    result = evaluate_pair_checkpoints(**args, **({} if noisy is None else dict(gumbel_noise=noisy)))
    assert result['score'] == .5
    assert a.noises == b.noises == [None if noisy else 0.] * 4
    identity = json.loads(args['output'].read_text())['identity']
    assert identity['inference']['exploration'] == ('gumbel_noise' if noisy else 'zero_gumbel_noise')
    with pytest.raises(ValueError, match='different checkpoints/protocol'):
        evaluate_pair_checkpoints(**args, gumbel_noise=not noisy)


def test_noise_campaign_compares_matched_complete_opening_pairs():
    from copy import deepcopy
    from scripts.compare_gumbel_eval_noise import comparison
    noisy = dict(identity=dict(pairs=1, inference=dict(exploration='gumbel_noise')),
                 results={'0:1': dict(status='completed', outcome_white=-1, candidate_white=True),
                          '0:0': dict(status='completed', outcome_white=1, candidate_white=False)},
                 interval=dict(score=0.))
    zero = deepcopy(noisy)
    zero['identity']['inference']['exploration'] = 'zero_gumbel_noise'
    for row in zero['results'].values():
        row['outcome_white'] = 0
    zero['interval'] = dict(score=.5)
    result = comparison(noisy, zero)
    assert result['difference'] == result['difference_lower'] == result['difference_upper'] == .5
    zero['identity']['pairs'] = 2
    with pytest.raises(ValueError, match='more than evaluation noise'):
        comparison(noisy, zero)
