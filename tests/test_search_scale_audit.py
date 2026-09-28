import pytest
from scripts.audit_search_scales import metrics, summarize


def test_target_quality_uses_all_mass_and_keeps_mates_out_of_cp_means():
    import math
    result=dict(root_log_priors=[math.log(.75),math.log(.25)],policy=[.25,.75],move_uci='b')
    scores={'a':dict(expectation=.2,cp=-100,mate=None),'b':dict(expectation=.8,cp=100,mate=None)}
    m=metrics(result,['a','b'],scores)
    assert m['target_gain']==pytest.approx(.3)
    assert m['target_cp_gain']==pytest.approx(100)
    assert m['selected_gain']==pytest.approx(.6)
    scores['b'].update(cp=None,mate=3)
    assert metrics(result,['a','b'],scores)['target_cp_gain'] is None


def test_bootstrap_clusters_noise_repeats_by_position():
    rows=[dict(scale=.1,seed_id=seed,metrics=dict(target_gain=gain,selected_gain=gain))
          for seed,gain in [('a',0),('a',1),('b',.5)]]
    m=summarize(rows)['0.1']
    assert m['positions']==2
    assert m['target_gain_95ci']==[.5,.5]
