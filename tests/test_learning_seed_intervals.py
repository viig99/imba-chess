import pytest
from pathlib import Path


def test_bootstrap_keeps_seeds_and_correlated_positions_together(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from report_learning_seed_intervals import paired_interval

    result = paired_interval([dict(a=0., b=0., c=.2), dict(a=.4, b=.4, c=.6)],
                             dict(a="game1", b="game1", c="game2"))
    assert result["mean_difference"] == pytest.approx(.8 / 3)
    assert result["per_seed_differences"] == pytest.approx([.2 / 3, 1.4 / 3])
    assert result["lower"] == pytest.approx(.2)
    assert result["upper"] == pytest.approx(.4)
    assert result["source_games"] == 2
    with pytest.raises(ValueError, match="identical"):
        paired_interval([dict(a=0), dict(b=0)], dict(a="g", b="g"))
