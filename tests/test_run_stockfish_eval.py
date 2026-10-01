"""The batched SF runner builds arguments the eval script accepts."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


def test_runner_arguments_parse_in_eval_script(monkeypatch, tmp_path):
    runner = runpy.run_path("scripts/run_stockfish_eval.py")
    evaluator = runpy.run_path("scripts/eval_vs_stockfish.py")
    args = SimpleNamespace(config=Path("config/imba_chess_v4_aux3.toml"), checkpoint=tmp_path / "actor.pt",
                           batch_games=50, elo=2600, stockfish="/usr/bin/stockfish", nodes=40000,
                           concurrent_games=8, seed_base=1042)
    extra = ["--gumbel-simulations", "512", "--gumbel-root-forcing", "--gumbel-forcing-floor",
             "--gumbel-minimax-weight", "0.5"]
    monkeypatch.setattr(sys, "argv", ["eval_vs_stockfish.py", *runner["eval_arguments"](args, 1, tmp_path, extra)])
    parsed = evaluator["_parse_args"]()
    specs = evaluator["_build_segment_specs"](parsed)
    assert len(specs) == 1 and specs[0].elo == 2600 and specs[0].games == 50 and specs[0].limit_strength
    assert parsed.seed == 1043 and parsed.gumbel_simulations == 512 and parsed.gumbel_minimax_weight == 0.5
    assert parsed.gumbel_root_forcing and parsed.gumbel_forcing_floor and parsed.gumbel_value_scale == 0.5 and not parsed.gumbel_rescale_values
