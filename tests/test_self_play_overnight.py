from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.runtime import run_lock
import scripts.run_self_play_overnight as overnight


@pytest.mark.parametrize("incomplete", [False, True])
def test_overnight_handoff_and_resume(tmp_path, monkeypatch, incomplete):
    run = tmp_path / "run"
    run.mkdir()
    (run / "actor-000000.pt").write_bytes(b"baseline")
    config, seeds, stockfish = [tmp_path / n for n in ("config", "seeds", "stockfish")]
    for path in (config, seeds, stockfish):
        path.write_bytes(b"input")
    output = tmp_path / "morning"
    deadline = datetime(2026, 9, 13, 12, tzinfo=timezone.utc)
    now = [deadline.timestamp() - 1800]
    monkeypatch.setattr(overnight.time, "time", lambda: now[0])
    monkeypatch.setattr(
        overnight.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    calls = []

    def command(argv, log, stop):
        calls.append(argv)
        if "scripts/run_self_play.py" in argv:
            assert "--resume" in argv and "--defer-confirmation" in argv
            assert argv[argv.index("--concurrent-games") + 1] == "24"
            assert argv[argv.index("--screen-every") + 1] == "3"
            (run / "actor-000001.pt").write_bytes(b"trained")
        else:
            assert now[0] >= deadline.timestamp()
            target = Path(argv[argv.index("--output") + 1])
            # The supervisor must not hold the child evaluator's output lock.
            with run_lock(target.parent):
                result = (
                    {}
                    if incomplete
                    else dict(interval=dict(score=0.5, lower=0.4, upper=0.6))
                )
                atomic_json(target, result)
        return 0

    monkeypatch.setattr(overnight, "run_command", command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overnight",
            "--run",
            str(run),
            "--config",
            str(config),
            "--seeds",
            str(seeds),
            "--output",
            str(output),
            "--until",
            deadline.isoformat(),
            "--concurrent-games",
            "24",
            "--stockfish",
            str(stockfish),
        ],
    )
    overnight.main()
    state = json.loads((output / "progress.json").read_text())
    assert state["phase"] == ("evaluate" if incomplete else "complete")
    assert len(calls) == 4
    assert Path(state["candidate"]["path"]).read_bytes() == b"trained"
    assert Path(state["ckpt34"]["path"]).read_bytes() == b"baseline"
    # Identical search settings, prefix list and pair count across all comparisons.
    for flag in ("--config", "--seeds", "--pairs", "--seconds"):
        assert len({c[c.index(flag) + 1] for c in calls[1:]}) == 1
    incomplete = False
    overnight.main()
    assert json.loads((output / "progress.json").read_text())["phase"] == "complete"
    assert len(calls) == (7 if state["phase"] == "evaluate" else 4)
