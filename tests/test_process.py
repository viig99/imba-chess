"""Real shutdown behavior plus thin wiring checks for both command entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path




def _run_driver(tmp_path: Path, fake_main_body: str) -> subprocess.CompletedProcess:
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
from imba_chess.process import main_with_hard_exit


def _fake_main():
{fake_main_body}


main_with_hard_exit(_fake_main)
"""
    )
    # A generous but finite timeout: pre-fix, the hanging-thread scenario
    # would block forever here, so a real timeout firing is itself evidence
    # the fix regressed (subprocess.TimeoutExpired fails the test loudly
    # rather than hanging the test suite).
    return subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_hard_exit_terminates_despite_lingering_non_daemon_thread(tmp_path):
    result = _run_driver(
        tmp_path,
        "    import threading\n"
        "    # A genuine non-daemon thread, blocked forever -- exactly the\n"
        "    # condition that made the real process hang at interpreter\n"
        "    # shutdown pre-fix (Py_FinalizeEx joins non-daemon threads).\n"
        "    threading.Thread(target=lambda: threading.Event().wait(), daemon=False).start()\n"
        "    raise RuntimeError('synthetic crash for hard-exit test')\n",
    )
    assert result.returncode == 1
    assert "synthetic crash for hard-exit test" in result.stderr
    assert "RuntimeError" in result.stderr


def test_hard_exit_wrapper_lets_systemexit_pass_through_unchanged(tmp_path):
    # argparse (and any explicit sys.exit()) must keep its own exit code,
    # not get clobbered to 1 by the hard-exit path.
    result = _run_driver(tmp_path, "    raise SystemExit(7)\n")
    assert result.returncode == 7


def test_hard_exit_terminates_on_success_despite_lingering_non_daemon_thread(
    tmp_path,
):
    # The crash path was hardened; the success path was not. A run that
    # completes all its work, writes its output and returns normally still
    # goes through CPython's ordinary shutdown, which blocks joining
    # non-daemon threads -- observed on a real 20-game rollout that finished
    # and then sat for 11 minutes holding 4 GB of GPU. A finished run must
    # release its GPU/shard slot as decisively as a crashed one.
    result = _run_driver(
        tmp_path,
        "    import threading\n"
        "    threading.Thread(target=lambda: threading.Event().wait(), daemon=False).start()\n"
        "    print('work complete')\n",
    )
    assert result.returncode == 0
    assert "work complete" in result.stdout


def test_scripts_delegate_to_shared_exit_helper(monkeypatch):
    import importlib.util
    import imba_chess.process as process

    for name in ("eval_vs_stockfish", "generate_search_rollouts"):
        path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"exit_wiring_{name}", path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        called = []
        monkeypatch.setattr(process, "main_with_hard_exit", called.append)
        def replacement():
            pass
        monkeypatch.setattr(module, "main", replacement)
        module._main_with_hard_exit_on_crash()
        assert called == [replacement]
