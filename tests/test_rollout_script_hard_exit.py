"""Regression coverage for the --concurrent-games GPU-gate crash fix (Bug 2):
an unhandled exception in generate_search_rollouts.main() must actually
terminate the process, even when a genuine non-daemon thread is still alive
and blocked forever -- the exact condition observed on a real GPU run
(process crashed, printed a traceback, then sat futex-parked forever instead
of exiting, hanging a shard slot in remote multi-shard operation).

These tests run the fix (`_main_with_hard_exit_on_crash`) in a real
subprocess: os._exit(1) cannot be exercised in-process without killing the
pytest worker itself, and the whole point of the fix is bypassing CPython's
normal interpreter-shutdown thread-join, which only a real process boundary
can actually prove. No GPU/model/checkpoint involved -- `main` is
monkeypatched to a synthetic crash before the wrapper is invoked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_search_rollouts.py"


def _run_driver(tmp_path: Path, fake_main_body: str) -> subprocess.CompletedProcess:
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "gsr_hard_exit_test", {str(_SCRIPT_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _fake_main():
{fake_main_body}


module.main = _fake_main
module._main_with_hard_exit_on_crash()
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


def test_hard_exit_terminates_on_plain_exception_with_no_extra_threads(tmp_path):
    result = _run_driver(tmp_path, "    raise ValueError('plain synthetic crash')\n")
    assert result.returncode == 1
    assert "plain synthetic crash" in result.stderr


def test_hard_exit_wrapper_lets_systemexit_pass_through_unchanged(tmp_path):
    # argparse (and any explicit sys.exit()) must keep its own exit code,
    # not get clobbered to 1 by the hard-exit path.
    result = _run_driver(tmp_path, "    raise SystemExit(7)\n")
    assert result.returncode == 7


def test_hard_exit_wrapper_is_transparent_on_success(tmp_path):
    result = _run_driver(tmp_path, "    pass\n")
    assert result.returncode == 0
