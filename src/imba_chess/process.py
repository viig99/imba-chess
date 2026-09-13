"""Process entrypoint cleanup for commands that can leave native threads alive."""

import os
import sys
import traceback
from collections.abc import Callable


def main_with_hard_exit(main: Callable[[], None]) -> None:
    """Flush completed output and bypass interpreter thread joins on return/crash.

    Model/compiler background threads have hung interpreter shutdown after both
    successful and failed runs. SystemExit retains its normal argparse semantics.
    Call only at a process entrypoint, after application cleanup has run.
    """
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
