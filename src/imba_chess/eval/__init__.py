"""imba_chess.eval package.

Lazy re-exports (PEP 562 module `__getattr__`) rather than eager top-level
imports: `metrics.py` and `position_evaluator.py` both import torch, and
eagerly importing them here would make `import imba_chess.eval.<anything>`
-- including `actor_protocol`/`actor_worker`, which MUST stay torch-free for
multiprocess eval actor workers (spawned as separate processes that never
load torch) -- transitively import torch merely by touching this package's
`__init__`, regardless of what the target submodule itself imports. Lazy
attribute resolution keeps `from imba_chess.eval import CachedPositionEvaluator`
working exactly as before while letting `import imba_chess.eval.search` /
`import imba_chess.eval.actor_worker` stay genuinely torch-free.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "create_next_move_evaluator",
    "normalize_topk",
    "NextMoveCrossEntropy",
    "NextMoveTopKAccuracy",
    "NextMoveMRR",
    "NextMoveTokenCount",
    "BatchCount",
    "GameCount",
    "CachedPositionEvaluator",
    "load_hstu_checkpoint",
]

_METRICS_NAMES = {
    "BatchCount",
    "GameCount",
    "NextMoveCrossEntropy",
    "NextMoveMRR",
    "NextMoveTokenCount",
    "NextMoveTopKAccuracy",
    "normalize_topk",
}
_POSITION_EVALUATOR_NAMES = {"CachedPositionEvaluator", "load_hstu_checkpoint"}


def __getattr__(name: str) -> Any:
    if name in _METRICS_NAMES:
        from . import metrics

        return getattr(metrics, name)
    if name in _POSITION_EVALUATOR_NAMES:
        from . import position_evaluator

        return getattr(position_evaluator, name)
    if name == "create_next_move_evaluator":
        try:
            from .ignite_evaluator import create_next_move_evaluator
        except ImportError:  # pragma: no cover - optional runtime dependency
            return None
        return create_next_move_evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
