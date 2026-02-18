from __future__ import annotations

import dataclasses
from typing import Any, Dict


def as_plain_dict(value: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value

