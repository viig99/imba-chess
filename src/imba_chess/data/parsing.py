from __future__ import annotations

from typing import Any, Optional


def parse_elo(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.replace(",", "").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def parse_time_control_seconds(value: Any) -> Optional[int]:
    """Estimated game duration for a PGN TimeControl tag like '300+2'.

    Uses the Lichess convention: base seconds + 40 * increment seconds.
    Returns None for missing/correspondence/unparseable values ('-', '?', '').
    """
    text = to_text(value, default="").strip()
    if not text or text in {"-", "?"}:
        return None
    base_text, _, increment_text = text.partition("+")
    try:
        base_sec = int(base_text)
        increment_sec = int(increment_text) if increment_text else 0
    except ValueError:
        return None
    if base_sec < 0 or increment_sec < 0:
        return None
    return base_sec + 40 * increment_sec


def to_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "isoformat"):
        try:
            value = value.isoformat()
        except TypeError:
            value = str(value)
    text = str(value).strip()
    return text if text else default
