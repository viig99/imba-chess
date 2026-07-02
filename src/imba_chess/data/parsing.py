from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import chess.pgn

CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d{2}):(\d{2}(?:\.\d+)?)\]")


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


def parse_clk_seconds(comment: str) -> Optional[float]:
    match = CLK_RE.search(comment or "")
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def normalize_date(value: Any) -> str:
    text = to_text(value, default="")
    if not text:
        return "????.??.??"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}.{text[4:6]}.{text[6:]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text.replace("-", ".")
    return text


def pgn_header_value(value: Any) -> str:
    text = to_text(value, default="?")
    text = text.replace('"', "'")
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def build_pgn_text(row: Dict[str, Any]) -> str:
    result = to_text(row.get("Result"), default="*")
    headers = {
        "Event": row.get("Event"),
        "Site": row.get("Site"),
        "Date": normalize_date(row.get("UTCDate") or row.get("Date")),
        "Round": row.get("Round"),
        "White": row.get("White"),
        "Black": row.get("Black"),
        "Result": result,
        "UTCDate": row.get("UTCDate"),
        "UTCTime": row.get("UTCTime"),
        "WhiteElo": row.get("WhiteElo"),
        "BlackElo": row.get("BlackElo"),
        "TimeControl": row.get("TimeControl"),
        "ECO": row.get("ECO"),
        "Opening": row.get("Opening"),
        "Termination": row.get("Termination"),
    }
    header_lines = [
        f'[{key} "{pgn_header_value(value)}"]' for key, value in headers.items()
    ]
    movetext = to_text(row.get("movetext"), default="")
    if not movetext.endswith(result):
        movetext = f"{movetext} {result}".strip()
    return "\n".join(header_lines) + "\n\n" + movetext + "\n"


def read_pgn_from_row(row: Dict[str, Any]) -> Optional["chess.pgn.Game"]:
    import chess.pgn

    return chess.pgn.read_game(io.StringIO(build_pgn_text(row)))
