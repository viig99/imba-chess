"""Regenerate the unchanged native parity corpus: python -m tests.fixtures.generate_native_positions.

Stores only the 470 selected positions without histories. These tests convert
through FEN; repetition/history is covered separately by replay tests.
"""
import json
from pathlib import Path
from tests.chess_positions import _random_boards


def main():
    data = {}
    for name, count, seed in (("board_state", 150, 31337), ("terminal", 120, 4242), ("projection", 200, 911)):
        boards = _random_boards(count, seed=seed)
        selected = boards[::max(1, len(boards) // count)][:count]
        data[name] = {"seed": seed, "games": count, "fens": [b.fen(en_passant="fen") for b in selected]}
    Path(__file__).with_name("native_positions.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
