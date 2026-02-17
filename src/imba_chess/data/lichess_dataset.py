from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import chess
import chess.pgn
from datasets import load_dataset

from .board_state import BoardStateEncoder
from .models import (
    BoardTokenConfig,
    GameMetadata,
    GamePlayers,
    GameRecord,
    PlayRecord,
    PlayerInfo,
)
from .parsing import parse_clk_seconds, parse_elo, read_pgn_from_row, to_text

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
DEFAULT_STREAM_COLUMNS = [
    "Event",
    "Site",
    "UTCDate",
    "UTCTime",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "ECO",
    "Opening",
    "Termination",
    "TimeControl",
    "movetext",
]


class LichessDataset:
    """Streaming Lichess parser with average-Elo filtering."""

    def __init__(
        self,
        min_avg_elo: int = 2000,
        split: str = "train",
        dataset_name: str = "Lichess/standard-chess-games",
        cache_dir: Optional[str] = None,
        stream_columns: Optional[Sequence[str]] = None,
        parquet_batch_size: int = 2048,
        return_dataclasses: bool = False,
        board_state_config: Optional[BoardTokenConfig] = None,
    ) -> None:
        self.min_avg_elo = min_avg_elo
        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.stream_columns = (
            list(stream_columns) if stream_columns is not None else list(DEFAULT_STREAM_COLUMNS)
        )
        self.parquet_batch_size = parquet_batch_size
        self.return_dataclasses = return_dataclasses
        self.board_state_encoder = BoardStateEncoder(board_state_config)

    def stream(self) -> Iterator[GameRecord | Dict[str, Any]]:
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        load_kwargs: Dict[str, Any] = {
            "split": self.split,
            "streaming": True,
            "cache_dir": self.cache_dir,
            "columns": self.stream_columns,
            "batch_size": self.parquet_batch_size,
        }

        try:
            rows = load_dataset(self.dataset_name, **load_kwargs)
        except TypeError:
            load_kwargs.pop("columns", None)
            load_kwargs.pop("batch_size", None)
            rows = load_dataset(self.dataset_name, **load_kwargs)

        yield from self.stream_from_rows(rows)

    def stream_from_rows(
        self, rows: Iterable[Dict[str, Any]]
    ) -> Iterator[GameRecord | Dict[str, Any]]:
        for row in rows:
            if not self._is_valid_game(row):
                continue

            game = self._parse_game_row(row)
            if game is None:
                continue

            if self.return_dataclasses:
                yield game
            else:
                yield asdict(game)

    def _is_valid_game(self, row: Dict[str, Any]) -> bool:
        white_elo = parse_elo(row.get("WhiteElo"))
        black_elo = parse_elo(row.get("BlackElo"))
        if white_elo is None or black_elo is None:
            return False

        if ((white_elo + black_elo) / 2) < self.min_avg_elo:
            return False

        result = to_text(row.get("Result"), default="")
        if result not in VALID_RESULTS:
            return False

        termination = to_text(row.get("Termination"), default="").lower()
        if "abandon" in termination or "abort" in termination:
            return False

        movetext = to_text(row.get("movetext"), default="")
        return bool(movetext)

    def _parse_game_row(self, row: Dict[str, Any]) -> Optional[GameRecord]:
        white_elo = parse_elo(row.get("WhiteElo"))
        black_elo = parse_elo(row.get("BlackElo"))
        if white_elo is None or black_elo is None:
            return None

        game = read_pgn_from_row(row)
        if game is None:
            return None

        white_player = to_text(row.get("White"), default="?")
        black_player = to_text(row.get("Black"), default="?")
        result = to_text(row.get("Result"), default="")
        winner_side, winner_player, winner_elo, loser_player, loser_elo = self._resolve_outcome(
            result=result,
            white_player=white_player,
            black_player=black_player,
            white_elo=white_elo,
            black_elo=black_elo,
        )

        metadata = GameMetadata(
            event=to_text(row.get("Event"), default="?"),
            termination=to_text(row.get("Termination"), default="?"),
            eco=to_text(row.get("ECO"), default="?"),
            opening=to_text(row.get("Opening"), default="?"),
            time_control=to_text(row.get("TimeControl"), default="?"),
            utc_date=to_text(row.get("UTCDate"), default="?"),
            utc_time=to_text(row.get("UTCTime"), default="?"),
        )

        players = GamePlayers(
            white=PlayerInfo(name=white_player, elo=white_elo),
            black=PlayerInfo(name=black_player, elo=black_elo),
        )

        plays = self._extract_plays(
            game=game,
            winner_side=winner_side,
            white_player=white_player,
            black_player=black_player,
            white_elo=white_elo,
            black_elo=black_elo,
        )

        return GameRecord(
            game_id=to_text(row.get("Site"), default=""),
            result=result,
            winner_side=winner_side,
            winner_player=winner_player,
            winner_elo=winner_elo,
            loser_player=loser_player,
            loser_elo=loser_elo,
            average_elo=(white_elo + black_elo) / 2,
            num_plies=len(plays),
            players=players,
            metadata=metadata,
            plays=plays,
        )

    def _extract_plays(
        self,
        game: chess.pgn.Game,
        winner_side: Optional[str],
        white_player: str,
        black_player: str,
        white_elo: int,
        black_elo: int,
    ) -> list[PlayRecord]:
        plays: list[PlayRecord] = []
        board = game.board()
        node = game
        last_clock_seconds: Dict[bool, float] = {}
        play_id = 0

        while node.variations:
            next_node = node.variations[0]
            move = next_node.move
            active_color = board.turn
            active_color_text = "white" if active_color == chess.WHITE else "black"
            active_player = white_player if active_color == chess.WHITE else black_player
            active_elo = white_elo if active_color == chess.WHITE else black_elo
            opponent_player = black_player if active_color == chess.WHITE else white_player
            opponent_elo = black_elo if active_color == chess.WHITE else white_elo

            if winner_side is None:
                outcome_for_player = "draw"
            elif winner_side == active_color_text:
                outcome_for_player = "win"
            else:
                outcome_for_player = "loss"

            remaining = parse_clk_seconds(next_node.comment)
            time_taken = None
            if remaining is not None and active_color in last_clock_seconds:
                previous = last_clock_seconds[active_color]
                if previous >= remaining:
                    time_taken = previous - remaining
            if remaining is not None:
                last_clock_seconds[active_color] = remaining

            state = self.board_state_encoder.encode(board)
            san = board.san(move)
            board.push(move)
            play_id += 1

            plays.append(
                PlayRecord(
                    play_id=play_id,
                    move_uci=move.uci(),
                    move_san=san,
                    state=state,
                    time_remaining_seconds=remaining,
                    time_taken_seconds=time_taken,
                    played_by_color=active_color_text,
                    played_by=active_player,
                    played_by_elo=active_elo,
                    opponent_player=opponent_player,
                    opponent_elo=opponent_elo,
                    outcome_for_player=outcome_for_player,
                )
            )
            node = next_node

        return plays

    @staticmethod
    def _resolve_outcome(
        result: str,
        white_player: str,
        black_player: str,
        white_elo: int,
        black_elo: int,
    ) -> tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[int]]:
        if result == "1-0":
            return "white", white_player, white_elo, black_player, black_elo
        if result == "0-1":
            return "black", black_player, black_elo, white_player, white_elo
        return None, None, None, None, None
