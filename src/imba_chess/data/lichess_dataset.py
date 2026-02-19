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
from .torch_iterable import TorchLichessIterableDataset

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
        train_start_month: Optional[str] = None,
        train_end_month: Optional[str] = None,
        val_start_month: Optional[str] = None,
        val_end_month: Optional[str] = None,
        test_start_month: Optional[str] = None,
        test_end_month: Optional[str] = None,
        val_max_games: Optional[int] = None,
        test_max_games: Optional[int] = None,
        cache_dir: Optional[str] = None,
        stream_columns: Optional[Sequence[str]] = None,
        parquet_batch_size: int = 2048,
        max_seq_len: Optional[int] = None,
        return_dataclasses: bool = False,
        board_state_config: Optional[BoardTokenConfig] = None,
    ) -> None:
        self.min_avg_elo = min_avg_elo
        self.split = split
        self.dataset_name = dataset_name
        self.train_start_month = train_start_month
        self.train_end_month = train_end_month
        self.val_start_month = val_start_month
        self.val_end_month = val_end_month
        self.test_start_month = test_start_month
        self.test_end_month = test_end_month
        self.val_max_games = val_max_games
        self.test_max_games = test_max_games
        self.cache_dir = cache_dir
        self.stream_columns = (
            list(stream_columns)
            if stream_columns is not None
            else list(DEFAULT_STREAM_COLUMNS)
        )
        self.parquet_batch_size = parquet_batch_size
        if max_seq_len is not None and max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1 when set, got {max_seq_len}")
        self.max_seq_len = max_seq_len
        self.return_dataclasses = return_dataclasses
        self.board_state_encoder = BoardStateEncoder(board_state_config)
        self._validate_split_settings()

    def stream(
        self,
        *,
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> Iterator[GameRecord | Dict[str, Any]]:
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        data_files = self._temporal_data_files()
        data_files = self._shard_data_files(
            data_files=data_files,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        if not data_files:
            return

        load_kwargs = self._build_load_kwargs(data_files=data_files)

        try:
            rows = load_dataset(**load_kwargs)
        except TypeError:
            load_kwargs.pop("columns", None)
            load_kwargs.pop("batch_size", None)
            rows = load_dataset(**load_kwargs)
        prefiltered = False
        if hasattr(rows, "filter"):
            rows = rows.filter(self._game_filter)
            prefiltered = True

        yield from self.stream_from_rows(
            rows,
            max_games=self._max_games_for_split(),
            assume_prefiltered=prefiltered,
        )

    def _game_filter(self, row: Dict[str, Any]) -> bool:
        """Cheap row-level filter to drop invalid games before PGN parsing."""
        return self._is_valid_game(row)

    def as_torch_iterable(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> TorchLichessIterableDataset:
        return TorchLichessIterableDataset(
            dataset=self,
            rank=rank,
            world_size=world_size,
        )

    def stream_from_rows(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        max_games: Optional[int] = None,
        assume_prefiltered: bool = False,
    ) -> Iterator[GameRecord | Dict[str, Any]]:
        emitted_games = 0
        for row in rows:
            white_elo = parse_elo(row.get("WhiteElo"))
            black_elo = parse_elo(row.get("BlackElo"))
            if white_elo is None or black_elo is None:
                continue

            if not assume_prefiltered and not self._is_valid_game(
                row,
                white_elo=white_elo,
                black_elo=black_elo,
            ):
                continue

            game = self._parse_game_row(
                row,
                white_elo=white_elo,
                black_elo=black_elo,
            )
            if game is None:
                continue

            if self.return_dataclasses:
                yield game
            else:
                yield asdict(game)

            emitted_games += 1
            if max_games is not None and emitted_games >= max_games:
                return

    def _is_valid_game(
        self,
        row: Dict[str, Any],
        *,
        white_elo: Optional[int] = None,
        black_elo: Optional[int] = None,
    ) -> bool:
        if white_elo is None:
            white_elo = parse_elo(row.get("WhiteElo"))
        if black_elo is None:
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

    def _parse_game_row(
        self,
        row: Dict[str, Any],
        *,
        white_elo: Optional[int] = None,
        black_elo: Optional[int] = None,
    ) -> Optional[GameRecord]:
        if white_elo is None:
            white_elo = parse_elo(row.get("WhiteElo"))
        if black_elo is None:
            black_elo = parse_elo(row.get("BlackElo"))
        if white_elo is None or black_elo is None:
            return None

        game = read_pgn_from_row(row)
        if game is None:
            return None

        white_player = to_text(row.get("White"), default="?")
        black_player = to_text(row.get("Black"), default="?")
        result = to_text(row.get("Result"), default="")
        winner_side, winner_player, winner_elo, loser_player, loser_elo = (
            self._resolve_outcome(
                result=result,
                white_player=white_player,
                black_player=black_player,
                white_elo=white_elo,
                black_elo=black_elo,
            )
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
        if not plays:
            return None

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

        while node.variations and (
            self.max_seq_len is None or play_id < self.max_seq_len
        ):
            next_node = node.variations[0]
            move = next_node.move
            active_color = board.turn
            active_color_text = "white" if active_color == chess.WHITE else "black"
            active_player = (
                white_player if active_color == chess.WHITE else black_player
            )
            active_elo = white_elo if active_color == chess.WHITE else black_elo
            opponent_player = (
                black_player if active_color == chess.WHITE else white_player
            )
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
    ) -> tuple[
        Optional[str], Optional[str], Optional[int], Optional[str], Optional[int]
    ]:
        if result == "1-0":
            return "white", white_player, white_elo, black_player, black_elo
        if result == "0-1":
            return "black", black_player, black_elo, white_player, white_elo
        return None, None, None, None, None

    def _build_load_kwargs(self, *, data_files: list[str]) -> Dict[str, Any]:
        return {
            "path": "parquet",
            "data_files": {"train": data_files},
            "split": "train",
            "streaming": True,
            "cache_dir": self.cache_dir,
            "columns": self.stream_columns,
            "batch_size": self.parquet_batch_size,
        }

    @staticmethod
    def _shard_data_files(
        *,
        data_files: list[str],
        shard_id: Optional[int],
        num_shards: Optional[int],
    ) -> list[str]:
        if shard_id is None and num_shards is None:
            return data_files
        if shard_id is None or num_shards is None:
            raise ValueError("shard_id and num_shards must both be set or both be None")
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
        return data_files[shard_id::num_shards]

    def _validate_split_settings(self) -> None:
        if self.split.lower() not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        if self.val_max_games is not None and self.val_max_games < 1:
            raise ValueError("val_max_games must be >= 1 when set")
        if self.test_max_games is not None and self.test_max_games < 1:
            raise ValueError("test_max_games must be >= 1 when set")

    def _max_games_for_split(self) -> Optional[int]:
        split_name = self.split.lower()
        if split_name == "val":
            return self.val_max_games
        if split_name == "test":
            return self.test_max_games
        return None

    def _month_window_for_split(self) -> tuple[str, str]:
        split_name = self.split.lower()
        if split_name == "train":
            return self._require_month_window(
                self.train_start_month, self.train_end_month, "train"
            )
        if split_name == "val":
            return self._require_month_window(
                self.val_start_month, self.val_end_month, "val"
            )
        if split_name == "test":
            return self._require_month_window(
                self.test_start_month, self.test_end_month, "test"
            )
        raise ValueError("split must be one of {'train', 'val', 'test'}")

    @staticmethod
    def _require_month_window(
        start_month: Optional[str], end_month: Optional[str], split_name: str
    ) -> tuple[str, str]:
        if not start_month or not end_month:
            raise ValueError(
                f"temporal split for {split_name!r} requires start and end month"
            )
        return start_month, end_month

    def _temporal_data_files(self) -> list[str]:
        start_month, end_month = self._month_window_for_split()
        start_index = self._parse_month_index(start_month)
        end_index = self._parse_month_index(end_month)
        self._validate_month_range(start_month, end_month)

        # Newest month first so recent games appear first in the stream.
        files: list[str] = []
        for month_index in range(end_index, start_index - 1, -1):
            year = month_index // 12
            month = (month_index % 12) + 1
            files.append(
                f"hf://datasets/{self.dataset_name}/data/year={year:04d}/month={month:02d}/*.parquet"
            )
        return files

    @staticmethod
    def _parse_month_index(value: str) -> int:
        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid month value {value!r}, expected YYYY-MM")
        year_text, month_text = parts
        try:
            year = int(year_text)
            month = int(month_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid month value {value!r}, expected YYYY-MM"
            ) from exc
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month value {value!r}, month must be 01..12")
        return (year * 12) + (month - 1)

    def _validate_month_range(self, start_month: str, end_month: str) -> None:
        if self._parse_month_index(start_month) > self._parse_month_index(end_month):
            raise ValueError(
                f"Invalid month range: start {start_month!r} is after end {end_month!r}"
            )
