from __future__ import annotations

import io
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import chess
import chess.pgn
from datasets import load_dataset

from .board_state import BoardStateEncoder
from .models import BoardTokenConfig
from .parsing import parse_elo, parse_time_control_seconds, to_text
from .torch_iterable import TorchLichessIterableDataset

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
DEFAULT_STREAM_COLUMNS = [
    "Site",
    "Result",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
    "Termination",
    "movetext",
]


class LichessDataset:
    """Streaming Lichess parser with average-Elo filtering.

    Yields lean plain-dict game records for training:
    {game_id, result, white_elo, black_elo, plays: [{move_uci, state, played_by_elo}]}
    """

    def __init__(
        self,
        min_avg_elo: int = 2000,
        min_time_control_sec: Optional[int] = None,
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
        shuffle_train_month_files_on_start: bool = False,
        train_month_shuffle_seed: Optional[int] = None,
        train_shuffle_buffer_size: int = 0,
        board_state_config: Optional[BoardTokenConfig] = None,
    ) -> None:
        self.min_avg_elo = min_avg_elo
        self.min_time_control_sec = (
            int(min_time_control_sec) if min_time_control_sec is not None else None
        )
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
        self.shuffle_train_month_files_on_start = bool(
            shuffle_train_month_files_on_start
        )
        # Fixed at construction (pre-fork) so all dataloader workers shuffle
        # identically, keeping their strided file shards disjoint.
        self.train_month_shuffle_seed = (
            int(train_month_shuffle_seed)
            if train_month_shuffle_seed is not None
            else random.SystemRandom().randrange(0, 2**63)
        )
        if train_shuffle_buffer_size < 0:
            raise ValueError("train_shuffle_buffer_size must be >= 0")
        self.train_shuffle_buffer_size = int(train_shuffle_buffer_size)
        self.board_state_encoder = BoardStateEncoder(board_state_config)
        self._validate_split_settings()

    def stream(
        self,
        *,
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
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
            try:
                rows = rows.filter(
                    self._game_filter_from_columns,
                    input_columns=["WhiteElo", "BlackElo", "TimeControl"],
                )
            except TypeError:
                rows = rows.filter(self._game_filter)
            prefiltered = True
        if (
            self.split.lower() == "train"
            and self.train_shuffle_buffer_size > 0
            and hasattr(rows, "shuffle")
        ):
            rows = rows.shuffle(
                seed=self.train_month_shuffle_seed,
                buffer_size=self.train_shuffle_buffer_size,
            )

        yield from self.stream_from_rows(
            rows,
            max_games=self._max_games_for_split(),
            assume_prefiltered=prefiltered,
        )

    def _game_filter(self, row: Dict[str, Any]) -> bool:
        """Cheap row-level filter to drop low-ELO/fast games before PGN parsing."""
        return self._game_filter_from_columns(
            row.get("WhiteElo"),
            row.get("BlackElo"),
            row.get("TimeControl"),
        )

    def _game_filter_from_columns(
        self,
        white_elo_raw: Any,
        black_elo_raw: Any,
        time_control_raw: Any,
    ) -> bool:
        white_elo = parse_elo(white_elo_raw)
        black_elo = parse_elo(black_elo_raw)
        if white_elo is None or black_elo is None:
            return False
        if ((white_elo + black_elo) / 2) < self.min_avg_elo:
            return False
        return self._passes_time_control(time_control_raw)

    def _passes_time_control(self, time_control_raw: Any) -> bool:
        if self.min_time_control_sec is None:
            return True
        estimated_sec = parse_time_control_seconds(time_control_raw)
        # Unknown/correspondence time controls fail a strict duration filter.
        if estimated_sec is None:
            return False
        return estimated_sec >= self.min_time_control_sec

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
    ) -> Iterator[Dict[str, Any]]:
        emitted_games = 0
        for row in rows:
            white_elo = parse_elo(row.get("WhiteElo"))
            black_elo = parse_elo(row.get("BlackElo"))
            if white_elo is None or black_elo is None:
                continue
            if not assume_prefiltered:
                if ((white_elo + black_elo) / 2) < self.min_avg_elo:
                    continue
                if not self._passes_time_control(row.get("TimeControl")):
                    continue

            result = to_text(row.get("Result"), default="")
            if result not in VALID_RESULTS:
                continue
            termination = to_text(row.get("Termination"), default="").lower()
            if "abandon" in termination or "abort" in termination:
                continue

            game = self._parse_game_row(
                row, result=result, white_elo=white_elo, black_elo=black_elo
            )
            if game is None:
                continue

            yield game
            emitted_games += 1
            if max_games is not None and emitted_games >= max_games:
                return

    def _parse_game_row(
        self,
        row: Dict[str, Any],
        *,
        result: str,
        white_elo: int,
        black_elo: int,
    ) -> Optional[Dict[str, Any]]:
        movetext = to_text(row.get("movetext"), default="")
        game = chess.pgn.read_game(io.StringIO(movetext))
        # game.errors means the movetext broke mid-parse (illegal/corrupt move);
        # the truncated prefix would carry a result label it never reached.
        if game is None or game.errors:
            return None

        plays = self._extract_plays(game, white_elo=white_elo, black_elo=black_elo)
        if not plays:
            return None

        return {
            "game_id": to_text(row.get("Site"), default=""),
            "result": result,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "plays": plays,
        }

    def _extract_plays(
        self,
        game: chess.pgn.Game,
        *,
        white_elo: int,
        black_elo: int,
    ) -> list[dict[str, Any]]:
        plays: list[dict[str, Any]] = []
        board = game.board()
        node = game

        while node.variations and (
            self.max_seq_len is None or len(plays) < self.max_seq_len
        ):
            node = node.variations[0]
            move = node.move
            state = self.board_state_encoder.encode(board)
            plays.append(
                {
                    "move_uci": move.uci(),
                    # vars() is a shallow, zero-copy view of the frozen
                    # BoardState; consumers must not mutate it.
                    "state": vars(state),
                    "played_by_elo": (
                        white_elo if board.turn == chess.WHITE else black_elo
                    ),
                }
            )
            board.push(move)

        return plays

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
        # NOTE: workers shard over these month-level globs, so num_workers
        # beyond the number of months in the window get empty shards.
        files: list[str] = []
        for month_index in range(end_index, start_index - 1, -1):
            year = month_index // 12
            month = (month_index % 12) + 1
            files.append(
                f"hf://datasets/{self.dataset_name}/data/year={year:04d}/month={month:02d}/*.parquet"
            )
        if self.split.lower() == "train" and self.shuffle_train_month_files_on_start:
            random.Random(self.train_month_shuffle_seed).shuffle(files)
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
