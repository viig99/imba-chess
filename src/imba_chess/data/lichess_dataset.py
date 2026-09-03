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
from .stockfish_evals import eval_from_comment
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
        local_corpus_path: Optional[str] = None,
        parse_stockfish_evals: bool = False,
        require_stockfish_eval: bool = False,
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
        self.local_corpus_path = local_corpus_path
        self.parse_stockfish_evals = bool(parse_stockfish_evals)
        self.require_stockfish_eval = bool(require_stockfish_eval)
        self.board_state_encoder = BoardStateEncoder(board_state_config)
        self._validate_split_settings()

    def filtered_shuffled_rows(
        self,
        *,
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> tuple[Optional[Iterable[Dict[str, Any]]], bool]:
        """The exact raw-row iterable `stream()` parses, plus its prefiltered flag.

        Split out of `stream()` so a local corpus can be materialized from the
        SAME point in the pipeline -- after `.filter()` and after
        `.shuffle(seed=train_month_shuffle_seed)`. Capturing rows here is what
        makes materialization order-identical, and therefore alignment-safe for
        the `(game_id, ply)` keys that join rollouts to training.
        """
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        data_files = self._temporal_data_files()
        data_files = self._shard_data_files(
            data_files=data_files,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        if not data_files:
            return None, False

        load_kwargs = self._build_load_kwargs(data_files=data_files)

        try:
            rows = load_dataset(**load_kwargs)
        except TypeError:
            load_kwargs.pop("columns", None)
            load_kwargs.pop("batch_size", None)
            rows = load_dataset(**load_kwargs)
        prefiltered = False
        if hasattr(rows, "filter"):
            filter_fn = self._game_filter_from_columns
            filter_columns = ["WhiteElo", "BlackElo", "TimeControl"]
            row_filter_fn = self._game_filter
            if self.require_stockfish_eval:
                # datasets 5.0.0 cannot reliably chain filter() calls when the
                # first FilteredExamplesIterable has features=None. Apply one
                # combined predicate instead, which also avoids an extra lazy
                # iterable layer in every worker.
                filter_fn = self._annotated_game_filter_from_columns
                filter_columns.append("movetext")
                row_filter_fn = self._annotated_game_filter
            try:
                rows = rows.filter(
                    filter_fn,
                    input_columns=filter_columns,
                )
            except TypeError:
                rows = rows.filter(row_filter_fn)
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
        return rows, prefiltered

    def stream(
        self,
        *,
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        if self.local_corpus_path is not None:
            # A materialized corpus is ONE captured stream. Slicing it per
            # dataloader worker would hand each worker a different subsequence
            # than upstream sharding would, silently breaking the
            # (game_id, ply) alignment that rollout and eval value targets
            # depend on -- the same failure mode as the 2026-07-25 sharding
            # bug, which zeroed every rollout target without erroring.
            # Refuse loudly instead.
            if num_shards is not None and int(num_shards) > 1:
                raise ValueError(
                    "dataset.local_corpus_path cannot be sharded (num_shards="
                    f"{num_shards}); set dataloader.num_workers = 0"
                )
            yield from self.stream_local(self.local_corpus_path)
            return
        rows, prefiltered = self.filtered_shuffled_rows(
            shard_id=shard_id, num_shards=num_shards
        )
        if rows is None:
            return

        yield from self.stream_from_rows(
            rows,
            max_games=self._max_games_for_split(),
            assume_prefiltered=prefiltered,
        )

    def stream_local(
        self,
        path: str,
        *,
        max_games: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Replay a corpus materialized by `scripts/materialize_corpus.py`.

        The local file holds the post-filter, post-shuffle rows in stream order,
        so this yields games IDENTICAL to `stream()` for as far as it reaches --
        gated by a bit-identical rollout diff, not by assumption. Rows were
        already filtered upstream, hence `assume_prefiltered=True`.
        """
        import pyarrow.parquet as pq

        def _rows() -> Iterator[Dict[str, Any]]:
            handle = pq.ParquetFile(path)
            for batch in handle.iter_batches(batch_size=self.parquet_batch_size):
                yield from batch.to_pylist()

        yield from self.stream_from_rows(
            _rows(),
            max_games=max_games if max_games is not None else self._max_games_for_split(),
            assume_prefiltered=True,
        )

    def _game_filter(self, row: Dict[str, Any]) -> bool:
        """Cheap row-level filter to drop low-ELO/fast games before PGN parsing."""
        return self._game_filter_from_columns(
            row.get("WhiteElo"),
            row.get("BlackElo"),
            row.get("TimeControl"),
        )

    def _annotated_game_filter(self, row: Dict[str, Any]) -> bool:
        return self._game_filter(row) and self._row_has_stockfish_eval(row)

    def _annotated_game_filter_from_columns(
        self,
        white_elo_raw: Any,
        black_elo_raw: Any,
        time_control_raw: Any,
        movetext_raw: Any,
    ) -> bool:
        return self._game_filter_from_columns(
            white_elo_raw,
            black_elo_raw,
            time_control_raw,
        ) and self._has_stockfish_eval(movetext_raw)

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
            # The remote HF iterable applies this predicate before shuffling
            # and PGN parsing. Keep the same guard here for plain iterables,
            # local corpora, and older datasets implementations whose filter
            # method does not execute lazily.
            if self.require_stockfish_eval and not self._row_has_stockfish_eval(row):
                continue
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
        pending_eval_white: tuple[float | None, int | None] = (None, None)

        while node.variations and (
            self.max_seq_len is None or len(plays) < self.max_seq_len
        ):
            node = node.variations[0]
            move = node.move
            state = self.board_state_encoder.encode(board)
            play = {
                "move_uci": move.uci(),
                # vars() is a shallow, zero-copy view of the frozen
                # BoardState; consumers must not mutate it.
                "state": vars(state),
                "played_by_elo": (
                    white_elo if board.turn == chess.WHITE else black_elo
                ),
            }
            if self.parse_stockfish_evals:
                cp_white, mate_white = pending_eval_white
                sign = 1.0 if board.turn == chess.WHITE else -1.0
                play["eval_cp_stm"] = None if cp_white is None else cp_white * sign
                play["eval_mate_stm"] = (
                    None if mate_white is None else int(mate_white * sign)
                )
                # This node's comment follows this node's move, so it targets
                # the next play/state. The final pending eval is intentionally
                # discarded because the final position has no play token.
                pending_eval_white = eval_from_comment(node.comment or "")
            plays.append(play)
            board.push(move)

        if node.variations:
            # Longer than max_seq_len: reject rather than truncate. A truncated
            # prefix would carry the full game's result as its value label,
            # tell the moves-left head the game ends at the cut, and put
            # progress=1.0 mid-game.
            return []

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

    @staticmethod
    def _has_stockfish_eval(movetext: Any) -> bool:
        return "[%eval" in to_text(movetext, default="")

    @classmethod
    def _row_has_stockfish_eval(cls, row: Dict[str, Any]) -> bool:
        return cls._has_stockfish_eval(row.get("movetext"))

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
