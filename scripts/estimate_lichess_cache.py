#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import logging
import math
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.parsing import parse_elo

ELO_COLUMNS = ("WhiteElo", "BlackElo")
LOGGER = logging.getLogger("estimate_lichess_cache")


@dataclass(frozen=True)
class ParquetFileInfo:
    path: str
    size_bytes: int


@dataclass(frozen=True)
class ParquetSample:
    path: str
    parquet_size_bytes: int
    parquet_rows: int
    sampled_rows: int
    sampled_valid_elo_rows: int
    sampled_threshold_rows: int


@dataclass(frozen=True)
class EstimateReport:
    dataset_name: str
    min_avg_elo: int
    source_patterns: list[str]
    total_parquet_files: int
    total_raw_parquet_size_bytes: int
    parquet_size_mean_bytes: float
    parquet_size_median_bytes: float
    parquet_size_min_bytes: int
    parquet_size_max_bytes: int
    sampled_parquet_files: int
    sampled_rows: int
    sampled_valid_elo_rows: int
    sampled_threshold_rows: int
    sampled_threshold_ratio: float
    sampled_threshold_ratio_ci95_low: float
    sampled_threshold_ratio_ci95_high: float
    sampled_rows_per_parquet_limit: int
    estimated_total_rows: float
    estimated_threshold_rows: float
    estimated_threshold_rows_ci95_low: float
    estimated_threshold_rows_ci95_high: float
    estimated_raw_bytes_per_row: float
    estimated_threshold_cache_size_bytes: float
    cache_overhead_factor: float
    cache_dir: Optional[str]
    cache_fs_free_bytes: Optional[int]
    cache_fs_total_bytes: Optional[int]
    cache_fs_estimated_remaining_bytes: Optional[float]
    target_free_gib: Optional[float]
    target_budget_bytes: Optional[float]
    max_storable_games_for_target_budget: Optional[float]
    threshold_search_min: Optional[int]
    threshold_search_max: Optional[int]
    threshold_search_step: Optional[int]
    recommended_min_avg_elo_for_target_budget: Optional[int]
    recommended_estimated_games_for_target_budget: Optional[float]
    recommended_estimated_cache_size_bytes_for_target_budget: Optional[float]
    recommended_status_for_target_budget: Optional[str]
    target_budget_scan: list[dict[str, Any]]
    recommended_time_start_month_for_target_budget_at_current_elo: Optional[str]
    recommended_time_end_month_for_target_budget_at_current_elo: Optional[str]
    recommended_time_estimated_games_for_target_budget_at_current_elo: Optional[float]
    recommended_time_estimated_cache_size_for_target_budget_at_current_elo: Optional[float]
    recommended_time_status_for_target_budget_at_current_elo: Optional[str]
    recommended_joint_min_avg_elo_for_target_budget: Optional[int]
    recommended_joint_start_month_for_target_budget: Optional[str]
    recommended_joint_end_month_for_target_budget: Optional[str]
    recommended_joint_estimated_games_for_target_budget: Optional[float]
    recommended_joint_estimated_cache_size_for_target_budget: Optional[float]
    recommended_joint_status_for_target_budget: Optional[str]
    target_budget_joint_scan: list[dict[str, Any]]
    sample_details: list[dict[str, Any]]


def _configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _with_progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    total: Optional[int],
    desc: str,
    unit: str,
) -> Iterable[Any]:
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast estimator for Lichess parquet size and avg-Elo filtered game counts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to repo config TOML.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default=None,
        help="Which configured split window to inspect. Defaults to [dataset].split.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Override dataset name (defaults to [dataset].dataset_name).",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        default=None,
        help="Optional custom month window start (YYYY-MM).",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        default=None,
        help="Optional custom month window end (YYYY-MM).",
    )
    parser.add_argument(
        "--data-files-glob",
        action="append",
        default=None,
        help=(
            "Optional explicit parquet glob(s), local or hf://, bypassing split/month"
            " discovery."
        ),
    )
    parser.add_argument(
        "--all-months",
        action="store_true",
        help=(
            "Use all available parquet shards under data/year=*/month=* for the"
            " dataset. This is useful for full-corpus estimates."
        ),
    )
    parser.add_argument(
        "--min-avg-elo",
        type=int,
        default=None,
        help="Elo threshold for filter ratio estimate (default [dataset].min_avg_elo).",
    )
    parser.add_argument(
        "--sample-parquets",
        type=int,
        default=12,
        help=(
            "How many parquet files to sample for row-count + Elo prevalence. 0 means"
            " all discovered files."
        ),
    )
    parser.add_argument(
        "--sample-rows-per-parquet",
        type=int,
        default=200_000,
        help="Max streamed rows per sampled parquet for Elo prevalence estimate.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=None,
        help="PyArrow row batch size for sampled Elo scans (default [dataset].parquet_batch_size).",
    )
    parser.add_argument(
        "--hf-open-block-size-mib",
        type=float,
        default=64.0,
        help=(
            "Range-request block size in MiB for hf:// parquet reads."
            " Set to 0 to use filesystem defaults."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parquet sampling.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory used for free-space check (default [dataset].cache_dir).",
    )
    parser.add_argument(
        "--cache-overhead-factor",
        type=float,
        default=1.10,
        help=(
            "Multiplier applied to estimated filtered bytes to include overhead."
            " Example: 1.10 adds 10 percent."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for full JSON report.",
    )
    parser.add_argument(
        "--target-free-gib",
        type=float,
        default=None,
        help=(
            "Optional target cache budget (GiB). When set, recommends budget-feasible"
            " Elo and time-window options."
        ),
    )
    parser.add_argument(
        "--threshold-search-min",
        type=int,
        default=None,
        help=(
            "Minimum avg-Elo threshold scanned for budget recommendations."
            " Defaults to effective --min-avg-elo."
        ),
    )
    parser.add_argument(
        "--threshold-search-max",
        type=int,
        default=3000,
        help="Maximum avg-Elo threshold scanned for budget recommendations.",
    )
    parser.add_argument(
        "--threshold-search-step",
        type=int,
        default=25,
        help="Step size for Elo threshold scanning in budget recommendation mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="CLI logging level.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def _month_to_index(value: str) -> int:
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid month value {value!r}; expected YYYY-MM.")
    year_text, month_text = parts
    try:
        year = int(year_text)
        month = int(month_text)
    except ValueError as exc:
        raise ValueError(f"Invalid month value {value!r}; expected YYYY-MM.") from exc
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month value {value!r}; month must be 01..12.")
    return (year * 12) + (month - 1)


def _build_month_globs(dataset_name: str, start_month: str, end_month: str) -> list[str]:
    start_index = _month_to_index(start_month)
    end_index = _month_to_index(end_month)
    if start_index > end_index:
        raise ValueError(
            f"Invalid month range: start {start_month!r} is after end {end_month!r}."
        )

    globs: list[str] = []
    for month_index in range(end_index, start_index - 1, -1):
        year = month_index // 12
        month = (month_index % 12) + 1
        globs.append(
            f"hf://datasets/{dataset_name}/data/year={year:04d}/month={month:02d}/*.parquet"
        )
    return globs


def _split_window(split: str, config) -> tuple[Optional[str], Optional[str]]:
    split_name = split.lower()
    if split_name == "train":
        return config.dataset.train_start_month, config.dataset.train_end_month
    if split_name == "val":
        return config.dataset.val_start_month, config.dataset.val_end_month
    if split_name == "test":
        return config.dataset.test_start_month, config.dataset.test_end_month
    raise ValueError(f"Unsupported split {split!r}.")


def _discover_patterns(args: argparse.Namespace, config) -> list[str]:
    if args.data_files_glob:
        if args.all_months:
            raise ValueError("--all-months cannot be combined with --data-files-glob.")
        return list(dict.fromkeys(args.data_files_glob))

    dataset_name = args.dataset_name or config.dataset.dataset_name
    if args.all_months:
        if args.start_month or args.end_month:
            raise ValueError("--all-months cannot be combined with --start-month/--end-month.")
        return [f"hf://datasets/{dataset_name}/data/year=*/month=*/*.parquet"]
    if (args.start_month is None) != (args.end_month is None):
        raise ValueError("Both --start-month and --end-month must be set together.")
    if args.start_month and args.end_month:
        return _build_month_globs(dataset_name, args.start_month, args.end_month)

    split = args.split or config.dataset.split
    splits = ("train", "val", "test") if split == "all" else (split,)

    patterns: list[str] = []
    missing_windows: list[str] = []
    for split_name in splits:
        start_month, end_month = _split_window(split_name, config)
        if not start_month or not end_month:
            missing_windows.append(split_name)
            continue
        patterns.extend(_build_month_globs(dataset_name, start_month, end_month))

    if not patterns:
        missing_text = ", ".join(missing_windows) if missing_windows else split
        raise ValueError(
            "No month windows available for the selected split configuration: "
            f"{missing_text}."
        )

    if missing_windows:
        missing_text = ", ".join(missing_windows)
        print(f"Warning: skipped splits with missing month windows: {missing_text}")
    return list(dict.fromkeys(patterns))


def _normalize_hf_path(path: str) -> str:
    if path.startswith("hf://"):
        return path
    return f"hf://{path.lstrip('/')}"


def _discover_hf_parquet_files(
    hf_patterns: Sequence[str],
    *,
    show_progress: bool,
) -> tuple[list[ParquetFileInfo], HfFileSystem]:
    fs = HfFileSystem()
    discovered: dict[str, ParquetFileInfo] = {}

    pattern_iter = _with_progress(
        hf_patterns,
        enabled=show_progress,
        total=len(hf_patterns),
        desc="Resolving HF globs",
        unit="glob",
    )
    for pattern in pattern_iter:
        LOGGER.debug("Scanning HF glob: %s", pattern)
        matches = fs.glob(pattern, detail=True)
        if isinstance(matches, dict):
            items = matches.items()
        else:
            items = ((path, None) for path in matches)

        for raw_path, detail in items:
            path_text = _normalize_hf_path(str(raw_path))
            if not path_text.endswith(".parquet") or path_text in discovered:
                continue
            size_raw = detail.get("size") if isinstance(detail, dict) else None
            if size_raw is None:
                info = fs.info(raw_path)
                size_raw = info.get("size")
            size_bytes = int(size_raw or 0)
            discovered[path_text] = ParquetFileInfo(path=path_text, size_bytes=size_bytes)

    return sorted(discovered.values(), key=lambda item: item.path), fs


def _discover_local_parquet_files(
    local_patterns: Sequence[str],
    *,
    show_progress: bool,
) -> list[ParquetFileInfo]:
    discovered: dict[str, ParquetFileInfo] = {}
    pattern_iter = _with_progress(
        local_patterns,
        enabled=show_progress,
        total=len(local_patterns),
        desc="Resolving local globs",
        unit="glob",
    )
    for pattern in pattern_iter:
        LOGGER.debug("Scanning local glob: %s", pattern)
        for candidate in sorted(glob.glob(pattern)):
            path = Path(candidate).resolve()
            if not path.is_file() or path.suffix != ".parquet":
                continue
            path_text = str(path)
            if path_text in discovered:
                continue
            discovered[path_text] = ParquetFileInfo(
                path=path_text,
                size_bytes=path.stat().st_size,
            )
    return sorted(discovered.values(), key=lambda item: item.path)


def _discover_parquet_files(
    patterns: Sequence[str],
    *,
    show_progress: bool,
) -> tuple[list[ParquetFileInfo], Optional[HfFileSystem]]:
    hf_patterns = [pattern for pattern in patterns if pattern.startswith("hf://")]
    local_patterns = [pattern for pattern in patterns if not pattern.startswith("hf://")]

    files: list[ParquetFileInfo] = []
    hf_fs: Optional[HfFileSystem] = None
    if hf_patterns:
        hf_files, hf_fs = _discover_hf_parquet_files(
            hf_patterns,
            show_progress=show_progress,
        )
        files.extend(hf_files)
    if local_patterns:
        files.extend(
            _discover_local_parquet_files(
                local_patterns,
                show_progress=show_progress,
            )
        )
    files = sorted(files, key=lambda item: item.path)
    deduped = {item.path: item for item in files}
    return sorted(deduped.values(), key=lambda item: item.path), hf_fs


def _to_python(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def _sample_elo_rows(
    parquet_path: str,
    *,
    min_avg_elo: int,
    row_limit: int,
    parquet_batch_size: int,
    hf_fs: Optional[HfFileSystem],
    hf_open_block_size_bytes: Optional[int],
) -> tuple[int, int, int, int, dict[int, int]]:
    parquet_file: pq.ParquetFile
    if parquet_path.startswith("hf://"):
        if hf_fs is None:
            raise ValueError("hf:// parquet path requires HfFileSystem.")
        with hf_fs.open(
            parquet_path,
            "rb",
            block_size=hf_open_block_size_bytes,
        ) as handle:
            parquet_file = pq.ParquetFile(handle)
            parquet_rows = int(parquet_file.metadata.num_rows)
            if row_limit < 1:
                return parquet_rows, 0, 0, 0, {}
            sampled = _sample_elo_rows_from_parquet_file(
                parquet_file,
                min_avg_elo=min_avg_elo,
                row_limit=row_limit,
                parquet_batch_size=parquet_batch_size,
            )
            return parquet_rows, *sampled

    parquet_file = pq.ParquetFile(parquet_path)
    parquet_rows = int(parquet_file.metadata.num_rows)
    if row_limit < 1:
        return parquet_rows, 0, 0, 0, {}
    sampled = _sample_elo_rows_from_parquet_file(
        parquet_file,
        min_avg_elo=min_avg_elo,
        row_limit=row_limit,
        parquet_batch_size=parquet_batch_size,
    )
    return parquet_rows, *sampled


def _sample_elo_rows_from_parquet_file(
    parquet_file: pq.ParquetFile,
    *,
    min_avg_elo: int,
    row_limit: int,
    parquet_batch_size: int,
) -> tuple[int, int, int, dict[int, int]]:
    if parquet_batch_size < 1:
        parquet_batch_size = 1

    sampled_rows = 0
    sampled_valid_elo_rows = 0
    sampled_threshold_rows = 0
    sampled_sum_elo_hist: dict[int, int] = defaultdict(int)

    batches = parquet_file.iter_batches(
        columns=list(ELO_COLUMNS),
        batch_size=parquet_batch_size,
    )
    for batch in batches:
        white_values = batch.column(0).to_pylist()
        black_values = batch.column(1).to_pylist()
        for white_raw, black_raw in zip(white_values, black_values):
            sampled_rows += 1
            if sampled_rows > row_limit:
                return (
                    sampled_rows - 1,
                    sampled_valid_elo_rows,
                    sampled_threshold_rows,
                    dict(sampled_sum_elo_hist),
                )
            white_elo = parse_elo(_to_python(white_raw))
            black_elo = parse_elo(_to_python(black_raw))
            if white_elo is None or black_elo is None:
                continue
            sum_elo = white_elo + black_elo
            sampled_valid_elo_rows += 1
            sampled_sum_elo_hist[sum_elo] += 1
            if (sum_elo / 2.0) >= min_avg_elo:
                sampled_threshold_rows += 1

    return (
        sampled_rows,
        sampled_valid_elo_rows,
        sampled_threshold_rows,
        dict(sampled_sum_elo_hist),
    )


def _choose_sample_parquets(
    parquet_files: Sequence[ParquetFileInfo],
    *,
    sample_parquets: int,
    seed: int,
) -> list[ParquetFileInfo]:
    ordered = sorted(parquet_files, key=lambda item: item.path)
    if sample_parquets <= 0 or sample_parquets >= len(ordered):
        return ordered
    rng = random.Random(seed)
    sampled = rng.sample(ordered, sample_parquets)
    return sorted(sampled, key=lambda item: item.path)


def _ci95_ratio(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    ratio = successes / trials
    variance = ratio * (1.0 - ratio)
    stderr = math.sqrt(max(variance, 0.0) / trials)
    margin = 1.96 * stderr
    return max(0.0, ratio - margin), min(1.0, ratio + margin)


def _human_bytes(value: float) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(float(value))
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    unit_index = 0
    while absolute >= 1024.0 and unit_index < (len(units) - 1):
        absolute /= 1024.0
        unit_index += 1
    return f"{sign}{absolute:.2f} {units[unit_index]}"


def _median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    n = len(ordered)
    middle = n // 2
    if n % 2 == 1:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


MONTH_IN_PATH_RE = re.compile(r"year=(\d{4})/month=(\d{2})")


def _extract_year_month_from_path(path: str) -> Optional[str]:
    match = MONTH_IN_PATH_RE.search(path)
    if match is None:
        return None
    month_key = f"{match.group(1)}-{match.group(2)}"
    try:
        _month_to_index(month_key)
    except ValueError:
        return None
    return month_key


def _successes_from_sum_hist(
    sampled_sum_elo_hist: dict[int, int],
    *,
    min_avg_elo: int,
) -> int:
    required_sum_elo = min_avg_elo * 2
    return sum(
        count
        for sum_elo, count in sampled_sum_elo_hist.items()
        if sum_elo >= required_sum_elo
    )


def _build_report(
    *,
    dataset_name: str,
    source_patterns: Sequence[str],
    min_avg_elo: int,
    parquet_files: Sequence[ParquetFileInfo],
    samples: Sequence[ParquetSample],
    sampled_rows_per_parquet_limit: int,
    cache_overhead_factor: float,
    cache_dir: Optional[str],
    target_free_gib: Optional[float],
    threshold_search_min: Optional[int],
    threshold_search_max: Optional[int],
    threshold_search_step: Optional[int],
    sampled_sum_elo_hist: Optional[dict[int, int]],
) -> EstimateReport:
    parquet_sizes = [item.size_bytes for item in parquet_files]
    total_raw_parquet_size_bytes = sum(parquet_sizes)
    parquet_size_mean_bytes = (
        (total_raw_parquet_size_bytes / len(parquet_sizes)) if parquet_sizes else 0.0
    )
    parquet_size_median_bytes = _median(parquet_sizes)
    parquet_size_min_bytes = min(parquet_sizes) if parquet_sizes else 0
    parquet_size_max_bytes = max(parquet_sizes) if parquet_sizes else 0

    sampled_rows = sum(item.sampled_rows for item in samples)
    sampled_valid_elo_rows = sum(item.sampled_valid_elo_rows for item in samples)
    sampled_threshold_rows = sum(item.sampled_threshold_rows for item in samples)
    sampled_threshold_ratio = (
        (sampled_threshold_rows / sampled_valid_elo_rows)
        if sampled_valid_elo_rows > 0
        else 0.0
    )
    ratio_ci95_low, ratio_ci95_high = _ci95_ratio(
        sampled_threshold_rows, sampled_valid_elo_rows
    )

    sample_parquet_rows = [item.parquet_rows for item in samples]
    avg_rows_per_parquet = (
        (sum(sample_parquet_rows) / len(sample_parquet_rows))
        if sample_parquet_rows
        else 0.0
    )
    estimated_total_rows = avg_rows_per_parquet * len(parquet_files)
    estimated_threshold_rows = estimated_total_rows * sampled_threshold_ratio
    estimated_threshold_rows_ci95_low = estimated_total_rows * ratio_ci95_low
    estimated_threshold_rows_ci95_high = estimated_total_rows * ratio_ci95_high

    sampled_parquet_bytes = sum(item.parquet_size_bytes for item in samples)
    sampled_parquet_rows_total = sum(item.parquet_rows for item in samples)
    if sampled_parquet_rows_total > 0:
        estimated_raw_bytes_per_row = sampled_parquet_bytes / sampled_parquet_rows_total
    else:
        estimated_raw_bytes_per_row = 0.0
    estimated_threshold_cache_size_bytes = (
        estimated_threshold_rows * estimated_raw_bytes_per_row * cache_overhead_factor
    )

    cache_fs_free_bytes: Optional[int] = None
    cache_fs_total_bytes: Optional[int] = None
    cache_fs_estimated_remaining_bytes: Optional[float] = None
    if cache_dir:
        cache_path = Path(cache_dir)
        probe_path = cache_path if cache_path.exists() else cache_path.parent
        if probe_path.exists():
            usage = shutil.disk_usage(probe_path)
            cache_fs_free_bytes = usage.free
            cache_fs_total_bytes = usage.total
            cache_fs_estimated_remaining_bytes = (
                float(cache_fs_free_bytes) - estimated_threshold_cache_size_bytes
            )

    target_budget_bytes: Optional[float] = None
    max_storable_games_for_target_budget: Optional[float] = None
    recommended_min_avg_elo_for_target_budget: Optional[int] = None
    recommended_estimated_games_for_target_budget: Optional[float] = None
    recommended_estimated_cache_size_bytes_for_target_budget: Optional[float] = None
    recommended_status_for_target_budget: Optional[str] = None
    target_budget_scan: list[dict[str, Any]] = []
    recommended_time_start_month_for_target_budget_at_current_elo: Optional[str] = None
    recommended_time_end_month_for_target_budget_at_current_elo: Optional[str] = None
    recommended_time_estimated_games_for_target_budget_at_current_elo: Optional[float] = None
    recommended_time_estimated_cache_size_for_target_budget_at_current_elo: Optional[float] = None
    recommended_time_status_for_target_budget_at_current_elo: Optional[str] = None
    recommended_joint_min_avg_elo_for_target_budget: Optional[int] = None
    recommended_joint_start_month_for_target_budget: Optional[str] = None
    recommended_joint_end_month_for_target_budget: Optional[str] = None
    recommended_joint_estimated_games_for_target_budget: Optional[float] = None
    recommended_joint_estimated_cache_size_for_target_budget: Optional[float] = None
    recommended_joint_status_for_target_budget: Optional[str] = None
    target_budget_joint_scan: list[dict[str, Any]] = []

    if target_free_gib is not None:
        if target_free_gib <= 0:
            raise ValueError("--target-free-gib must be > 0 when set.")
        if threshold_search_min is None or threshold_search_max is None:
            raise ValueError("threshold search bounds are required for budget mode.")
        if threshold_search_step is None or threshold_search_step < 1:
            raise ValueError("--threshold-search-step must be >= 1.")
        if threshold_search_min > threshold_search_max:
            raise ValueError("--threshold-search-min must be <= --threshold-search-max.")

        target_budget_bytes = float(target_free_gib) * (1024.0**3)
        if estimated_raw_bytes_per_row > 0:
            max_storable_games_for_target_budget = (
                target_budget_bytes / (estimated_raw_bytes_per_row * cache_overhead_factor)
            )

        hist = sampled_sum_elo_hist or {}
        hist_total = sum(hist.values())

        month_to_raw_bytes: dict[str, int] = defaultdict(int)
        for parquet_file in parquet_files:
            month_key = _extract_year_month_from_path(parquet_file.path)
            if month_key is None:
                continue
            month_to_raw_bytes[month_key] += parquet_file.size_bytes

        sorted_months_desc = sorted(
            month_to_raw_bytes.keys(),
            key=_month_to_index,
            reverse=True,
        )
        cumulative_month_windows: list[dict[str, Any]] = []
        cumulative_raw_bytes = 0
        newest_month = sorted_months_desc[0] if sorted_months_desc else None
        for month_key in sorted_months_desc:
            cumulative_raw_bytes += month_to_raw_bytes[month_key]
            raw_fraction = (
                cumulative_raw_bytes / total_raw_parquet_size_bytes
                if total_raw_parquet_size_bytes > 0
                else 0.0
            )
            cumulative_month_windows.append(
                {
                    "start_month": month_key,
                    "end_month": newest_month,
                    "cumulative_raw_bytes": float(cumulative_raw_bytes),
                    "estimated_total_rows": estimated_total_rows * raw_fraction,
                }
            )

        if hist_total > 0:
            thresholds = range(
                threshold_search_min,
                threshold_search_max + 1,
                threshold_search_step,
            )
            for threshold in thresholds:
                threshold_successes = _successes_from_sum_hist(
                    hist,
                    min_avg_elo=threshold,
                )
                ratio = threshold_successes / hist_total
                est_rows_full = estimated_total_rows * ratio
                est_cache_full = (
                    est_rows_full * estimated_raw_bytes_per_row * cache_overhead_factor
                )
                fits_full = est_cache_full <= target_budget_bytes
                target_budget_scan.append(
                    {
                        "min_avg_elo": threshold,
                        "estimated_ratio": ratio,
                        "estimated_rows_full_window": est_rows_full,
                        "estimated_cache_size_bytes_full_window": est_cache_full,
                        "fits_target_budget_full_window": fits_full,
                    }
                )
                if fits_full and recommended_min_avg_elo_for_target_budget is None:
                    recommended_min_avg_elo_for_target_budget = threshold
                    recommended_estimated_games_for_target_budget = est_rows_full
                    recommended_estimated_cache_size_bytes_for_target_budget = est_cache_full

                if cumulative_month_windows:
                    best_window_for_threshold: Optional[dict[str, Any]] = None
                    for window in cumulative_month_windows:
                        est_rows_window = window["estimated_total_rows"] * ratio
                        est_cache_window = (
                            est_rows_window
                            * estimated_raw_bytes_per_row
                            * cache_overhead_factor
                        )
                        if est_cache_window <= target_budget_bytes:
                            best_window_for_threshold = {
                                "min_avg_elo": threshold,
                                "start_month": window["start_month"],
                                "end_month": window["end_month"],
                                "estimated_rows": est_rows_window,
                                "estimated_cache_size_bytes": est_cache_window,
                            }
                        else:
                            # cumulative windows only grow as we move to older months
                            break

                    if best_window_for_threshold is not None:
                        target_budget_joint_scan.append(best_window_for_threshold)

            if recommended_min_avg_elo_for_target_budget is not None:
                recommended_status_for_target_budget = "fit_found"
            else:
                recommended_status_for_target_budget = "no_fit_in_search_range"

            if cumulative_month_windows:
                current_elo_ratio = sampled_threshold_ratio
                best_current_elo_window: Optional[dict[str, Any]] = None
                for window in cumulative_month_windows:
                    est_rows_window = window["estimated_total_rows"] * current_elo_ratio
                    est_cache_window = (
                        est_rows_window
                        * estimated_raw_bytes_per_row
                        * cache_overhead_factor
                    )
                    if est_cache_window <= target_budget_bytes:
                        best_current_elo_window = {
                            "start_month": window["start_month"],
                            "end_month": window["end_month"],
                            "estimated_rows": est_rows_window,
                            "estimated_cache_size_bytes": est_cache_window,
                        }
                    else:
                        break

                if best_current_elo_window is not None:
                    recommended_time_start_month_for_target_budget_at_current_elo = (
                        best_current_elo_window["start_month"]
                    )
                    recommended_time_end_month_for_target_budget_at_current_elo = (
                        best_current_elo_window["end_month"]
                    )
                    recommended_time_estimated_games_for_target_budget_at_current_elo = (
                        best_current_elo_window["estimated_rows"]
                    )
                    recommended_time_estimated_cache_size_for_target_budget_at_current_elo = (
                        best_current_elo_window["estimated_cache_size_bytes"]
                    )
                    recommended_time_status_for_target_budget_at_current_elo = "fit_found"
                else:
                    recommended_time_status_for_target_budget_at_current_elo = (
                        "no_fit_in_time_windows"
                    )

                if target_budget_joint_scan:
                    target_budget_joint_scan.sort(
                        key=lambda item: (
                            item["estimated_rows"],
                            -item["min_avg_elo"],
                        ),
                        reverse=True,
                    )
                    best_joint = target_budget_joint_scan[0]
                    recommended_joint_min_avg_elo_for_target_budget = int(
                        best_joint["min_avg_elo"]
                    )
                    recommended_joint_start_month_for_target_budget = str(
                        best_joint["start_month"]
                    )
                    recommended_joint_end_month_for_target_budget = str(
                        best_joint["end_month"]
                    )
                    recommended_joint_estimated_games_for_target_budget = float(
                        best_joint["estimated_rows"]
                    )
                    recommended_joint_estimated_cache_size_for_target_budget = float(
                        best_joint["estimated_cache_size_bytes"]
                    )
                    recommended_joint_status_for_target_budget = "fit_found"
                else:
                    recommended_joint_status_for_target_budget = "no_fit_in_joint_search"
            else:
                recommended_time_status_for_target_budget_at_current_elo = (
                    "month_metadata_unavailable"
                )
                recommended_joint_status_for_target_budget = "month_metadata_unavailable"
        else:
            recommended_status_for_target_budget = "insufficient_sample_data"
            recommended_time_status_for_target_budget_at_current_elo = (
                "insufficient_sample_data"
            )
            recommended_joint_status_for_target_budget = "insufficient_sample_data"

    sample_details = [asdict(item) for item in samples]
    return EstimateReport(
        dataset_name=dataset_name,
        min_avg_elo=min_avg_elo,
        source_patterns=list(source_patterns),
        total_parquet_files=len(parquet_files),
        total_raw_parquet_size_bytes=total_raw_parquet_size_bytes,
        parquet_size_mean_bytes=parquet_size_mean_bytes,
        parquet_size_median_bytes=parquet_size_median_bytes,
        parquet_size_min_bytes=parquet_size_min_bytes,
        parquet_size_max_bytes=parquet_size_max_bytes,
        sampled_parquet_files=len(samples),
        sampled_rows=sampled_rows,
        sampled_valid_elo_rows=sampled_valid_elo_rows,
        sampled_threshold_rows=sampled_threshold_rows,
        sampled_threshold_ratio=sampled_threshold_ratio,
        sampled_threshold_ratio_ci95_low=ratio_ci95_low,
        sampled_threshold_ratio_ci95_high=ratio_ci95_high,
        sampled_rows_per_parquet_limit=sampled_rows_per_parquet_limit,
        estimated_total_rows=estimated_total_rows,
        estimated_threshold_rows=estimated_threshold_rows,
        estimated_threshold_rows_ci95_low=estimated_threshold_rows_ci95_low,
        estimated_threshold_rows_ci95_high=estimated_threshold_rows_ci95_high,
        estimated_raw_bytes_per_row=estimated_raw_bytes_per_row,
        estimated_threshold_cache_size_bytes=estimated_threshold_cache_size_bytes,
        cache_overhead_factor=cache_overhead_factor,
        cache_dir=cache_dir,
        cache_fs_free_bytes=cache_fs_free_bytes,
        cache_fs_total_bytes=cache_fs_total_bytes,
        cache_fs_estimated_remaining_bytes=cache_fs_estimated_remaining_bytes,
        target_free_gib=target_free_gib,
        target_budget_bytes=target_budget_bytes,
        max_storable_games_for_target_budget=max_storable_games_for_target_budget,
        threshold_search_min=threshold_search_min,
        threshold_search_max=threshold_search_max,
        threshold_search_step=threshold_search_step,
        recommended_min_avg_elo_for_target_budget=recommended_min_avg_elo_for_target_budget,
        recommended_estimated_games_for_target_budget=recommended_estimated_games_for_target_budget,
        recommended_estimated_cache_size_bytes_for_target_budget=recommended_estimated_cache_size_bytes_for_target_budget,
        recommended_status_for_target_budget=recommended_status_for_target_budget,
        target_budget_scan=target_budget_scan,
        recommended_time_start_month_for_target_budget_at_current_elo=recommended_time_start_month_for_target_budget_at_current_elo,
        recommended_time_end_month_for_target_budget_at_current_elo=recommended_time_end_month_for_target_budget_at_current_elo,
        recommended_time_estimated_games_for_target_budget_at_current_elo=recommended_time_estimated_games_for_target_budget_at_current_elo,
        recommended_time_estimated_cache_size_for_target_budget_at_current_elo=recommended_time_estimated_cache_size_for_target_budget_at_current_elo,
        recommended_time_status_for_target_budget_at_current_elo=recommended_time_status_for_target_budget_at_current_elo,
        recommended_joint_min_avg_elo_for_target_budget=recommended_joint_min_avg_elo_for_target_budget,
        recommended_joint_start_month_for_target_budget=recommended_joint_start_month_for_target_budget,
        recommended_joint_end_month_for_target_budget=recommended_joint_end_month_for_target_budget,
        recommended_joint_estimated_games_for_target_budget=recommended_joint_estimated_games_for_target_budget,
        recommended_joint_estimated_cache_size_for_target_budget=recommended_joint_estimated_cache_size_for_target_budget,
        recommended_joint_status_for_target_budget=recommended_joint_status_for_target_budget,
        target_budget_joint_scan=target_budget_joint_scan,
        sample_details=sample_details,
    )


def _print_report(report: EstimateReport) -> None:
    print("== Lichess Cache Estimate ==")
    print(f"Dataset: {report.dataset_name}")
    print(f"Threshold: avg_elo >= {report.min_avg_elo}")
    print(f"Source globs: {len(report.source_patterns)}")
    print()

    print("Parquet size stats:")
    print(f"- files: {report.total_parquet_files}")
    print(f"- total raw parquet size: {_human_bytes(report.total_raw_parquet_size_bytes)}")
    print(f"- mean parquet size: {_human_bytes(report.parquet_size_mean_bytes)}")
    print(f"- median parquet size: {_human_bytes(report.parquet_size_median_bytes)}")
    print(f"- min parquet size: {_human_bytes(report.parquet_size_min_bytes)}")
    print(f"- max parquet size: {_human_bytes(report.parquet_size_max_bytes)}")
    print()

    print("Elo prevalence sample:")
    print(f"- sampled parquet files: {report.sampled_parquet_files}")
    print(
        "- sampled rows: "
        f"{report.sampled_rows:,} "
        f"(limit {report.sampled_rows_per_parquet_limit:,} / parquet)"
    )
    print(f"- sampled valid elo rows: {report.sampled_valid_elo_rows:,}")
    print(
        "- sampled threshold rows: "
        f"{report.sampled_threshold_rows:,} "
        f"({report.sampled_threshold_ratio * 100.0:.2f}%)"
    )
    print(
        "- threshold ratio 95% CI: "
        f"{report.sampled_threshold_ratio_ci95_low * 100.0:.2f}% .. "
        f"{report.sampled_threshold_ratio_ci95_high * 100.0:.2f}%"
    )
    print()

    print("Projected totals:")
    print(f"- estimated total rows: {report.estimated_total_rows:,.0f}")
    print(
        "- estimated threshold rows: "
        f"{report.estimated_threshold_rows:,.0f} "
        f"(95% CI {report.estimated_threshold_rows_ci95_low:,.0f} .. "
        f"{report.estimated_threshold_rows_ci95_high:,.0f})"
    )
    print(
        "- estimated threshold cache size: "
        f"{_human_bytes(report.estimated_threshold_cache_size_bytes)} "
        f"(overhead x{report.cache_overhead_factor:.2f})"
    )
    print(
        "- estimated raw bytes / row: "
        f"{report.estimated_raw_bytes_per_row:.2f} B"
    )
    print()

    if report.cache_fs_free_bytes is not None:
        print("Cache filesystem check:")
        print(f"- cache dir: {report.cache_dir}")
        print(f"- free space: {_human_bytes(report.cache_fs_free_bytes)}")
        remaining = report.cache_fs_estimated_remaining_bytes or 0.0
        print(
            "- free space after estimate: "
            f"{_human_bytes(remaining)}"
        )
        if remaining >= 0:
            print("- status: estimated space is sufficient.")
        else:
            print("- status: estimated space is insufficient.")
        print()

    if report.target_budget_bytes is not None:
        print("Target budget recommendation:")
        print(
            "- target budget: "
            f"{_human_bytes(report.target_budget_bytes)} "
            f"({report.target_free_gib:.2f} GiB)"
        )
        if report.max_storable_games_for_target_budget is not None:
            print(
                "- max storable games (overall estimate): "
                f"{report.max_storable_games_for_target_budget:,.0f}"
            )

        print("- Elo-only recommendation (full selected time window):")
        if report.recommended_min_avg_elo_for_target_budget is not None:
            print(
                f"  min_avg_elo={report.recommended_min_avg_elo_for_target_budget}, "
                f"games~{report.recommended_estimated_games_for_target_budget:,.0f}, "
                f"size~{_human_bytes(report.recommended_estimated_cache_size_bytes_for_target_budget)}"
            )
        else:
            print(
                "  no fitting Elo in search range "
                f"[{report.threshold_search_min}, {report.threshold_search_max}]"
            )

        print(f"- Time-only recommendation (at avg_elo>={report.min_avg_elo}):")
        if (
            report.recommended_time_status_for_target_budget_at_current_elo
            == "month_metadata_unavailable"
        ):
            print("  month metadata unavailable in discovered parquet paths")
        elif (
            report.recommended_time_start_month_for_target_budget_at_current_elo
            and report.recommended_time_end_month_for_target_budget_at_current_elo
        ):
            print(
                "  window="
                f"{report.recommended_time_start_month_for_target_budget_at_current_elo}"
                f" -> {report.recommended_time_end_month_for_target_budget_at_current_elo}, "
                f"games~{report.recommended_time_estimated_games_for_target_budget_at_current_elo:,.0f}, "
                f"size~{_human_bytes(report.recommended_time_estimated_cache_size_for_target_budget_at_current_elo)}"
            )
        else:
            print("  no fitting time window at current Elo threshold")

        print("- Joint recommendation (time + Elo):")
        if report.recommended_joint_status_for_target_budget == "month_metadata_unavailable":
            print("  month metadata unavailable in discovered parquet paths")
        elif (
            report.recommended_joint_min_avg_elo_for_target_budget is not None
            and report.recommended_joint_start_month_for_target_budget
            and report.recommended_joint_end_month_for_target_budget
        ):
            print(
                "  "
                f"min_avg_elo={report.recommended_joint_min_avg_elo_for_target_budget}, "
                f"window={report.recommended_joint_start_month_for_target_budget}"
                f" -> {report.recommended_joint_end_month_for_target_budget}, "
                f"games~{report.recommended_joint_estimated_games_for_target_budget:,.0f}, "
                f"size~{_human_bytes(report.recommended_joint_estimated_cache_size_for_target_budget)}"
            )
        else:
            print("  no fitting joint combination in search space")

        print("- status:")
        print(
            "  "
            f"elo={report.recommended_status_for_target_budget}, "
            f"time={report.recommended_time_status_for_target_budget_at_current_elo}, "
            f"joint={report.recommended_joint_status_for_target_budget}"
        )

        if report.target_budget_joint_scan:
            print("- Joint scan preview:")
            for item in report.target_budget_joint_scan[:8]:
                print(
                    "  "
                    f"elo>={int(item['min_avg_elo'])}, "
                    f"window={item['start_month']} -> {item['end_month']}, "
                    f"games~{item['estimated_rows']:,.0f}, "
                    f"size~{_human_bytes(item['estimated_cache_size_bytes'])}"
                )
            omitted = len(report.target_budget_joint_scan) - min(
                len(report.target_budget_joint_scan),
                8,
            )
            if omitted > 0:
                print(f"  ... {omitted} more joint rows omitted")
        print()

    print("Sampled parquet details:")
    for detail in report.sample_details[:10]:
        print(
            "- "
            f"path={detail['path']}, "
            f"size={_human_bytes(detail['parquet_size_bytes'])}, "
            f"rows={detail['parquet_rows']:,}, "
            f"sampled={detail['sampled_rows']:,}, "
            f"hit={detail['sampled_threshold_rows']:,}"
        )
    remaining_rows = len(report.sample_details) - min(len(report.sample_details), 10)
    if remaining_rows > 0:
        print(f"- ... {remaining_rows} more sampled parquets omitted")


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    config = load_repo_config(args.config)
    show_progress = not args.no_progress
    LOGGER.info("Starting parquet estimate.")

    min_avg_elo = (
        int(args.min_avg_elo)
        if args.min_avg_elo is not None
        else int(config.dataset.min_avg_elo)
    )
    parquet_batch_size = (
        int(args.parquet_batch_size)
        if args.parquet_batch_size is not None
        else int(config.dataset.parquet_batch_size)
    )
    hf_open_block_size_bytes = (
        int(args.hf_open_block_size_mib * 1024 * 1024)
        if float(args.hf_open_block_size_mib) > 0.0
        else None
    )
    cache_dir = args.cache_dir if args.cache_dir is not None else config.dataset.cache_dir
    dataset_name = args.dataset_name or config.dataset.dataset_name

    LOGGER.info("Discovering source patterns...")
    patterns = _discover_patterns(args, config)
    LOGGER.info("Resolving parquet files from %d glob(s)...", len(patterns))
    parquet_files, hf_fs = _discover_parquet_files(
        patterns,
        show_progress=show_progress,
    )
    if not parquet_files:
        raise RuntimeError("No parquet files matched the provided split/window/glob.")
    LOGGER.info("Discovered %d parquet file(s).", len(parquet_files))

    sampled_parquets = _choose_sample_parquets(
        parquet_files,
        sample_parquets=args.sample_parquets,
        seed=args.seed,
    )
    LOGGER.info(
        "Sampling %d parquet file(s) for Elo prevalence (row limit=%d each).",
        len(sampled_parquets),
        args.sample_rows_per_parquet,
    )
    samples: list[ParquetSample] = []
    sampled_sum_elo_hist: dict[int, int] = defaultdict(int)
    sample_iter = _with_progress(
        sampled_parquets,
        enabled=show_progress,
        total=len(sampled_parquets),
        desc="Sampling Elo rows",
        unit="parquet",
    )
    for parquet in sample_iter:
        started = time.perf_counter()
        (
            parquet_rows,
            sampled_rows,
            sampled_valid_rows,
            sampled_threshold_rows,
            sample_sum_elo_hist,
        ) = _sample_elo_rows(
            parquet.path,
            min_avg_elo=min_avg_elo,
            row_limit=args.sample_rows_per_parquet,
            parquet_batch_size=parquet_batch_size,
            hf_fs=hf_fs,
            hf_open_block_size_bytes=hf_open_block_size_bytes,
        )
        for sum_elo, count in sample_sum_elo_hist.items():
            sampled_sum_elo_hist[sum_elo] += count
        elapsed_s = time.perf_counter() - started
        LOGGER.debug(
            "Sampled %s | rows=%d sampled=%d valid=%d threshold=%d in %.2fs",
            parquet.path,
            parquet_rows,
            sampled_rows,
            sampled_valid_rows,
            sampled_threshold_rows,
            elapsed_s,
        )
        samples.append(
            ParquetSample(
                path=parquet.path,
                parquet_size_bytes=parquet.size_bytes,
                parquet_rows=parquet_rows,
                sampled_rows=sampled_rows,
                sampled_valid_elo_rows=sampled_valid_rows,
                sampled_threshold_rows=sampled_threshold_rows,
            )
        )

    LOGGER.info("Computing projected totals.")
    threshold_search_min = (
        int(args.threshold_search_min)
        if args.threshold_search_min is not None
        else int(min_avg_elo)
    )
    threshold_search_max = (
        int(args.threshold_search_max)
        if args.target_free_gib is not None
        else None
    )
    threshold_search_step = (
        int(args.threshold_search_step)
        if args.target_free_gib is not None
        else None
    )
    report = _build_report(
        dataset_name=dataset_name,
        source_patterns=patterns,
        min_avg_elo=min_avg_elo,
        parquet_files=parquet_files,
        samples=samples,
        sampled_rows_per_parquet_limit=args.sample_rows_per_parquet,
        cache_overhead_factor=float(args.cache_overhead_factor),
        cache_dir=cache_dir,
        target_free_gib=args.target_free_gib,
        threshold_search_min=threshold_search_min,
        threshold_search_max=threshold_search_max,
        threshold_search_step=threshold_search_step,
        sampled_sum_elo_hist=dict(sampled_sum_elo_hist),
    )
    _print_report(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"Wrote JSON report to {args.output_json}")
        LOGGER.info("Wrote JSON report: %s", args.output_json)


if __name__ == "__main__":
    main()
