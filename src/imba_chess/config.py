from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, TypeVar

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG_PATH = Path("config/imba_chess.toml")


@dataclass(frozen=True)
class DatasetConfig:
    min_avg_elo: int = 2000
    split: str = "train"
    dataset_name: str = "Lichess/standard-chess-games"
    train_start_month: Optional[str] = None
    train_end_month: Optional[str] = None
    val_start_month: Optional[str] = None
    val_end_month: Optional[str] = None
    test_start_month: Optional[str] = None
    test_end_month: Optional[str] = None
    val_max_games: Optional[int] = None
    test_max_games: Optional[int] = None
    cache_dir: Optional[str] = None
    parquet_batch_size: int = 2048
    max_seq_len: Optional[int] = None
    return_dataclasses: bool = False


@dataclass(frozen=True)
class VocabConfig:
    path: str = "artifacts/move_vocab_static_uci.json"
    include_unk: bool = False


@dataclass(frozen=True)
class BoardStateConfig:
    en_passant: str = "legal"
    halfmove_max: int = 100
    halfmove_bucket_size: int = 2
    fullmove_max: int = 200
    fullmove_bucket_size: int = 2


@dataclass(frozen=True)
class DataloaderConfig:
    max_tokens_per_batch: int = 6144
    rank: Optional[int] = None
    world_size: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int = 384
    linear_hidden_dim: int = 128
    attention_dim: int = 128
    num_heads: int = 4
    num_layers: int = 6
    dropout: float = 0.1
    max_position_embeddings: int = 6144
    halfmove_vocab_size: int = 128
    fullmove_vocab_size: int = 128
    ignore_index: int = -100
    relative_attention_bias: str = "position"
    label_smoothing: float = 0.0
    elo_weight_min_elo: int = 2200
    elo_weight_max_elo: int = 2800
    elo_loss_weight_alpha: float = 1.0
    elo_loss_weight_strength: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    steps_per_epoch: int = 1_000_000
    eval_every_steps: int = 100_000
    log_every_steps: int = 100
    full_val_every_epochs: int = 1
    fast_val_max_games: int = 10_000
    max_lr: float = 1e-3
    lr_start_factor: float = 0.1
    lr_end_factor: float = 0.5
    onecycle_warmup_fraction_first_epoch: float = 0.1
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    seed: int = 42
    deterministic_eval: bool = True
    eval_num_workers: int = 0
    save_last_every_steps: int = 100_000
    last_checkpoint_keep: int = 1
    optimizer_triton: bool = True
    optimizer_kahan_sum: bool = True
    compile_model: bool = False
    device: str = "auto"
    dtype: str = "bfloat16"
    checkpoint_dir: str = "artifacts/checkpoints"
    checkpoint_keep: int = 3


@dataclass(frozen=True)
class RepoConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    board_state: BoardStateConfig = field(default_factory=BoardStateConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_repo_config(path: str | Path | None = None) -> RepoConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return RepoConfig()

    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    return RepoConfig(
        dataset=_read_section(DatasetConfig, payload.get("dataset", {}), "dataset"),
        board_state=_read_section(BoardStateConfig, payload.get("board_state", {}), "board_state"),
        vocab=_read_section(VocabConfig, payload.get("vocab", {}), "vocab"),
        dataloader=_read_section(DataloaderConfig, payload.get("dataloader", {}), "dataloader"),
        model=_read_section(ModelConfig, payload.get("model", {}), "model"),
        training=_read_section(TrainingConfig, payload.get("training", {}), "training"),
    )


T = TypeVar("T")


def _read_section(section_type: type[T], raw: Any, section_name: str) -> T:
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{section_name}] must be a table")

    allowed = {field.name for field in fields(section_type)}
    unknown = sorted(set(raw.keys()) - allowed)
    if unknown:
        unknown_csv = ", ".join(unknown)
        raise ValueError(f"Unknown keys in [{section_name}]: {unknown_csv}")

    return section_type(**dict(raw))
