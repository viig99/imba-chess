"""KLENT run configuration: one flat TOML table; defaults follow the reference."""

from dataclasses import dataclass, fields
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class KlentConfig:
    # Model architecture, vocab and board encoding come from the repo config.
    base_config: str = "config/imba_chess_v4.toml"
    # "scratch" or initial weights; ignored when a run checkpoint already exists.
    init: str = "scratch"
    # Reference hyperparameters (KazukiOhta/klent main.py).
    alpha: float = 0.03
    beta: float = 0.1
    tau: float = 8.0
    positions_per_iteration: int = 2**21
    total_positions: int = 75_000_000
    batch_tokens: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    # Ours: transformer safety (the reference ResNet runs unclipped).
    grad_clip: float = 1.0
    # Concurrent games; bounded by K/V cache memory (16.8 MB per slot in bf16 at v4 size).
    slots: int = 256
    max_plies: int = 512
    # Iterations that train only the action-value and value heads, with the
    # value-head bootstrap (needed when the policy starts from a checkpoint).
    q_warmup_iterations: int = 0
    warmup_lr: float = 1e-3
    value_weight: float = 1.0
    bootstrap: str = "q"
    inference_dtype: str = "bfloat16"
    # Training autocast, as in supervised training; master weights stay fp32.
    train_dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.bootstrap not in ("q", "value"):
            raise ValueError("bootstrap must be 'q' or 'value'")
        if self.inference_dtype not in ("float32", "bfloat16"):
            raise ValueError("inference_dtype must be float32 or bfloat16")
        if self.train_dtype not in ("float32", "bfloat16"):
            raise ValueError("train_dtype must be float32 or bfloat16")
        positive = ("alpha", "beta", "tau", "positions_per_iteration", "total_positions",
                    "batch_tokens", "lr", "grad_clip", "slots", "max_plies", "warmup_lr")
        if any(not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0 for k in positive):
            raise ValueError(f"{positive} must be positive")
        if self.q_warmup_iterations < 0 or self.value_weight < 0 or self.weight_decay < 0:
            raise ValueError("q_warmup_iterations, value_weight and weight_decay must be >= 0")
        if self.batch_tokens <= self.max_plies:
            raise ValueError("batch_tokens must hold one full game (max_plies + BOS)")


def load_klent_config(path):
    raw = tomllib.loads(Path(path).read_text())
    unknown = set(raw) - {f.name for f in fields(KlentConfig)}
    if unknown:
        raise ValueError(f"unknown KLENT config keys: {sorted(unknown)}")
    return KlentConfig(**raw)
