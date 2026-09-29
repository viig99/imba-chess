from dataclasses import asdict, dataclass, field
import hashlib
from functools import cached_property
import json
from pathlib import Path
import math

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from imba_chess.eval.gumbel_search import GumbelConfig


@dataclass(frozen=True)
class CollectionConfig:
    concurrent_games: int = 8
    root_batch_tokens: int = 1024
    max_game_plies: int = 512
    fresh_positions: int = 4096


@dataclass(frozen=True)
class ReplayConfig:
    window_positions: int = 50000
    flush_games: int = 16
    flush_positions: int = 2048


@dataclass(frozen=True)
class LearningConfig:
    lr: float = 1e-5
    value_weight: float = 1.0
    auxiliary_value_weight: float = 1.0
    # A list trains one auxiliary WDL readout per TD horizon, each weighted
    # by auxiliary_value_weight; the model needs as many auxiliary heads.
    auxiliary_value_lambda: float | tuple[float, ...] = 0.95
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    reuse: float = 2.0
    microbatch_tokens: int = 1024
    # Main value label = (1 - mix) * game result + mix * this ply's search WDL.
    value_search_mix: float = 0.0
    # Microbatches summed per optimizer step; each step spans ~10 games per microbatch.
    gradient_accumulation: int = 1
    policy_surprise_enabled: bool = False
    policy_surprise_fraction: float = 0.5
    policy_surprise_cap: float = 3.0

    @property
    def auxiliary_value_lambdas(self):
        value = self.auxiliary_value_lambda
        return value if isinstance(value, tuple) else (value,)

    def __post_init__(self):
        if not math.isfinite(self.auxiliary_value_weight) or self.auxiliary_value_weight < 0:
            raise ValueError("auxiliary_value_weight must be finite and nonnegative")
        if isinstance(self.auxiliary_value_lambda, (list, tuple)):
            object.__setattr__(self, "auxiliary_value_lambda", tuple(self.auxiliary_value_lambda))
            if not self.auxiliary_value_lambda:
                raise ValueError("auxiliary_value_lambda list must be nonempty")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in self.auxiliary_value_lambdas
        ):
            raise ValueError("auxiliary_value_lambda must be in [0, 1]")
        if not math.isfinite(self.value_weight) or self.value_weight < 0:
            raise ValueError("value_weight must be finite and nonnegative")
        if not math.isfinite(self.value_search_mix) or not 0 <= self.value_search_mix <= 1:
            raise ValueError("value_search_mix must be in [0, 1]")
        if type(self.gradient_accumulation) is not int or self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be a positive integer")
        if type(self.policy_surprise_enabled) is not bool:
            raise ValueError("policy_surprise_enabled must be boolean")
        if not math.isfinite(self.policy_surprise_fraction) or not 0 <= self.policy_surprise_fraction <= 1:
            raise ValueError("policy_surprise_fraction must be in [0, 1]")
        if not math.isfinite(self.policy_surprise_cap) or self.policy_surprise_cap < 1:
            raise ValueError("policy_surprise_cap must be >= 1")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    hours: float = 8.0
    reserve_minutes: float = 75.0
    drain_minutes: float = 15.0
    screen_pairs: int = 50
    confirmation_pairs: int = 250


@dataclass(frozen=True)
class StreamingConfig:
    # Durable block shuffle replaces HF's non-checkpointable shuffle buffer.
    block_rows: int = 10000
    prefetch_blocks: int = 2
    startup_timeout: float = 300.0
    shutdown_timeout: float = 5.0


@dataclass(frozen=True)
class RegretConfig:
    capacity: int = 256
    temperature: float = 0.1
    ema_alpha: float = 0.5

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("regret capacity must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("regret temperature must be finite and positive")
        if not math.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("regret ema_alpha must be in (0, 1]")


@dataclass(frozen=True)
class SelfPlayConfig:
    base_config: str = "config/imba_chess_v4.toml"
    search: GumbelConfig = field(default_factory=GumbelConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    run: RunConfig = field(default_factory=RunConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    regret: RegretConfig | None = None

    @cached_property
    def identifier(self):
        return hashlib.sha256(
            json.dumps(
                dict(
                    settings=asdict(self),
                    base_sha256=hashlib.sha256(
                        Path(self.base_config).read_bytes()
                    ).hexdigest(),
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()


def load_config(path):
    raw = tomllib.loads(Path(path).read_text())
    constructors = dict(
        search=GumbelConfig,
        collection=CollectionConfig,
        replay=ReplayConfig,
        learning=LearningConfig,
        run=RunConfig,
        streaming=StreamingConfig,
        regret=RegretConfig,
    )
    cfg = SelfPlayConfig(
        **{k: constructors[k](**v) if k in constructors else v for k, v in raw.items()}
    )
    for section in (cfg.collection, cfg.replay, cfg.learning):
        if any(not math.isfinite(v) or v <= 0 for k, v in asdict(section).items() if not k.startswith(("policy_surprise_", "auxiliary_value_")) and k not in ("value_weight", "value_search_mix")):
            raise ValueError("stage-2 sizes and learning settings must be positive")
    if not all(math.isfinite(v) for v in asdict(cfg.run).values()):
        raise ValueError("run settings must be finite")
    if not 0 < cfg.run.drain_minutes <= cfg.run.reserve_minutes < cfg.run.hours * 60:
        raise ValueError("require 0 < drain <= reserve < run duration")
    if cfg.run.screen_pairs < 1 or cfg.run.confirmation_pairs < 1:
        raise ValueError("evaluation pair counts must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in asdict(cfg.streaming).values()):
        raise ValueError("streaming sizes and timeouts must be positive")
    if cfg.streaming.block_rows < 4:
        raise ValueError("streaming block_rows must be >= 4")
    if any(
        type(v) is not int
        for v in (cfg.streaming.block_rows, cfg.streaming.prefetch_blocks)
    ):
        raise ValueError("streaming block_rows and prefetch_blocks must be integers")
    if cfg.streaming.prefetch_blocks < 2:
        raise ValueError(
            "streaming prefetch_blocks must be >= 2 for independent bucket cursors"
        )
    return cfg
