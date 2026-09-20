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


# Exact-byte aliases for the shipped evaluation-only config migration. This
# preserves existing training resume identities without accepting changes to
# training settings or changing any checkpoint/state serialization format.
_BASE_CONFIG_IDENTITIES = {
    "6e6a2ff15ea643eeed82ef2c78d5133631d4110cfd73fdd4918269105a935d00": "a369dbc762b1af34f333b2df3fb261ab1b7ad7c70d4b19cac0e94924eed59445",
    "92a0bc44b798ed5e2d018152882b179bd259366753f93bbbc16454411f5c4077": "2501b10fd1b0b7c31f84ad538e846479d15f6ea859ef730e70111d80bc6d3343",
    "7dc8abc056195c5dc879132a648395b8ba17da72069d28420cf899986ce8b20c": "8100c8190faa6602d301cb9e77f8c8aad4e85e77f66a5401205650e0f539469c",
    "fe9645250e3a0124c75a38f2429d1f2f243e710fe4dc68d8cea4380ec1f2afbb": "7552807580e97193780d60fa5a6fcd46843ddbe861287386cc0e33ac66b2f996",
}


def _base_config_identity(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _BASE_CONFIG_IDENTITIES.get(digest, digest)


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
    detach_value_features: bool = True
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    reuse: float = 2.0
    microbatch_tokens: int = 1024
    policy_surprise_enabled: bool = False
    policy_surprise_fraction: float = 0.5
    policy_surprise_cap: float = 3.0

    def __post_init__(self):
        if type(self.detach_value_features) is not bool:
            raise ValueError("detach_value_features must be boolean")
        if not math.isfinite(self.value_weight) or self.value_weight < 0:
            raise ValueError("value_weight must be finite and nonnegative")
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
class SelfPlayConfig:
    base_config: str = "config/imba_chess_v4.toml"
    search: GumbelConfig = field(default_factory=GumbelConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    run: RunConfig = field(default_factory=RunConfig)
    streaming: StreamingConfig | None = None

    @cached_property
    def identifier(self):
        settings = asdict(self)
        if not self.learning.detach_value_features:
            settings["learning"].pop("detach_value_features")
        defaults = LearningConfig()
        keys = ("policy_surprise_enabled", "policy_surprise_fraction", "policy_surprise_cap")
        if all(getattr(self.learning, k) == getattr(defaults, k) for k in keys):
            for key in keys:
                settings["learning"].pop(key)
        if self.streaming is None:
            settings.pop("streaming")  # Preserve existing run identities.
        return hashlib.sha256(
            json.dumps(
                dict(
                    settings=settings,
                    base_sha256=_base_config_identity(self.base_config),
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
    )
    cfg = SelfPlayConfig(
        **{k: constructors[k](**v) if k in constructors else v for k, v in raw.items()}
    )
    for section in (cfg.collection, cfg.replay, cfg.learning):
        if any(not math.isfinite(v) or v <= 0 for k, v in asdict(section).items() if not k.startswith("policy_surprise_") and k not in ("value_weight", "detach_value_features")):
            raise ValueError("stage-2 sizes and learning settings must be positive")
    if not all(math.isfinite(v) for v in asdict(cfg.run).values()):
        raise ValueError("run settings must be finite")
    if not 0 < cfg.run.drain_minutes <= cfg.run.reserve_minutes < cfg.run.hours * 60:
        raise ValueError("require 0 < drain <= reserve < run duration")
    if cfg.run.screen_pairs < 1 or cfg.run.confirmation_pairs < 1:
        raise ValueError("evaluation pair counts must be positive")
    if cfg.streaming is not None:
        if any(not math.isfinite(v) or v <= 0 for v in asdict(cfg.streaming).values()):
            raise ValueError("streaming sizes and timeouts must be positive")
        if cfg.streaming.block_rows < 4:
            raise ValueError("streaming block_rows must be >= 4")
        if any(
            type(v) is not int
            for v in (cfg.streaming.block_rows, cfg.streaming.prefetch_blocks)
        ):
            raise ValueError(
                "streaming block_rows and prefetch_blocks must be integers"
            )
        if cfg.streaming.prefetch_blocks < 2:
            raise ValueError(
                "streaming prefetch_blocks must be >= 2 for independent bucket cursors"
            )
    return cfg
