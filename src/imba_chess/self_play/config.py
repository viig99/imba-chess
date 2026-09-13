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
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    reuse: float = 2.0
    microbatch_tokens: int = 1024


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    hours: float = 8.0
    reserve_minutes: float = 75.0
    drain_minutes: float = 15.0
    screen_pairs: int = 50
    confirmation_pairs: int = 250


@dataclass(frozen=True)
class SelfPlayConfig:
    base_config: str = "config/imba_chess_v4.toml"
    search: GumbelConfig = field(default_factory=GumbelConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    run: RunConfig = field(default_factory=RunConfig)

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
    )
    cfg = SelfPlayConfig(
        **{k: constructors[k](**v) if k in constructors else v for k, v in raw.items()}
    )
    for section in (cfg.collection, cfg.replay, cfg.learning):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(section).values()):
            raise ValueError("stage-2 sizes and learning settings must be positive")
    if not all(math.isfinite(v) for v in asdict(cfg.run).values()):
        raise ValueError("run settings must be finite")
    if not 0 < cfg.run.drain_minutes <= cfg.run.reserve_minutes < cfg.run.hours * 60:
        raise ValueError("require 0 < drain <= reserve < run duration")
    if cfg.run.screen_pairs < 1 or cfg.run.confirmation_pairs < 1:
        raise ValueError("evaluation pair counts must be positive")
    return cfg
