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
    # Phase size in games instead of positions (0 = use fresh_positions): a phase
    # stops launching once this many training games completed during it. With
    # carry_over_games the in-flight games finish under the next phase's actor.
    fresh_games: int = 0
    # Collection-worker search precision; training always keeps FP32 weights.
    inference_dtype: str = "float32"
    # Playout cap randomization (KataGo) with fast play-out of decided games. Off by
    # default; when off none of the fields below affect play or the run identity.
    # Each move runs the full [search] with probability full_search_probability, and
    # otherwise a fast search (fast_simulations, fast_top_m, no Gumbel noise). A game
    # is decided once the last decided_plies root values all have |v| >= decided_value;
    # decided positions always get the fast search. Fast moves are played and keep
    # their value targets but carry policy_training_weight 0 (no policy target).
    playout_cap: bool = False
    full_search_probability: float = 0.25
    fast_simulations: int = 64
    fast_top_m: int = 8
    decided_value: float = 0.98
    decided_plies: int = 3
    # Multi-worker collection only. Off: once a phase stops launching, in-flight
    # games finish under the phase's actor (the drain). On: they pause at their
    # next move boundary and resume next phase under the new weights, so a game
    # can span actors; the completing phase's actor and iteration label it, and
    # actor_segments records which actor searched each ply.
    carry_over_games: bool = False

    PLAYOUT_CAP_FIELDS = (
        "playout_cap", "full_search_probability", "fast_simulations", "fast_top_m",
        "decided_value", "decided_plies",
    )
    # Fields a resumed run may change with --previous-config.
    # The phase size only regroups the same training (reuse x positions), so a run
    # may change it too.
    RESUMABLE_FIELDS = PLAYOUT_CAP_FIELDS + ("carry_over_games", "fresh_positions", "fresh_games")

    def phase_complete(self, counts):
        """Whether a phase's CollectionMetrics counts reached its size."""
        if self.fresh_games:
            return counts["training_games"] >= self.fresh_games
        return counts["training_positions"] >= self.fresh_positions

    def __post_init__(self):
        if type(self.playout_cap) is not bool:
            raise ValueError("playout_cap must be boolean")
        if type(self.carry_over_games) is not bool:
            raise ValueError("carry_over_games must be boolean")
        if type(self.fresh_games) is not int or self.fresh_games < 0:
            raise ValueError("fresh_games must be a nonnegative integer")
        if type(self.fresh_positions) is not int or self.fresh_positions < 1:
            raise ValueError("fresh_positions must be a positive integer")
        if not math.isfinite(self.full_search_probability) or not 0 < self.full_search_probability <= 1:
            raise ValueError("full_search_probability must be in (0, 1]")
        if any(type(v) is not int or v < 1 for v in (self.fast_simulations, self.fast_top_m, self.decided_plies)):
            raise ValueError("fast_simulations, fast_top_m and decided_plies must be positive integers")
        if not math.isfinite(self.decided_value) or not 0 < self.decided_value <= 1:
            raise ValueError("decided_value must be in (0, 1]")


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
    # Per-step decay of an exponential moving average of the weights, screened
    # next to the raw actor; 0 disables it. Collection always uses raw weights.
    ema_decay: float = 0.0

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
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")


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
        settings = asdict(self)
        if self.collection.inference_dtype == "float32":
            # Runs started before the option existed keep their identity.
            del settings["collection"]["inference_dtype"]
        if not self.collection.playout_cap:
            for key in CollectionConfig.PLAYOUT_CAP_FIELDS:
                del settings["collection"][key]
        if not self.collection.carry_over_games:
            del settings["collection"]["carry_over_games"]
        # Exactly one phase size is in effect; runs from before fresh_games keep
        # their identity.
        if self.collection.fresh_games:
            del settings["collection"]["fresh_positions"]
        else:
            del settings["collection"]["fresh_games"]
        if self.learning.ema_decay == 0:
            del settings["learning"]["ema_decay"]
        return hashlib.sha256(
            json.dumps(
                dict(
                    settings=settings,
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
    if cfg.collection.inference_dtype not in ("float32", "bfloat16"):
        raise ValueError("collection.inference_dtype must be float32 or bfloat16")
    if cfg.collection.playout_cap and (
        cfg.collection.fast_simulations >= cfg.search.simulations
        or cfg.collection.fast_top_m > cfg.search.top_m
    ):
        raise ValueError("the fast search must be smaller than the full [search]")
    for section in (cfg.collection, cfg.replay, cfg.learning):
        if any(not math.isfinite(v) or v <= 0 for k, v in asdict(section).items() if not k.startswith(("policy_surprise_", "auxiliary_value_")) and k not in ("value_weight", "value_search_mix", "inference_dtype", "ema_decay", *CollectionConfig.RESUMABLE_FIELDS)):
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
