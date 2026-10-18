"""Public search names and the PUCT presets used in the held-out study.

RPO names select a root policy target, not a different tree traversal.
The runtime receives the canonical algorithm and an explicit PuctConfig.
"""

from dataclasses import replace
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .puct_search import PUCT_ALGORITHMS, PuctConfig


PUCT_METHODS = (*PUCT_ALGORITHMS, "puct_rpo", "puct_variance_rpo")
SEARCH_METHODS = ("gumbel", *PUCT_METHODS, "value_search_halving", "policy", "policy_q")


def search_backend(method: str) -> str:
    if method not in SEARCH_METHODS:
        raise ValueError(f"unsupported search method: {method}")
    return method.removesuffix("_rpo") if method in PUCT_METHODS else method


def puct_config(method: str, *, path: Path | None = None,
                simulations: int | None = None) -> PuctConfig:
    """Resolve a study preset, optionally replaced by a TOML [search] section.

    A config file is authoritative for parameters; it must agree with the
    requested traversal and target policy so labels cannot silently mislead.
    """
    if method not in PUCT_METHODS:
        raise ValueError("--search-config applies only to PUCT methods")
    backend = search_backend(method)
    rpo = method.endswith("_rpo")
    if path is not None:
        with Path(path).open("rb") as handle:
            settings = tomllib.load(handle)["search"]
        config = PuctConfig(**settings)
        if config.algorithm != backend or config.target_policy != ("rpo" if rpo else "visits"):
            raise ValueError("search config algorithm/target_policy does not match the selected method")
    else:
        config = PuctConfig(
            algorithm=backend,
            cpuct=1.0 if backend == "puct" else 1.75,
            variance_c2=3.0 if backend == "puct_v" else 6.0,
            target_policy="rpo" if rpo else "visits",
            rpo_regularization=(0.25 if backend == "puct" else 0.015625) if rpo else 1.0,
        )
    return replace(config, simulations=simulations) if simulations is not None else config
