#!/usr/bin/env python3
"""Stockfish node-budget calibration probe.

Replays the model-vs-Stockfish game loop from `scripts/eval_vs_stockfish.py`
(`_run_segment`, in miniature) for a handful of games at the CURRENT
time-based Stockfish settings (config's `stockfish_time_sec`, default
0.05s/move, with UCI_Elo set from `--stockfish-elo`), and records how many
nodes Stockfish actually searched on every one of its moves. The resulting
distribution is used to pick a `nodes=` budget for future evals that
approximates today's time-based strength without depending on wall-clock
(which is noisy across machines/load).

Nodes-source survey (python-chess 1.11.2, see chess/engine.py):
  - `SimpleEngine.play(board, limit, info=chess.engine.INFO_ALL)` accepts
    `info=` and returns a `PlayResult` whose `.info` dict is populated from
    the last parsed UCI `info` line before `bestmove` -- `_parse_uci_info`
    populates `info["nodes"]` (chess/engine.py line ~2622) whenever
    Stockfish reports a `nodes` token, which every modern build does for
    both `go movetime` and `go depth`/`go nodes` searches. This means
    `result.info.get("nodes")` is populated on `engine.play` itself; a
    separate `engine.analyse` call is unnecessary. We use the play-info
    path (PLAY_INFO_NODES_PATH below) and fall back to a same-position
    `engine.analyse` call only in the (unexpected) case a move's play-info
    is missing `nodes`, so the probe stays robust to engines that omit it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch
from tqdm.auto import tqdm

from imba_chess.config import load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import load_or_create_static_move_vocab
from imba_chess.eval.position_evaluator import _SequenceHistory, load_hstu_checkpoint
from imba_chess.eval.search import HalvingConfig

DEFAULT_CALIBRATION_CONFIG_PATH = Path("config/imba_chess_exit_full.toml")


def _load_eval_vs_stockfish_module():
    """Dynamically load scripts/eval_vs_stockfish.py to reuse its
    `_select_model_move` helper without requiring `scripts/` to be an
    installed/importable package (it isn't -- only `src/imba_chess` is
    packaged per pyproject.toml). Mirrors the loader pattern already used
    by tests/test_eval_vs_stockfish.py.
    """
    script_path = Path(__file__).resolve().with_name("eval_vs_stockfish.py")
    spec = importlib.util.spec_from_file_location("eval_vs_stockfish_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_vs_stockfish.py module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_select_model_move = _load_eval_vs_stockfish_module()._select_model_move

# Which path produced the recorded node counts: "play_info" (the common
# case) or "analyse_fallback" (used only when play-info lacked "nodes").
PLAY_INFO_NODES_PATH = "play_info"
ANALYSE_FALLBACK_NODES_PATH = "analyse_fallback"


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default "linear" method).

    `p` is in [0, 100]. Values are sorted first; the rank position is
    `(n - 1) * p / 100`. When that position falls between two samples, the
    result is the linear interpolation between them. This is the same
    convention `numpy.percentile(..., method="linear")` uses, and it makes
    `percentile(values, 50)` equal the conventional median (average of the
    two middle elements for even-length lists, exact middle for odd).

    Raises ValueError on an empty `values`.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"p must be in [0, 100], got {p}")
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (n - 1) * (p / 100.0)
    lower_index = int(rank // 1)
    upper_index = min(lower_index + 1, n - 1)
    frac = rank - lower_index
    if frac == 0.0:
        return float(sorted_values[lower_index])
    return float(
        sorted_values[lower_index] * (1.0 - frac) + sorted_values[upper_index] * frac
    )


def median(values: list[float]) -> float:
    """Median via `percentile(values, 50)`; see that docstring for ties."""
    return percentile(values, 50.0)


def round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round `value` to `sig_figs` significant figures, ties rounding away
    from zero (ROUND_HALF_UP on the decimal magnitude).

    Uses `decimal.Decimal` (seeded from `str(value)`) rather than binary
    float arithmetic so that decimal-exact ties (e.g. 125 -> 130 at 2 sig
    figs) resolve deterministically instead of depending on float64
    representation error. `value == 0` returns `0.0` unconditionally
    (significant figures are undefined for zero). Negative values preserve
    sign; only the magnitude is rounded.

    Examples (2 sig figs):
      149500 -> 150000.0   (1.495e5 rounds up to 1.50e5)
      94999  -> 95000.0    (9.4999e4 rounds up to 9.5e4)
      125    -> 130.0      (1.25e2 is an exact tie -> rounds up)
      7      -> 7.0
      0      -> 0.0
    """
    if sig_figs < 1:
        raise ValueError("sig_figs must be >= 1")
    if value == 0:
        return 0.0
    decimal_value = Decimal(str(value))
    sign = -1 if decimal_value < 0 else 1
    magnitude = abs(decimal_value)
    exponent = magnitude.adjusted()
    quantum = Decimal(1).scaleb(exponent - sig_figs + 1)
    rounded = magnitude.quantize(quantum, rounding=ROUND_HALF_UP)
    return float(sign * rounded)


def build_nodes_stats(nodes: list[int]) -> dict[str, Any]:
    """Summarize a flat list of per-move Stockfish node counts.

    Raises ValueError if `nodes` is empty (there is nothing to calibrate
    from -- callers should fail fast rather than emit a bogus 0-node
    recommendation).
    """
    if not nodes:
        raise ValueError("build_nodes_stats() requires at least one node count")
    nodes_median = median([float(n) for n in nodes])
    return {
        "nodes": list(nodes),
        "count": len(nodes),
        "median": nodes_median,
        "p25": percentile([float(n) for n in nodes], 25.0),
        "p75": percentile([float(n) for n in nodes], 75.0),
        "recommended_stockfish_nodes": int(round_to_sig_figs(nodes_median, 2)),
    }


@dataclass
class CalibrationResult:
    nodes: list[int] = field(default_factory=list)
    nodes_source: str = PLAY_INFO_NODES_PATH
    games_played: int = 0
    total_plies: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play the model against time-limited Stockfish and record "
            "Stockfish's reported node counts per move, to calibrate a "
            "nodes= budget equivalent to the current time-based settings."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CALIBRATION_CONFIG_PATH
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--stockfish-elo", type=int, default=2200)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's eval compile setting. Pass --no-compile to "
        "match production eval runs (the nightly always does: the compiled "
        "eval decode path has a known pre-existing Inductor crash).",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_arg]


def _stockfish_move_with_nodes(
    *,
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
) -> tuple[chess.Move, int, str]:
    """Play one Stockfish move, returning (move, nodes, source).

    Primary path: `engine.play(..., info=chess.engine.INFO_ALL)` -- the
    PlayResult's `.info` dict is populated from the engine's last UCI
    `info` line before `bestmove`, which includes `nodes` for every modern
    Stockfish build (confirmed by reading python-chess 1.11.2's
    `_parse_uci_info`; see module docstring). Fallback path: if `nodes` is
    absent from play-info, probe the same (pre-move) position with
    `engine.analyse` under the same limit -- this costs a second search
    but only triggers if the primary path is unexpectedly missing data.
    """
    result = engine.play(board, limit, info=chess.engine.INFO_ALL)
    if result.move is None:
        raise RuntimeError("Stockfish returned no move.")
    nodes = result.info.get("nodes")
    if nodes is not None:
        return result.move, int(nodes), PLAY_INFO_NODES_PATH

    info = engine.analyse(board, limit)
    fallback_nodes = info.get("nodes")
    if fallback_nodes is None:
        raise RuntimeError(
            "Stockfish reported no 'nodes' via play-info or analyse fallback; "
            "cannot calibrate node budget."
        )
    return result.move, int(fallback_nodes), ANALYSE_FALLBACK_NODES_PATH


def _run_calibration_games(
    *,
    engine: chess.engine.SimpleEngine,
    model: torch.nn.Module,
    move_vocab,
    board_state_encoder: BoardStateEncoder,
    games: int,
    max_plies: int,
    engine_limit: chess.engine.Limit,
    device: torch.device,
    dtype: torch.dtype,
    model_move_policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    opening_random_plies: int,
    halving_config: HalvingConfig | None,
) -> CalibrationResult:
    result = CalibrationResult()
    with tqdm(total=games, desc="calibrate-stockfish-nodes", unit="game") as progress:
        for game_idx in range(games):
            board = chess.Board()
            history = _SequenceHistory(
                move_vocab=move_vocab, board_state_encoder=board_state_encoder
            )
            model_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
            plies = 0

            while not board.is_game_over(claim_draw=True):
                if plies >= max_plies:
                    break
                if plies < opening_random_plies:
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    move = random.choice(legal)
                elif board.turn == model_color:
                    batch = history.build_batch_for_current_position(board)
                    move, _debug_info = _select_model_move(
                        model=model,
                        batch=batch,
                        board=board,
                        move_vocab=move_vocab,
                        board_state_encoder=board_state_encoder,
                        device=device,
                        dtype=dtype,
                        policy=model_move_policy,
                        value_rerank_top_k=value_rerank_top_k,
                        value_rerank_lambda=value_rerank_lambda,
                        halving_config=halving_config,
                    )
                else:
                    move, nodes, source = _stockfish_move_with_nodes(
                        engine=engine, board=board, limit=engine_limit
                    )
                    result.nodes.append(nodes)
                    # Only escalate to the fallback label -- never let a
                    # later play-info move overwrite evidence that the
                    # fallback path was needed at least once.
                    if source == ANALYSE_FALLBACK_NODES_PATH:
                        result.nodes_source = source

                history.append_observed_position(board)
                history.record_played_move(move.uci())
                board.push(move)
                plies += 1

            result.games_played += 1
            result.total_plies += plies
            progress.update(1)
            progress.set_postfix({"sf_moves_recorded": len(result.nodes)})
    return result


def main() -> None:
    args = _parse_args()
    if args.games < 1:
        raise ValueError("--games must be >= 1")
    if args.stockfish_elo < 100:
        raise ValueError("--stockfish-elo must be >= 100")

    repo_config = load_repo_config(args.config)
    eval_cfg = repo_config.eval_vs_stockfish

    stockfish_path = Path(eval_cfg.stockfish_path)
    if not stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish binary not found: {stockfish_path}")

    random.seed(int(eval_cfg.seed))
    torch.manual_seed(int(eval_cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(eval_cfg.seed))

    device = _resolve_device(str(eval_cfg.device))
    dtype = _resolve_dtype(str(eval_cfg.dtype))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path, include_unk=repo_config.vocab.include_unk
    )
    board_state_encoder = BoardStateEncoder(repo_config.board_state)

    model_move_policy = str(eval_cfg.model_move_policy)
    model, _compile_enabled = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=bool(
            eval_cfg.compile if args.compile is None else args.compile
        ),
        require_value_head=model_move_policy
        in {"value_rerank", "value_search_d2", "value_search_halving"},
    )

    engine_limit = chess.engine.Limit(time=float(eval_cfg.stockfish_time_sec))
    stockfish_options = {
        "Threads": int(eval_cfg.stockfish_threads),
        "Hash": int(eval_cfg.stockfish_hash_mb),
        "UCI_LimitStrength": True,
        "UCI_Elo": int(args.stockfish_elo),
    }

    halving_config = HalvingConfig(
        budget=int(eval_cfg.search_budget),
        top_m=int(eval_cfg.search_top_m),
        rounds=int(eval_cfg.halving_rounds),
        refutation_top_r=int(eval_cfg.search_refutation_top_r),
        expand_top=int(eval_cfg.search_expand_top),
        max_depth=int(eval_cfg.search_max_depth),
        lam=float(eval_cfg.value_rerank_lambda),
    )

    print("Calibrating Stockfish node budget")
    print(f"  games={args.games}, stockfish_elo={args.stockfish_elo}")
    print(f"  limit={engine_limit}, options={stockfish_options}")
    print(f"  model_move_policy={model_move_policy}, device={device}, dtype={dtype}")

    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        engine.configure(stockfish_options)
        calibration = _run_calibration_games(
            engine=engine,
            model=model,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            games=int(args.games),
            max_plies=int(eval_cfg.max_plies),
            engine_limit=engine_limit,
            device=device,
            dtype=dtype,
            model_move_policy=model_move_policy,
            value_rerank_top_k=int(eval_cfg.value_rerank_top_k),
            value_rerank_lambda=float(eval_cfg.value_rerank_lambda),
            opening_random_plies=int(eval_cfg.opening_random_plies),
            halving_config=halving_config,
        )

    if not calibration.nodes:
        raise RuntimeError(
            "No Stockfish moves were recorded (0 engine turns played) -- "
            "cannot calibrate a node budget."
        )

    stats = build_nodes_stats(calibration.nodes)
    payload = {
        **stats,
        "nodes_source": calibration.nodes_source,
        "games_played": calibration.games_played,
        "total_plies": calibration.total_plies,
        "run_config": {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "stockfish_path": str(stockfish_path),
            "stockfish_elo": int(args.stockfish_elo),
            "stockfish_limit": str(engine_limit),
            "stockfish_options": stockfish_options,
            "model_move_policy": model_move_policy,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[calibrate_stockfish_nodes] count={stats['count']} "
        f"median={stats['median']:.1f} p25={stats['p25']:.1f} p75={stats['p75']:.1f} "
        f"recommended_stockfish_nodes={stats['recommended_stockfish_nodes']} "
        f"(source={calibration.nodes_source}) -> wrote {args.output_json}"
    )


if __name__ == "__main__":
    main()
