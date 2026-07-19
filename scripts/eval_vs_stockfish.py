#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator

import chess
import chess.engine
import chess.pgn
import torch

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab
from imba_chess.eval import search
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.engine_pool import EnginePool, make_sf_move_executor
from imba_chess.eval.game_animation import render_game_html
from imba_chess.eval.merged_executors import (
    _make_decode_wave_executor,
    _make_root_eval_executor,
)
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _SequenceHistory,
    _forward_model,
    _project_legal_logits,
    _value_scalar_from_logits,
    load_hstu_checkpoint,
)
from imba_chess.eval.search import (
    EvalRequest,
    HalvingConfig,
    PositionEval,
    select_greedy,
    select_value_rerank,
    select_value_search_d2,
    select_value_search_halving,
)
from tqdm.auto import tqdm


@dataclass
class EvalSummary:
    games: int = 0
    completed_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_as_white: int = 0
    games_as_black: int = 0
    wins_as_white: int = 0
    losses_as_white: int = 0
    draws_as_white: int = 0
    wins_as_black: int = 0
    losses_as_black: int = 0
    draws_as_black: int = 0
    incomplete_games: int = 0
    total_plies: int = 0
    model_turns: int = 0
    legal_moves_total: int = 0
    legal_moves_mapped_total: int = 0
    turns_with_no_vocab_legal_move: int = 0

    @property
    def avg_plies(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_plies / self.games

    @property
    def avg_full_moves(self) -> float:
        return self.avg_plies / 2.0

    @property
    def legal_coverage_rate(self) -> float:
        if self.legal_moves_total == 0:
            return float("nan")
        return self.legal_moves_mapped_total / self.legal_moves_total


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    games: int
    limit_strength: bool
    elo: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained imba-chess model against Stockfish via UCI."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--stockfish-path", type=Path, default=None
    )
    parser.add_argument("--stockfish-time-sec", type=float, default=None)
    parser.add_argument("--stockfish-nodes", type=int, default=None)
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--stockfish-threads", type=int, default=None)
    parser.add_argument("--stockfish-hash-mb", type=int, default=None)
    parser.add_argument(
        "--stockfish-limit-strength",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--stockfish-elo", type=int, default=None)
    parser.add_argument(
        "--ladder-elos",
        type=str,
        default=None,
        help=(
            "Comma-separated Elo ladder for segmented eval, e.g. "
            "'1600,1800,2000,2200,2400,2600,2800'."
        ),
    )
    parser.add_argument(
        "--ladder-games-per-segment",
        type=int,
        default=None,
        help="Games per ladder segment (defaults to --games).",
    )
    parser.add_argument(
        "--include-full-strength-segment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In ladder mode, also run one full-strength Stockfish segment.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default=None,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--model-move-policy",
        choices=["greedy", "value_rerank", "value_search_d2", "value_search_halving"],
        default=None,
        help="Model move selection on legal moves.",
    )
    parser.add_argument(
        "--value-rerank-top-k",
        type=int,
        default=None,
        help="Top-k policy legal moves to evaluate with value_rerank.",
    )
    parser.add_argument(
        "--value-rerank-lambda",
        type=float,
        default=None,
        help="Weight for value_rerank score adjustment.",
    )
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--search-top-m", type=int, default=None)
    parser.add_argument("--halving-rounds", type=int, default=None)
    parser.add_argument("--search-refutation-top-r", type=int, default=None)
    parser.add_argument("--search-expand-top", type=int, default=None)
    parser.add_argument("--search-max-depth", type=int, default=None)
    parser.add_argument(
        "--opening-random-plies",
        type=int,
        default=None,
        help="Uniform random legal moves for first N plies.",
    )
    parser.add_argument(
        "--debug-trace-games",
        type=int,
        default=None,
        help="Number of initial games per segment to print per-turn model debug traces.",
    )
    parser.add_argument(
        "--debug-trace-max-plies",
        type=int,
        default=None,
        help="Max plies per traced game for debug printing.",
    )
    parser.add_argument(
        "--debug-topk",
        type=int,
        default=None,
        help="Top-k legal model moves to print in debug traces.",
    )
    parser.add_argument(
        "--save-games",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save PGN + HTML replay for each debug-traced game.",
    )
    parser.add_argument(
        "--save-games-dir",
        type=Path,
        default=None,
        help="Directory to write saved game PGN/HTML files into.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=None,
        help="Run this many game coroutines concurrently per segment via the "
        "batch scheduler, merging their root-eval/search decode waves and "
        "sf_move engine calls into shared calls each tick, with one "
        "Stockfish engine process per concurrent slot. Default 1 (a value "
        "of 1 merges nothing per tick: byte-identical single-item executor "
        "calls to the pre-scheduler sequential driver).",
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


def _build_engine_limit(args: argparse.Namespace) -> chess.engine.Limit:
    kwargs: dict[str, float | int] = {}
    if args.stockfish_time_sec is not None:
        kwargs["time"] = float(args.stockfish_time_sec)
    if args.stockfish_nodes is not None:
        kwargs["nodes"] = int(args.stockfish_nodes)
    if args.stockfish_depth is not None:
        kwargs["depth"] = int(args.stockfish_depth)
    if not kwargs:
        kwargs["time"] = 0.05
    return chess.engine.Limit(**kwargs)


def _parse_ladder_elos(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("--ladder-elos must contain at least one Elo value")
    if any(v < 100 for v in values):
        raise ValueError("Elo values in --ladder-elos must be >= 100")
    return values


def _select_model_move(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    board: chess.Board,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    device: torch.device,
    dtype: torch.dtype,
    policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    debug_topk: int = 0,
    halving_config: HalvingConfig | None = None,
) -> tuple[chess.Move, dict[str, Any]]:
    output = _forward_model(
        model=model,
        batch=batch,
        device=device,
        dtype=dtype,
        return_kv=policy != "greedy",
    )

    logits = output["logits"][-1]
    legal_logits, legal_moves_with_ids, total_legal, mapped_legal = _project_legal_logits(
        logits=logits,
        board=board,
        move_vocab=move_vocab,
    )
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    evaluator = None
    if policy != "greedy":
        evaluator = CachedPositionEvaluator(
            model=model,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            device=device,
            dtype=dtype,
            prefix_kv=output["kv_caches"],
            prefix_len=int(batch["total_tokens"]),
        )
    rerank_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    halving_rows: list[dict[str, Any]] = []
    if policy == "greedy":
        chosen_index = select_greedy(legal_log_priors)
    elif policy == "value_rerank":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_rerank requires a checkpoint with value head enabled."
            )
        chosen_index, rerank_rows = select_value_rerank(
            evaluator=evaluator,
            root_handle=None,
            board=board,
            legal_moves=legal_moves_with_ids,
            legal_log_priors=legal_log_priors,
            top_k=value_rerank_top_k,
            lam=value_rerank_lambda,
        )
    elif policy == "value_search_d2":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_d2 requires a checkpoint with value head enabled."
            )
        chosen_index, search_rows = select_value_search_d2(
            evaluator=evaluator,
            root_handle=None,
            board=board,
            legal_moves=legal_moves_with_ids,
            legal_log_priors=legal_log_priors,
            top_k=value_rerank_top_k,
            lam=value_rerank_lambda,
        )
    elif policy == "value_search_halving":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_halving requires a checkpoint with value head enabled."
            )
        if halving_config is None:
            raise ValueError("policy=value_search_halving requires halving_config")
        chosen_index, halving_rows = select_value_search_halving(
            evaluator=evaluator,
            root_handle=None,
            board=board,
            legal_moves=legal_moves_with_ids,
            legal_log_priors=legal_log_priors,
            config=halving_config,
        )
    else:
        raise ValueError(f"Unknown model move policy: {policy}")
    debug: dict[str, Any] = {
        "total_legal_moves": total_legal,
        "mapped_legal_moves": mapped_legal,
        "coverage": (mapped_legal / total_legal) if total_legal > 0 else float("nan"),
        "policy": policy,
    }
    if policy == "value_rerank":
        debug["value_rerank_top_k"] = int(min(int(value_rerank_top_k), mapped_legal))
        debug["value_rerank_lambda"] = float(value_rerank_lambda)
        debug["value_rerank_candidates"] = rerank_rows
    if policy == "value_search_d2":
        debug["value_rerank_top_k"] = int(min(int(value_rerank_top_k), mapped_legal))
        debug["value_rerank_lambda"] = float(value_rerank_lambda)
        debug["value_search_d2_candidates"] = search_rows
    if policy == "value_search_halving":
        debug["search_budget"] = int(halving_config.budget)
        debug["value_search_halving_candidates"] = halving_rows
    if debug_topk > 0:
        k = min(int(debug_topk), mapped_legal)
        top_values, top_indices = torch.topk(legal_logits, k=k, largest=True)
        debug["topk_legal"] = [
            {
                "move_uci": legal_moves_with_ids[int(local_idx)].uci(),
                "logit": float(value.item()),
            }
            for value, local_idx in zip(top_values, top_indices)
        ]
    return legal_moves_with_ids[chosen_index], debug


def _drive_stepwise_as_decode_waves(
    gen: Generator[EvalRequest, list[PositionEval], Any],
    evaluator: CachedPositionEvaluator,
) -> Generator[WorkRequest, Any, Any]:
    """Pump a `*_stepwise` search generator to completion, forwarding every
    `EvalRequest` it yields as `WorkRequest("decode_wave", (evaluator,
    batch))` for the batch scheduler to answer.

    `search._rerank_stepwise` / `search._d2_stepwise` / `search.
    _halving_stepwise` all share this same `EvalRequest` yield contract (via
    `search._expand_root_candidates_stepwise`), so one driver here handles
    all three policies. Mirrors `scripts/generate_search_rollouts.py`'s
    `_generate_rollout_row` driving pattern for `_halving_stepwise`
    (duplicated rather than imported from there: that script's version is
    fused with its own `_TimingStats` bookkeeping via `_timed_advance`,
    which this eval driver has no equivalent of).
    """
    try:
        request = next(gen)
        while True:
            position_evals = yield WorkRequest("decode_wave", (evaluator, request.batch))
            request = gen.send(position_evals)
    except StopIteration as stop:
        return stop.value


def _select_model_move_stepwise(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    board: chess.Board,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    device: torch.device,
    dtype: torch.dtype,
    policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    debug_topk: int = 0,
    halving_config: HalvingConfig | None = None,
) -> Generator[WorkRequest, Any, tuple[chess.Move, dict[str, Any]]]:
    """Scheduler-driven twin of `_select_model_move`: yields
    `WorkRequest("root_eval", batch)` for the root forward instead of
    calling `_forward_model` synchronously -- the payload is the bare
    `batch` dict, the same contract `generate_search_rollouts.py`'s
    `_generate_rollout_row` uses, answered by the shared
    `_make_root_eval_executor` (which always requests `return_kv=True`,
    unlike `_select_model_move`'s policy-conditional `return_kv=policy !=
    "greedy"` -- a harmless compute-only difference for `greedy`: kv_caches
    get computed but are never consulted, since `greedy` never builds an
    evaluator or issues a decode wave).

    For any non-greedy policy this then drives that policy's `*_stepwise`
    generator core via `_drive_stepwise_as_decode_waves`, forwarding every
    `EvalRequest` as `WorkRequest("decode_wave", ...)`. All three non-greedy
    policies route through their `*_stepwise` core exactly as
    `_select_model_move` routes through their synchronous `select_*`
    wrapper -- same if/elif dispatch, same requires-value-head guards, same
    debug-dict shape.
    """
    output = yield WorkRequest("root_eval", batch)

    logits = output["logits"][-1]
    legal_logits, legal_moves_with_ids, total_legal, mapped_legal = _project_legal_logits(
        logits=logits,
        board=board,
        move_vocab=move_vocab,
    )
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    evaluator: CachedPositionEvaluator | None = None
    if policy != "greedy":
        evaluator = CachedPositionEvaluator(
            model=model,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            device=device,
            dtype=dtype,
            prefix_kv=output["kv_caches"],
            prefix_len=int(batch["total_tokens"]),
        )
    rerank_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    halving_rows: list[dict[str, Any]] = []
    if policy == "greedy":
        chosen_index = select_greedy(legal_log_priors)
    elif policy == "value_rerank":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_rerank requires a checkpoint with value head enabled."
            )
        chosen_index, rerank_rows = yield from _drive_stepwise_as_decode_waves(
            search._rerank_stepwise(
                extend=evaluator.extend,
                root_handle=None,
                board=board,
                legal_moves=legal_moves_with_ids,
                legal_log_priors=legal_log_priors,
                top_k=value_rerank_top_k,
                lam=value_rerank_lambda,
            ),
            evaluator,
        )
    elif policy == "value_search_d2":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_d2 requires a checkpoint with value head enabled."
            )
        chosen_index, search_rows = yield from _drive_stepwise_as_decode_waves(
            search._d2_stepwise(
                extend=evaluator.extend,
                root_handle=None,
                board=board,
                legal_moves=legal_moves_with_ids,
                legal_log_priors=legal_log_priors,
                top_k=value_rerank_top_k,
                lam=value_rerank_lambda,
            ),
            evaluator,
        )
    elif policy == "value_search_halving":
        if output.get("value_logits") is None:
            raise RuntimeError(
                "model_move_policy=value_search_halving requires a checkpoint with value head enabled."
            )
        if halving_config is None:
            raise ValueError("policy=value_search_halving requires halving_config")
        chosen_index, halving_rows = yield from _drive_stepwise_as_decode_waves(
            search._halving_stepwise(
                extend=evaluator.extend,
                root_handle=None,
                board=board,
                legal_moves=legal_moves_with_ids,
                legal_log_priors=legal_log_priors,
                config=halving_config,
            ),
            evaluator,
        )
    else:
        raise ValueError(f"Unknown model move policy: {policy}")
    debug: dict[str, Any] = {
        "total_legal_moves": total_legal,
        "mapped_legal_moves": mapped_legal,
        "coverage": (mapped_legal / total_legal) if total_legal > 0 else float("nan"),
        "policy": policy,
    }
    if policy == "value_rerank":
        debug["value_rerank_top_k"] = int(min(int(value_rerank_top_k), mapped_legal))
        debug["value_rerank_lambda"] = float(value_rerank_lambda)
        debug["value_rerank_candidates"] = rerank_rows
    if policy == "value_search_d2":
        debug["value_rerank_top_k"] = int(min(int(value_rerank_top_k), mapped_legal))
        debug["value_rerank_lambda"] = float(value_rerank_lambda)
        debug["value_search_d2_candidates"] = search_rows
    if policy == "value_search_halving":
        debug["search_budget"] = int(halving_config.budget)
        debug["value_search_halving_candidates"] = halving_rows
    if debug_topk > 0:
        k = min(int(debug_topk), mapped_legal)
        top_values, top_indices = torch.topk(legal_logits, k=k, largest=True)
        debug["topk_legal"] = [
            {
                "move_uci": legal_moves_with_ids[int(local_idx)].uci(),
                "logit": float(value.item()),
            }
            for value, local_idx in zip(top_values, top_indices)
        ]
    return legal_moves_with_ids[chosen_index], debug


def _update_summary(
    summary: EvalSummary,
    *,
    result: str,
    model_color: chess.Color,
    completed: bool,
    plies: int,
) -> None:
    summary.games += 1
    summary.total_plies += int(plies)
    if model_color == chess.WHITE:
        summary.games_as_white += 1
    else:
        summary.games_as_black += 1

    if not completed:
        summary.incomplete_games += 1
        return

    summary.completed_games += 1
    if result == "1/2-1/2":
        summary.draws += 1
        if model_color == chess.WHITE:
            summary.draws_as_white += 1
        else:
            summary.draws_as_black += 1
        return

    model_won = (model_color == chess.WHITE and result == "1-0") or (
        model_color == chess.BLACK and result == "0-1"
    )
    if model_won:
        summary.wins += 1
        if model_color == chess.WHITE:
            summary.wins_as_white += 1
        else:
            summary.wins_as_black += 1
    else:
        summary.losses += 1
        if model_color == chess.WHITE:
            summary.losses_as_white += 1
        else:
            summary.losses_as_black += 1


def _summary_to_payload(
    *,
    summary: EvalSummary,
    checkpoint_path: Path,
    stockfish_path: Path,
    engine_limit: chess.engine.Limit,
    stockfish_options: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    compile_enabled: bool,
    seed: int,
    max_plies: int,
    model_move_policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    opening_random_plies: int,
    search_knobs: dict[str, int],
) -> dict[str, Any]:
    if summary.completed_games > 0:
        win_rate = summary.wins / summary.completed_games
        draw_rate = summary.draws / summary.completed_games
        loss_rate = summary.losses / summary.completed_games
        score_rate_completed = (
            summary.wins + 0.5 * summary.draws
        ) / summary.completed_games
    else:
        win_rate = float("nan")
        draw_rate = float("nan")
        loss_rate = float("nan")
        score_rate_completed = float("nan")
    score_rate_all_games = (
        (summary.wins + 0.5 * summary.draws) / summary.games
        if summary.games > 0
        else float("nan")
    )

    return {
        "games": summary.games,
        "completed_games": summary.completed_games,
        "wins": summary.wins,
        "draws": summary.draws,
        "losses": summary.losses,
        "incomplete_games": summary.incomplete_games,
        "average_plies_per_game": summary.avg_plies,
        "average_full_moves_per_game": summary.avg_full_moves,
        "model_turns": summary.model_turns,
        "legal_moves_total": summary.legal_moves_total,
        "legal_moves_mapped_total": summary.legal_moves_mapped_total,
        "legal_move_coverage_rate": summary.legal_coverage_rate,
        "turns_with_no_vocab_legal_move": summary.turns_with_no_vocab_legal_move,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "score_rate": score_rate_completed,
        "score_rate_all_games": score_rate_all_games,
        "rate_denominator_games": summary.completed_games,
        "by_color": {
            "white": {
                "games": summary.games_as_white,
                "wins": summary.wins_as_white,
                "draws": summary.draws_as_white,
                "losses": summary.losses_as_white,
            },
            "black": {
                "games": summary.games_as_black,
                "wins": summary.wins_as_black,
                "draws": summary.draws_as_black,
                "losses": summary.losses_as_black,
            },
        },
        "run_config": {
            "checkpoint": str(checkpoint_path),
            "stockfish_path": str(stockfish_path),
            "stockfish_limit": str(engine_limit),
            "stockfish_options": stockfish_options,
            "device": str(device),
            "dtype": str(dtype),
            "compile": bool(compile_enabled),
            "seed": int(seed),
            "max_plies": int(max_plies),
            "model_move_policy": model_move_policy,
            "value_rerank_top_k": int(value_rerank_top_k),
            "value_rerank_lambda": float(value_rerank_lambda),
            "opening_random_plies": int(opening_random_plies),
            "search": search_knobs,
        },
    }


def _print_segment_summary(*, segment_name: str, payload: dict[str, Any]) -> None:
    print(f"\n[{segment_name}] summary")
    print(f"  games: {payload['games']}")
    print(
        f"  wins/draws/losses: {payload['wins']} / {payload['draws']} / {payload['losses']}"
    )
    print(
        f"  completed_games: {payload['completed_games']} "
        f"(incomplete={payload['incomplete_games']})"
    )
    print(
        f"  average plies/game: {payload['average_plies_per_game']:.2f} "
        f"(avg full moves: {payload['average_full_moves_per_game']:.2f})"
    )
    print(
        f"  legal coverage: {payload['legal_move_coverage_rate']:.4f} "
        f"(mapped={payload['legal_moves_mapped_total']}, total={payload['legal_moves_total']})"
    )
    print(
        f"  score_rate (completed games): {payload['score_rate']:.4f} "
        f"(denominator={payload['rate_denominator_games']})"
    )
    print(f"  score_rate (all games): {payload['score_rate_all_games']:.4f}")
    white = payload["by_color"]["white"]
    black = payload["by_color"]["black"]
    print(
        f"  as_white (W/D/L): {white['wins']}/{white['draws']}/{white['losses']}"
    )
    print(
        f"  as_black (W/D/L): {black['wins']}/{black['draws']}/{black['losses']}"
    )


def _stockfish_label(*, limit_strength: bool, elo: int | None) -> str:
    if limit_strength:
        return f"Stockfish (elo={elo})"
    return "Stockfish (full strength)"


def _outcome_label(
    *, completed: bool, result: str, model_color: chess.Color
) -> str:
    if not completed:
        return "incomplete"
    if result == "1/2-1/2":
        return "draw"
    model_won = (model_color == chess.WHITE and result == "1-0") or (
        model_color == chess.BLACK and result == "0-1"
    )
    return "model_win" if model_won else "model_loss"


def _save_traced_game(
    *,
    board: chess.Board,
    model_color: chess.Color,
    result: str,
    completed: bool,
    segment_name: str,
    stockfish_label: str,
    game_idx: int,
    save_games_dir: Path,
) -> None:
    """Overwrites {segment}_game{N:03d}_{outcome}.*; if outcome changes between
    reruns (e.g. nondeterministic engine timing), the prior file is orphaned —
    use a different --save-games-dir to keep runs side by side."""
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = segment_name
    game.headers["White"] = (
        "imba-chess" if model_color == chess.WHITE else stockfish_label
    )
    game.headers["Black"] = (
        stockfish_label if model_color == chess.WHITE else "imba-chess"
    )
    game.headers["Result"] = result

    outcome = _outcome_label(completed=completed, result=result, model_color=model_color)
    base_name = f"{segment_name}_game{game_idx + 1:03d}_{outcome}"
    save_games_dir.mkdir(parents=True, exist_ok=True)
    (save_games_dir / f"{base_name}.pgn").write_text(str(game), encoding="utf-8")
    (save_games_dir / f"{base_name}.html").write_text(
        render_game_html(game), encoding="utf-8"
    )


def _build_segment_options(
    *,
    base_threads: int,
    base_hash_mb: int,
    spec: SegmentSpec,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "Threads": int(base_threads),
        "Hash": int(base_hash_mb),
        "UCI_LimitStrength": bool(spec.limit_strength),
    }
    if spec.limit_strength:
        if spec.elo is None:
            raise ValueError("Segment with limit_strength=true requires elo")
        options["UCI_Elo"] = int(spec.elo)
    return options


def _build_segment_specs(args: argparse.Namespace) -> list[SegmentSpec]:
    if args.ladder_elos is None:
        if args.stockfish_limit_strength and args.stockfish_elo is None:
            raise ValueError(
                "--stockfish-elo is required when --stockfish-limit-strength is set"
            )
        name = (
            f"sf_elo_{args.stockfish_elo}"
            if args.stockfish_limit_strength
            else "sf_full_strength"
        )
        return [
            SegmentSpec(
                name=name,
                games=int(args.games),
                limit_strength=bool(args.stockfish_limit_strength),
                elo=int(args.stockfish_elo) if args.stockfish_elo is not None else None,
            )
        ]

    ladder_elos = _parse_ladder_elos(args.ladder_elos)
    games_per_segment = (
        int(args.games)
        if args.ladder_games_per_segment is None
        else int(args.ladder_games_per_segment)
    )
    if games_per_segment < 1:
        raise ValueError("--ladder-games-per-segment must be >= 1")
    specs = [
        SegmentSpec(
            name=f"sf_elo_{elo}",
            games=games_per_segment,
            limit_strength=True,
            elo=int(elo),
        )
        for elo in ladder_elos
    ]
    if bool(args.include_full_strength_segment):
        specs.append(
            SegmentSpec(
                name="sf_full_strength",
                games=games_per_segment,
                limit_strength=False,
                elo=None,
            )
        )
    return specs


def _play_game(
    *,
    game_idx: int,
    engine: Any,
    segment_name: str,
    model: torch.nn.Module,
    move_vocab: MoveVocab,
    board_state_encoder: BoardStateEncoder,
    max_plies: int,
    engine_limit: chess.engine.Limit,
    device: torch.device,
    dtype: torch.dtype,
    model_move_policy: str,
    value_rerank_top_k: int,
    value_rerank_lambda: float,
    opening_random_plies: int,
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
    stockfish_label: str,
    save_games_dir: Path | None,
    halving_config: "HalvingConfig | None" = None,
) -> Generator[WorkRequest, Any, EvalSummary]:
    """One game's coroutine core: the `BatchScheduler` game-factory contract.

    Mirrors today's (pre-scheduler) `_run_segment` inner loop body exactly,
    except its two model-call sites and its one engine-call site are now
    `yield WorkRequest(...)` instead of synchronous calls, so
    `BatchScheduler` can merge them across concurrently-live games:
      - model turn: `yield from _select_model_move_stepwise(...)`, which
        itself yields `WorkRequest("root_eval", ...)` then, for non-greedy
        policies, `WorkRequest("decode_wave", ...)` per search wave.
      - engine turn: `yield WorkRequest("sf_move", (engine, board.copy(),
        engine_limit))` -- `board.copy()` so the engine thread (sf_move
        payloads fan out over a `ThreadPoolExecutor`, see `engine_pool.
        make_sf_move_executor`) never touches the live game board this
        coroutine keeps mutating across ticks.

    Opening-random plies, summary-fragment bookkeeping, debug traces, and
    the `save_games` hook are otherwise untouched from today's inline logic
    -- they need no yield, so they stay plain synchronous code between
    yields exactly as before.

    `engine` is this game's checked-out slot engine (see
    `EnginePool.acquire`/`_release_engine_on_finish`): reused as-is for
    every engine turn in this one game, matching today's one-engine-for-
    all-games-in-a-segment behavior generalized to one-engine-per-
    concurrent-slot.

    Returns (via `StopIteration.value`) a per-game `EvalSummary` fragment
    (`games == 1` once `_update_summary` runs below) -- `_run_segment`'s
    `on_game_done` folds it into the running segment total via
    `_accumulate_summary`, in the scheduler's stream order.
    """
    summary = EvalSummary()
    board = chess.Board()
    history = _SequenceHistory(
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
    )
    model_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
    completed = True
    plies = 0

    while not board.is_game_over(claim_draw=True):
        if plies >= max_plies:
            completed = False
            break
        if plies < opening_random_plies:
            legal = list(board.legal_moves)
            if not legal:
                break
            move = random.choice(legal)
            if game_idx < debug_trace_games and plies < debug_trace_max_plies:
                turn = "W" if board.turn == chess.WHITE else "B"
                tqdm.write(
                    f"[debug][{segment_name}] game={game_idx + 1} ply={plies + 1} turn={turn} "
                    f"opening_random selected={move.uci()}"
                )
        elif board.turn == model_color:
            batch = history.build_batch_for_current_position(board)
            move, debug_info = yield from _select_model_move_stepwise(
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
                debug_topk=debug_topk,
                halving_config=halving_config,
            )
            summary.model_turns += 1
            summary.legal_moves_total += int(debug_info["total_legal_moves"])
            summary.legal_moves_mapped_total += int(
                debug_info["mapped_legal_moves"]
            )
            if int(debug_info["mapped_legal_moves"]) == 0:
                summary.turns_with_no_vocab_legal_move += 1
            if (
                game_idx < debug_trace_games
                and plies < debug_trace_max_plies
            ):
                turn = "W" if board.turn == chess.WHITE else "B"
                coverage = float(debug_info["coverage"])
                tqdm.write(
                    f"[debug][{segment_name}] game={game_idx + 1} ply={plies + 1} turn={turn} "
                    f"coverage={coverage:.3f} selected={move.uci()}"
                )
                topk = debug_info.get("topk_legal")
                if isinstance(topk, list) and topk:
                    topk_str = ", ".join(
                        f"{entry['move_uci']}:{entry['logit']:.3f}"
                        for entry in topk
                    )
                    tqdm.write(f"[debug][{segment_name}]   topk={topk_str}")
                rerank_rows = debug_info.get("value_rerank_candidates")
                if isinstance(rerank_rows, list) and rerank_rows:
                    rerank_str = ", ".join(
                        f"{entry['move_uci']}:logit={entry['policy_logit']:.3f}|"
                        f"v_next={entry['value_next']:.3f}|score={entry['rerank_score']:.3f}"
                        for entry in rerank_rows
                    )
                    tqdm.write(
                        f"[debug][{segment_name}]   value_rerank={rerank_str}"
                    )
                search_rows = debug_info.get("value_search_d2_candidates")
                if isinstance(search_rows, list) and search_rows:
                    search_str = ", ".join(
                        f"{entry['move_uci']}:logit={entry['policy_logit']:.3f}|"
                        f"worst_reply={entry['worst_reply_value']:.3f}|"
                        f"best_reply={entry['best_reply_uci']}|score={entry['search_score']:.3f}"
                        for entry in search_rows
                    )
                    tqdm.write(
                        f"[debug][{segment_name}]   value_search_d2={search_str}"
                    )
                halving_rows = debug_info.get("value_search_halving_candidates")
                if isinstance(halving_rows, list) and halving_rows:
                    halving_str = ", ".join(
                        f"{entry['move_uci']}:evals={entry['evals_spent']}"
                        f"|backed={entry['backed_value']}"
                        f"|score={entry['search_score']}"
                        f"|out_r={entry['eliminated_round']}"
                        for entry in halving_rows
                    )
                    tqdm.write(
                        f"[debug][{segment_name}]   value_search_halving={halving_str}"
                    )
        else:
            result = yield WorkRequest("sf_move", (engine, board.copy(), engine_limit))
            if result.move is None:
                raise RuntimeError("Stockfish returned no move.")
            move = result.move

        history.append_observed_position(board)
        history.record_played_move(move.uci())
        board.push(move)
        plies += 1

    result = board.result(claim_draw=True) if completed else "*"
    if save_games_dir is not None and game_idx < debug_trace_games:
        _save_traced_game(
            board=board,
            model_color=model_color,
            result=result,
            completed=completed,
            segment_name=segment_name,
            stockfish_label=stockfish_label,
            game_idx=game_idx,
            save_games_dir=save_games_dir,
        )
    _update_summary(
        summary,
        result=result,
        model_color=model_color,
        completed=completed,
        plies=plies,
    )
    return summary


def _release_engine_on_finish(
    gen: Generator[WorkRequest, Any, EvalSummary], pool: EnginePool, slot_index: int
) -> Generator[WorkRequest, Any, EvalSummary]:
    """Wrap one game's `_play_game` coroutine so its checked-out engine slot
    is always returned to `pool`, whether the game finishes normally or
    raises.

    A generator's `finally` block runs during a `yield from`'s exception
    unwind, before the exception re-emerges from `next()`/`send()` on the
    *outer* (this) generator -- so `pool.release` fires even on the
    fail-fast path where `BatchScheduler._advance` catches the propagating
    exception and hands it to `_run_segment`'s `on_game_error`, which
    re-raises to kill the run: the slot is freed before that exception ever
    reaches this function's caller.
    """
    try:
        return (yield from gen)
    finally:
        pool.release(slot_index)


def _run_segment(
    *,
    stockfish_path: Path,
    segment_options: dict[str, Any],
    segment_name: str,
    model: torch.nn.Module,
    move_vocab: MoveVocab,
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
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
    stockfish_label: str,
    save_games_dir: Path | None,
    concurrent_games: int,
    halving_config: "HalvingConfig | None" = None,
) -> EvalSummary:
    """Run one segment's `games` games through `BatchScheduler`, for any
    `concurrent_games >= 1`.

    Owns one `EnginePool` of `concurrent_games` Stockfish processes, spawned
    fresh for this segment (each configured with `segment_options` at spawn
    time, matching today's per-segment `engine.configure(...)` call
    generalized to every pool engine) and closed when the segment ends --
    see the module docstring / task report for why this replaces the old
    single-engine-reused-and-reconfigured-across-segments lifecycle.

    Engine-to-game assignment goes through `EnginePool.acquire`/`release`
    (checked out exactly when the scheduler admits a game into a live slot,
    released exactly when that game's coroutine finishes or raises) rather
    than a static `game_idx % concurrent_games` round robin: games can take
    wildly different numbers of plies, so a static round robin can hand two
    *simultaneously live* games the same physical engine the moment their
    durations diverge -- corrupting that engine's UCI session and racing its
    `play()` calls across two `ThreadPoolExecutor` workers in the same tick.
    Checkout/return brackets each game's actual live-slot lifetime instead,
    which is race-free by construction regardless of game-length variance.

    At `concurrent_games=1` the scheduler never holds more than one game
    live at a time (`_fill_slots` only admits a new game once the current
    one's slot frees), so every root_eval/decode_wave/sf_move executor call
    carries exactly one payload -- the same single-item call sequence
    `_select_model_move`/`engine.play()` made directly, pre-scheduler.
    """
    summary = EvalSummary()

    def _spawn_engine() -> chess.engine.SimpleEngine:
        spawned = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
        spawned.configure(segment_options)
        return spawned

    pool = EnginePool(spawn=_spawn_engine, size=concurrent_games)
    try:
        def _game_factory() -> Iterator[tuple[str, Any]]:
            for game_idx in range(games):
                slot_index, slot_engine = pool.acquire()
                gen = _play_game(
                    game_idx=game_idx,
                    engine=slot_engine,
                    segment_name=segment_name,
                    model=model,
                    move_vocab=move_vocab,
                    board_state_encoder=board_state_encoder,
                    max_plies=max_plies,
                    engine_limit=engine_limit,
                    device=device,
                    dtype=dtype,
                    model_move_policy=model_move_policy,
                    value_rerank_top_k=value_rerank_top_k,
                    value_rerank_lambda=value_rerank_lambda,
                    opening_random_plies=opening_random_plies,
                    debug_trace_games=debug_trace_games,
                    debug_trace_max_plies=debug_trace_max_plies,
                    debug_topk=debug_topk,
                    stockfish_label=stockfish_label,
                    save_games_dir=save_games_dir,
                    halving_config=halving_config,
                )
                yield (
                    f"{segment_name}-game{game_idx}",
                    _release_engine_on_finish(gen, pool, slot_index),
                )

        with tqdm(
            total=games,
            desc=f"stockfish-eval[{segment_name}]",
            unit="game",
            dynamic_ncols=True,
        ) as progress:

            def _on_game_done(game_id: str, rows: EvalSummary | None) -> None:
                # rows is never None here: _on_game_error below always
                # re-raises rather than letting the scheduler continue on to
                # report a (game_id, None) completion for a failed game.
                assert rows is not None
                _accumulate_summary(summary, rows)

                progress.update(1)
                white_completed = (
                    summary.wins_as_white
                    + summary.draws_as_white
                    + summary.losses_as_white
                )
                black_completed = (
                    summary.wins_as_black
                    + summary.draws_as_black
                    + summary.losses_as_black
                )
                white_score = (
                    (summary.wins_as_white + 0.5 * summary.draws_as_white)
                    / white_completed
                    if white_completed > 0
                    else float("nan")
                )
                black_score = (
                    (summary.wins_as_black + 0.5 * summary.draws_as_black)
                    / black_completed
                    if black_completed > 0
                    else float("nan")
                )
                live_coverage = (
                    summary.legal_moves_mapped_total / summary.legal_moves_total
                    if summary.legal_moves_total > 0
                    else float("nan")
                )
                progress.set_postfix(
                    {
                        "W": summary.wins,
                        "D": summary.draws,
                        "L": summary.losses,
                        "inc": summary.incomplete_games,
                        "avg_plies": f"{summary.avg_plies:.1f}",
                        "avg_moves": f"{summary.avg_full_moves:.1f}",
                        "cov": "--"
                        if summary.legal_moves_total == 0
                        else f"{live_coverage:.3f}",
                        "no_map": summary.turns_with_no_vocab_legal_move,
                        "srW": "--"
                        if white_completed == 0
                        else f"{white_score:.2f}",
                        "srB": "--"
                        if black_completed == 0
                        else f"{black_score:.2f}",
                    }
                )

            def _on_game_error(game_id: str, exc: BaseException) -> None:
                # Fail-fast policy (Task 3): unlike generate_search_rollouts.
                # py's on_game_error (logs and skips one bad game out of a
                # large batch-generation run), eval play has no equivalent
                # "skip this game" semantics -- a mid-game crash means the
                # segment's results are no longer trustworthy, so this
                # re-raises to kill the whole run rather than silently
                # continuing with a hole in the summary.
                raise exc

            scheduler = BatchScheduler(
                game_factory=_game_factory(),
                executors={
                    "root_eval": _make_root_eval_executor(
                        model=model, device=device, dtype=dtype, stats=None
                    ),
                    "decode_wave": _make_decode_wave_executor(
                        model=model, device=device, dtype=dtype, stats=None
                    ),
                    "sf_move": make_sf_move_executor(pool_threads=concurrent_games),
                },
                concurrent_games=concurrent_games,
                on_game_done=_on_game_done,
                on_game_error=_on_game_error,
            )
            scheduler.run()
    finally:
        pool.close()

    return summary


def _accumulate_summary(target: EvalSummary, fragment: EvalSummary) -> None:
    """Add `fragment`'s counters into `target` in place.

    Shared by `_merge_summaries` (combining whole-segment summaries into the
    aggregate) and `_run_segment`'s `on_game_done` (folding one game's
    just-finished `EvalSummary` fragment into the running segment total, in
    the scheduler's stream order) -- same field-by-field addition either
    way, whether `fragment` covers many games or exactly one.
    """
    target.games += fragment.games
    target.completed_games += fragment.completed_games
    target.wins += fragment.wins
    target.losses += fragment.losses
    target.draws += fragment.draws
    target.games_as_white += fragment.games_as_white
    target.games_as_black += fragment.games_as_black
    target.wins_as_white += fragment.wins_as_white
    target.losses_as_white += fragment.losses_as_white
    target.draws_as_white += fragment.draws_as_white
    target.wins_as_black += fragment.wins_as_black
    target.losses_as_black += fragment.losses_as_black
    target.draws_as_black += fragment.draws_as_black
    target.incomplete_games += fragment.incomplete_games
    target.total_plies += fragment.total_plies
    target.model_turns += fragment.model_turns
    target.legal_moves_total += fragment.legal_moves_total
    target.legal_moves_mapped_total += fragment.legal_moves_mapped_total
    target.turns_with_no_vocab_legal_move += fragment.turns_with_no_vocab_legal_move


def _merge_summaries(summaries: list[EvalSummary]) -> EvalSummary:
    merged = EvalSummary()
    for summary in summaries:
        _accumulate_summary(merged, summary)
    return merged


def main() -> None:
    args = _parse_args()
    repo_config = load_repo_config(args.config)
    eval_cfg = repo_config.eval_vs_stockfish

    args.games = int(eval_cfg.games if args.games is None else args.games)
    args.max_plies = int(
        eval_cfg.max_plies if args.max_plies is None else args.max_plies
    )
    args.seed = int(eval_cfg.seed if args.seed is None else args.seed)
    args.stockfish_path = Path(
        eval_cfg.stockfish_path if args.stockfish_path is None else args.stockfish_path
    )
    args.stockfish_time_sec = (
        eval_cfg.stockfish_time_sec
        if args.stockfish_time_sec is None
        else args.stockfish_time_sec
    )
    args.stockfish_nodes = (
        eval_cfg.stockfish_nodes
        if args.stockfish_nodes is None
        else args.stockfish_nodes
    )
    args.stockfish_depth = (
        eval_cfg.stockfish_depth
        if args.stockfish_depth is None
        else args.stockfish_depth
    )
    args.stockfish_threads = int(
        eval_cfg.stockfish_threads
        if args.stockfish_threads is None
        else args.stockfish_threads
    )
    args.stockfish_hash_mb = int(
        eval_cfg.stockfish_hash_mb
        if args.stockfish_hash_mb is None
        else args.stockfish_hash_mb
    )
    args.stockfish_limit_strength = bool(
        eval_cfg.stockfish_limit_strength
        if args.stockfish_limit_strength is None
        else args.stockfish_limit_strength
    )
    args.stockfish_elo = (
        eval_cfg.stockfish_elo if args.stockfish_elo is None else args.stockfish_elo
    )
    args.ladder_elos = (
        eval_cfg.ladder_elos if args.ladder_elos is None else args.ladder_elos
    )
    args.ladder_games_per_segment = (
        eval_cfg.ladder_games_per_segment
        if args.ladder_games_per_segment is None
        else args.ladder_games_per_segment
    )
    args.include_full_strength_segment = bool(
        eval_cfg.include_full_strength_segment
        if args.include_full_strength_segment is None
        else args.include_full_strength_segment
    )
    args.device = str(eval_cfg.device if args.device is None else args.device)
    args.dtype = str(eval_cfg.dtype if args.dtype is None else args.dtype)
    args.compile = bool(eval_cfg.compile if args.compile is None else args.compile)
    args.model_move_policy = str(
        eval_cfg.model_move_policy
        if args.model_move_policy is None
        else args.model_move_policy
    )
    args.value_rerank_top_k = int(
        eval_cfg.value_rerank_top_k
        if args.value_rerank_top_k is None
        else args.value_rerank_top_k
    )
    args.value_rerank_lambda = float(
        eval_cfg.value_rerank_lambda
        if args.value_rerank_lambda is None
        else args.value_rerank_lambda
    )
    args.search_budget = int(
        eval_cfg.search_budget if args.search_budget is None else args.search_budget
    )
    args.search_top_m = int(
        eval_cfg.search_top_m if args.search_top_m is None else args.search_top_m
    )
    args.halving_rounds = int(
        eval_cfg.halving_rounds if args.halving_rounds is None else args.halving_rounds
    )
    args.search_refutation_top_r = int(
        eval_cfg.search_refutation_top_r
        if args.search_refutation_top_r is None
        else args.search_refutation_top_r
    )
    args.search_expand_top = int(
        eval_cfg.search_expand_top
        if args.search_expand_top is None
        else args.search_expand_top
    )
    args.search_max_depth = int(
        eval_cfg.search_max_depth
        if args.search_max_depth is None
        else args.search_max_depth
    )
    args.opening_random_plies = int(
        eval_cfg.opening_random_plies
        if args.opening_random_plies is None
        else args.opening_random_plies
    )
    args.debug_trace_games = int(
        eval_cfg.debug_trace_games
        if args.debug_trace_games is None
        else args.debug_trace_games
    )
    args.debug_trace_max_plies = int(
        eval_cfg.debug_trace_max_plies
        if args.debug_trace_max_plies is None
        else args.debug_trace_max_plies
    )
    args.debug_topk = int(
        eval_cfg.debug_topk if args.debug_topk is None else args.debug_topk
    )
    args.save_games = bool(
        eval_cfg.save_games if args.save_games is None else args.save_games
    )
    args.save_games_dir = Path(
        eval_cfg.save_games_dir
        if args.save_games_dir is None
        else args.save_games_dir
    )
    args.concurrent_games = int(
        eval_cfg.concurrent_games
        if args.concurrent_games is None
        else args.concurrent_games
    )

    if args.games < 1:
        raise ValueError("--games must be >= 1")
    if args.max_plies < 1:
        raise ValueError("--max-plies must be >= 1")
    if args.stockfish_limit_strength and args.stockfish_elo is None:
        raise ValueError(
            "--stockfish-elo is required when --stockfish-limit-strength is set"
        )
    if args.stockfish_threads < 1:
        raise ValueError("--stockfish-threads must be >= 1")
    if args.stockfish_hash_mb < 1:
        raise ValueError("--stockfish-hash-mb must be >= 1")
    if args.opening_random_plies < 0:
        raise ValueError("--opening-random-plies must be >= 0")
    if args.value_rerank_top_k < 1:
        raise ValueError("--value-rerank-top-k must be >= 1")
    if float(args.value_rerank_lambda) < 0.0:
        raise ValueError("--value-rerank-lambda must be >= 0")
    if args.model_move_policy not in {
        "greedy",
        "value_rerank",
        "value_search_d2",
        "value_search_halving",
    }:
        raise ValueError(
            "--model-move-policy must be one of: greedy, value_rerank, "
            "value_search_d2, value_search_halving"
        )
    if args.search_budget < 1:
        raise ValueError("--search-budget must be >= 1")
    if args.search_top_m < 1:
        raise ValueError("--search-top-m must be >= 1")
    if args.halving_rounds < 0:
        raise ValueError("--halving-rounds must be >= 0")
    if args.search_refutation_top_r < 1:
        raise ValueError("--search-refutation-top-r must be >= 1")
    if args.search_expand_top < 1:
        raise ValueError("--search-expand-top must be >= 1")
    if args.search_max_depth < 1:
        raise ValueError("--search-max-depth must be >= 1")
    if args.concurrent_games < 1:
        raise ValueError("--concurrent-games must be >= 1")
    if not args.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish binary not found: {args.stockfish_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path,
        include_unk=repo_config.vocab.include_unk,
    )
    board_state_encoder = BoardStateEncoder(repo_config.board_state)

    model, compile_enabled = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=bool(args.compile),
        require_value_head=(
            str(args.model_move_policy)
            in {"value_rerank", "value_search_d2", "value_search_halving"}
        ),
    )
    engine_limit = _build_engine_limit(args)
    segment_specs = _build_segment_specs(args)
    print("Running model vs Stockfish")
    print(f"  segments={len(segment_specs)}")
    print(f"  stockfish={args.stockfish_path}")
    print(f"  limit={engine_limit}")
    print(f"  device={device}, dtype={dtype}, compile={compile_enabled}")
    print(
        "  model_policy="
        f"{args.model_move_policy}, "
        f"value_rerank_top_k={args.value_rerank_top_k}, "
        f"value_rerank_lambda={args.value_rerank_lambda}, "
        f"opening_random_plies={args.opening_random_plies}"
    )
    print(f"  concurrent_games={args.concurrent_games}")

    segment_results: list[dict[str, Any]] = []
    segment_summaries: list[EvalSummary] = []

    for spec in segment_specs:
        segment_options = _build_segment_options(
            base_threads=args.stockfish_threads,
            base_hash_mb=args.stockfish_hash_mb,
            spec=spec,
        )
        print(
            f"\nRunning segment '{spec.name}' "
            f"(games={spec.games}, options={segment_options}, "
            f"concurrent_games={args.concurrent_games})"
        )
        segment_summary = _run_segment(
            stockfish_path=args.stockfish_path,
            segment_options=segment_options,
            segment_name=spec.name,
            model=model,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            games=spec.games,
            max_plies=args.max_plies,
            engine_limit=engine_limit,
            device=device,
            dtype=dtype,
            model_move_policy=str(args.model_move_policy),
            value_rerank_top_k=int(args.value_rerank_top_k),
            value_rerank_lambda=float(args.value_rerank_lambda),
            opening_random_plies=int(args.opening_random_plies),
            debug_trace_games=max(0, int(args.debug_trace_games)),
            debug_trace_max_plies=max(0, int(args.debug_trace_max_plies)),
            debug_topk=max(0, int(args.debug_topk)),
            stockfish_label=_stockfish_label(
                limit_strength=bool(spec.limit_strength),
                elo=int(spec.elo) if spec.elo is not None else None,
            ),
            save_games_dir=Path(args.save_games_dir) if args.save_games else None,
            concurrent_games=int(args.concurrent_games),
            halving_config=HalvingConfig(
                budget=int(args.search_budget),
                top_m=int(args.search_top_m),
                rounds=int(args.halving_rounds),
                refutation_top_r=int(args.search_refutation_top_r),
                expand_top=int(args.search_expand_top),
                max_depth=int(args.search_max_depth),
                lam=float(args.value_rerank_lambda),
            ),
        )
        segment_payload = _summary_to_payload(
            summary=segment_summary,
            checkpoint_path=args.checkpoint,
            stockfish_path=args.stockfish_path,
            engine_limit=engine_limit,
            stockfish_options=segment_options,
            device=device,
            dtype=dtype,
            compile_enabled=compile_enabled,
            seed=args.seed,
            max_plies=args.max_plies,
            model_move_policy=str(args.model_move_policy),
            value_rerank_top_k=int(args.value_rerank_top_k),
            value_rerank_lambda=float(args.value_rerank_lambda),
            opening_random_plies=int(args.opening_random_plies),
            search_knobs={
                "search_budget": int(args.search_budget),
                "search_top_m": int(args.search_top_m),
                "halving_rounds": int(args.halving_rounds),
                "search_refutation_top_r": int(args.search_refutation_top_r),
                "search_expand_top": int(args.search_expand_top),
                "search_max_depth": int(args.search_max_depth),
            },
        )
        _print_segment_summary(segment_name=spec.name, payload=segment_payload)
        segment_summaries.append(segment_summary)
        segment_results.append(
            {
                "name": spec.name,
                "games_requested": int(spec.games),
                "stockfish": {
                    "limit_strength": bool(spec.limit_strength),
                    "elo": None if spec.elo is None else int(spec.elo),
                    "options": segment_options,
                },
                "results": segment_payload,
            }
        )

    aggregate_summary = _merge_summaries(segment_summaries)
    aggregate_payload = _summary_to_payload(
        summary=aggregate_summary,
        checkpoint_path=args.checkpoint,
        stockfish_path=args.stockfish_path,
        engine_limit=engine_limit,
        stockfish_options={
            "segments": [
                {
                    "name": result["name"],
                    "options": result["stockfish"]["options"],
                }
                for result in segment_results
            ]
        },
        device=device,
        dtype=dtype,
        compile_enabled=compile_enabled,
        seed=args.seed,
        max_plies=args.max_plies,
        model_move_policy=str(args.model_move_policy),
        value_rerank_top_k=int(args.value_rerank_top_k),
        value_rerank_lambda=float(args.value_rerank_lambda),
        opening_random_plies=int(args.opening_random_plies),
        search_knobs={
            "search_budget": int(args.search_budget),
            "search_top_m": int(args.search_top_m),
            "halving_rounds": int(args.halving_rounds),
            "search_refutation_top_r": int(args.search_refutation_top_r),
            "search_expand_top": int(args.search_expand_top),
            "search_max_depth": int(args.search_max_depth),
        },
    )
    _print_segment_summary(segment_name="aggregate", payload=aggregate_payload)

    if len(segment_results) == 1 and args.ladder_elos is None:
        # Backward-compatible single-segment payload shape.
        payload = segment_results[0]["results"]
    else:
        payload = {
            "mode": "ladder" if args.ladder_elos is not None else "multi_segment",
            "segments": segment_results,
            "aggregate": aggregate_payload,
        }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote: {args.output_json}")


def _main_with_hard_exit_on_crash() -> None:
    """Entry-point wrapper: guarantees the process actually terminates on an
    unhandled exception (or Ctrl-C), instead of hanging.

    Duplicated from `scripts/generate_search_rollouts.py`'s wrapper of the
    same name rather than factored into a shared helper (Task 3's brief:
    extract only if trivial, otherwise duplicate the ~15 lines with a
    comment -- a shared helper would need to import a script-independent
    "hard exit" module, which is more indirection than the ~15 duplicated
    lines below are worth) -- see that copy's docstring for the full
    root-cause writeup (PyTorch Inductor's AsyncCompile background
    ThreadPoolExecutor + CPython's non-daemon-thread-joining shutdown path).
    The same fail-fast policy applies here: BatchScheduler's on_game_error
    (Task 3, above) re-raises immediately on any per-game exception, and
    this wrapper is what turns that re-raise into an actual process exit
    instead of a hang.
    """
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    _main_with_hard_exit_on_crash()
