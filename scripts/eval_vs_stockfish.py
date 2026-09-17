from __future__ import annotations
from imba_chess.eval.inference_runtime import load_runtime
from imba_chess.eval.gumbel_search import GumbelConfig
import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator
import chess
import chess.engine
import chess.pgn
import torch
from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval import search
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.engine_pool import EnginePool, make_sf_move_executor
from imba_chess.eval.game_animation import render_game_html
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.eval.search import HalvingConfig
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
    search_stats: dict[str, int] = field(default_factory=dict)
    model_selection_seconds: float = 0.0
    game_records: list[dict[str, Any]] = field(default_factory=list)
    inference_stats: dict[str, float | int] = field(default_factory=dict)

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


_ACTOR_PROFILE = os.environ.get("IMBA_ACTOR_PROFILE") == "1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained imba-chess model against Stockfish via UCI."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stockfish-path", type=Path, default=None)
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
        help="Comma-separated Elo ladder for segmented eval, e.g. '1600,1800,2000,2200,2400,2600,2800'.",
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--model-move-policy",
        choices=["gumbel", "value_search_halving"],
        default=None,
        help="Model move selection on legal moves.",
    )
    parser.add_argument(
        "--search-lambda",
        type=float,
        default=None,
        help="Weight for halving search value adjustment.",
    )
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--search-top-m", type=int, default=None)
    parser.add_argument("--halving-rounds", type=int, default=None)
    parser.add_argument("--search-refutation-top-r", type=int, default=None)
    parser.add_argument("--search-expand-top", type=int, default=None)
    parser.add_argument(
        "--search-max-depth",
        type=int,
        default=None,
        help="Halving search depth, counting plies below a candidate root move.",
    )
    parser.add_argument(
        "--search-tactical-coverage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include forcing moves on both sides and all legal check evasions in halving search.",
    )
    parser.add_argument(
        "--search-quiescence-plies",
        type=int,
        default=None,
        help="Extra capture/promotion/evasion plies within the existing halving budget (default 0).",
    )
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
        help="Concurrent games; preserve algorithm-specific workload settings.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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
    if any((v < 100 for v in values)):
        raise ValueError("Elo values in --ladder-elos must be >= 100")
    return values


def _select_model_move(*, runtime, batch, board, config):
    gen = _select_model_move_stepwise(
        runtime=runtime, batch=batch, board=board, config=config
    )
    try:
        request = next(gen)
        while True:
            request = gen.send(runtime.executors[request.kind]([request.payload])[0])
    except StopIteration as stop:
        return stop.value
    finally:
        gen.close()


def _select_model_move_stepwise(*, runtime, batch, board, config, debug_topk=0):
    topk = []

    def observe(moves, logits):
        values, indices = torch.topk(logits, min(debug_topk, len(moves)))
        topk.extend(
            dict(move_uci=moves[index].uci(), logit=value)
            for value, index in zip(values.tolist(), indices.tolist())
        )

    result = yield from runtime.search_batch(
        board=board,
        batch=batch,
        owner=(id(runtime), id(batch)),
        config=config,
        noise=0.0 if runtime.algorithm == "gumbel" else None,
        root_observer=observe if debug_topk else None,
    )
    legal = board.legal_moves.count()
    debug = dict(
        total_legal_moves=legal,
        mapped_legal_moves=legal,
        coverage=1.0,
        policy=runtime.algorithm,
    )
    if topk:
        debug["topk_legal"] = topk
    if runtime.algorithm == "value_search_halving":
        debug.update(
            search_budget=config.budget,
            value_search_halving_candidates=result.candidates,
            search_stats=search.summarize_search_rows(result.candidates),
        )
    else:
        debug["search_stats"] = dict(
            simulations=result.simulations,
            neural_evaluations=result.neural_evaluations,
            terminal_hits=result.terminal_hits,
            depth_cutoffs=result.depth_cutoffs,
        )
    return (chess.Move.from_uci(result.move_uci), debug)


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
    model_won = (
        model_color == chess.WHITE
        and result == "1-0"
        or (model_color == chess.BLACK and result == "0-1")
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
    search_lambda: float,
    opening_random_plies: int,
    search_knobs: dict[str, int | bool],
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
        "search_stats": dict(summary.search_stats),
        "game_records": summary.game_records,
        "inference_stats": summary.inference_stats,
        "model_selection_seconds": summary.model_selection_seconds,
        "mean_model_selection_seconds": summary.model_selection_seconds
        / summary.model_turns
        if summary.model_turns
        else 0.0,
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
            "search_lambda": float(search_lambda),
            "opening_random_plies": int(opening_random_plies),
            "search": search_knobs,
            "algorithm": model_move_policy,
            "precision": "float32",
            "tf32": False,
            "exploration": "zero_noise"
            if model_move_policy == "gumbel"
            else "deterministic",
            "runtime_revision": "shared-search-v1",
            "budget": search_knobs["gumbel_simulations"]
            if model_move_policy == "gumbel"
            else search_knobs["search_budget"],
            "budget_unit": "simulations"
            if model_move_policy == "gumbel"
            else "neural_evaluations",
        },
    }


def _print_segment_summary(*, segment_name: str, payload: dict[str, Any]) -> None:
    print(f"\n[{segment_name}] summary")
    print(f"  games: {payload['games']}")
    print(
        f"  wins/draws/losses: {payload['wins']} / {payload['draws']} / {payload['losses']}"
    )
    print(
        f"  completed_games: {payload['completed_games']} (incomplete={payload['incomplete_games']})"
    )
    print(
        f"  average plies/game: {payload['average_plies_per_game']:.2f} (avg full moves: {payload['average_full_moves_per_game']:.2f})"
    )
    print(
        f"  legal coverage: {payload['legal_move_coverage_rate']:.4f} (mapped={payload['legal_moves_mapped_total']}, total={payload['legal_moves_total']})"
    )
    print(
        f"  score_rate (completed games): {payload['score_rate']:.4f} (denominator={payload['rate_denominator_games']})"
    )
    print(f"  score_rate (all games): {payload['score_rate_all_games']:.4f}")
    white = payload["by_color"]["white"]
    black = payload["by_color"]["black"]
    print(f"  as_white (W/D/L): {white['wins']}/{white['draws']}/{white['losses']}")
    print(f"  as_black (W/D/L): {black['wins']}/{black['draws']}/{black['losses']}")


def _stockfish_label(*, limit_strength: bool, elo: int | None) -> str:
    if limit_strength:
        return f"Stockfish (elo={elo})"
    return "Stockfish (full strength)"


def _outcome_label(*, completed: bool, result: str, model_color: chess.Color) -> str:
    if not completed:
        return "incomplete"
    if result == "1/2-1/2":
        return "draw"
    model_won = (
        model_color == chess.WHITE
        and result == "1-0"
        or (model_color == chess.BLACK and result == "0-1")
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
    outcome = _outcome_label(
        completed=completed, result=result, model_color=model_color
    )
    base_name = f"{segment_name}_game{game_idx + 1:03d}_{outcome}"
    save_games_dir.mkdir(parents=True, exist_ok=True)
    (save_games_dir / f"{base_name}.pgn").write_text(str(game), encoding="utf-8")
    (save_games_dir / f"{base_name}.html").write_text(
        render_game_html(game), encoding="utf-8"
    )


def _build_segment_options(
    *, base_threads: int, base_hash_mb: int, spec: SegmentSpec
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
    search_lambda: float,
    opening_random_plies: int,
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
    stockfish_label: str,
    save_games_dir: Path | None,
    halving_config: "HalvingConfig | None" = None,
    runtime,
) -> Generator[WorkRequest, Any, EvalSummary]:
    """One game's coroutine core: the `BatchScheduler` game-factory contract.

    Mirrors today's (pre-scheduler) `_run_segment` inner loop body exactly,
    except its two model-call sites and its one engine-call site are now
    `yield WorkRequest(...)` instead of synchronous calls, so
    `BatchScheduler` can merge them across concurrently-live games:
      - model turn: `yield from _select_model_move_stepwise(...)`, which
        itself yields `WorkRequest("root_eval", ...)` then, for retained search
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
        move_vocab=move_vocab, board_state_encoder=board_state_encoder
    )
    model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
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
                    f"[debug][{segment_name}] game={game_idx + 1} ply={plies + 1} turn={turn} opening_random selected={move.uci()}"
                )
        elif board.turn == model_color:
            selection_start = time.perf_counter()
            batch = history.build_batch_for_current_position(board)
            move, debug_info = yield from _select_model_move_stepwise(
                batch=batch,
                board=board,
                runtime=runtime,
                config=halving_config,
                debug_topk=debug_topk
                if game_idx < debug_trace_games and plies < debug_trace_max_plies
                else 0,
            )
            summary.model_turns += 1
            summary.model_selection_seconds += time.perf_counter() - selection_start
            search.merge_search_stats(
                summary.search_stats, debug_info.get("search_stats", {})
            )
            summary.legal_moves_total += int(debug_info["total_legal_moves"])
            summary.legal_moves_mapped_total += int(debug_info["mapped_legal_moves"])
            if int(debug_info["mapped_legal_moves"]) == 0:
                summary.turns_with_no_vocab_legal_move += 1
            if game_idx < debug_trace_games and plies < debug_trace_max_plies:
                turn = "W" if board.turn == chess.WHITE else "B"
                coverage = float(debug_info["coverage"])
                tqdm.write(
                    f"[debug][{segment_name}] game={game_idx + 1} ply={plies + 1} turn={turn} coverage={coverage:.3f} selected={move.uci()}"
                )
                topk = debug_info.get("topk_legal")
                if isinstance(topk, list) and topk:
                    topk_str = ", ".join(
                        (f"{entry['move_uci']}:{entry['logit']:.3f}" for entry in topk)
                    )
                    tqdm.write(f"[debug][{segment_name}]   topk={topk_str}")
                halving_rows = debug_info.get("value_search_halving_candidates")
                if isinstance(halving_rows, list) and halving_rows:
                    halving_str = ", ".join(
                        (
                            f"{entry['move_uci']}:evals={entry['evals_spent']}|backed={entry['backed_value']}|score={entry['search_score']}|out_r={entry['eliminated_round']}"
                            for entry in halving_rows
                        )
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
    summary.game_records.append(
        dict(
            game_idx=game_idx,
            result=result,
            completed=completed,
            model_color="white" if model_color else "black",
            plies=plies,
        )
    )
    return summary


def _release_engine_on_finish(
    gen: Generator[WorkRequest, Any, EvalSummary], pool: EnginePool, slot_index: int
) -> Generator[WorkRequest, Any, EvalSummary]:
    """Wrap one game's `_play_game` coroutine so its checked-out engine slot
    is returned to `pool` on the paths this wrapper can actually observe:
    normal completion, and any exception raised by the game coroutine's own
    code (e.g. `_play_game`'s `raise RuntimeError("Stockfish returned no
    move.")`, or anything else executing between yields). Those paths
    surface as `next()`/`send()` raising on `slot.gen` inside
    `BatchScheduler._advance`'s try block; a generator's `finally` runs
    during a `yield from`'s exception unwind, so `pool.release` fires
    *before* that exception reaches `_advance`'s `except` clause (which then
    calls `_run_segment`'s `on_game_error`, re-raising to kill the run under
    this script's fail-fast policy).

    This wrapper does NOT cover exceptions raised by a merged *executor*
    call (`root_eval`/`decode_wave`/`sf_move` -- e.g. Stockfish crashing
    inside `make_sf_move_executor`'s `engine.play()`). Those calls
    (`self._executors[kind](...)` in `BatchScheduler.run()`'s Phase 3) sit
    OUTSIDE `_advance`'s try block entirely, so such an exception propagates
    straight out of `run()` without ever resuming the tick's suspended
    game generator(s) via `.send()` -- this generator's own `finally` above
    never runs for that game, and the slot is simply abandoned (its
    `_play_game` generator is neither closed nor released here). Engine
    cleanup for that case comes from `_run_segment`'s `finally: pool.close()`,
    which quits every pool engine unconditionally regardless of any
    individual slot's release state -- not from this function.

    This split is intentional: this script's fail-fast policy treats an
    infra-level failure (an engine crash, a GPU error) as fatal to the
    whole run, with no per-game continuation, so there is currently no
    scenario where a released-vs-abandoned distinction at the single-slot
    level matters -- `pool.close()` always fires exactly once, at the very
    end of `_run_segment`, regardless of which path triggered the abort.
    A future "continue past one bad game" mode must NOT be built on top of
    this pool's current acquire/release semantics without first adding
    structured release for the executor-exception path too (e.g. releasing
    every live slot's engine from `BatchScheduler.run()`'s own exception
    handling, not just from each game's wrapper).
    """
    try:
        return (yield from gen)
    finally:
        pool.release(slot_index)


def _progress_postfix(summary: EvalSummary) -> dict[str, Any]:
    """tqdm postfix dict derived purely from a running `EvalSummary` --
    shared by `_run_segment`'s scheduler-driven `_on_game_done` (G=1) and
    concurrent game completion, so
    both progress bars report the same fields the same way. Pure refactor
    out of what `_run_segment`'s `_on_game_done` always computed inline; no
    change to the numbers themselves."""
    white_completed = (
        summary.wins_as_white + summary.draws_as_white + summary.losses_as_white
    )
    black_completed = (
        summary.wins_as_black + summary.draws_as_black + summary.losses_as_black
    )
    white_score = (
        (summary.wins_as_white + 0.5 * summary.draws_as_white) / white_completed
        if white_completed > 0
        else float("nan")
    )
    black_score = (
        (summary.wins_as_black + 0.5 * summary.draws_as_black) / black_completed
        if black_completed > 0
        else float("nan")
    )
    live_coverage = (
        summary.legal_moves_mapped_total / summary.legal_moves_total
        if summary.legal_moves_total > 0
        else float("nan")
    )
    return {
        "W": summary.wins,
        "D": summary.draws,
        "L": summary.losses,
        "inc": summary.incomplete_games,
        "avg_plies": f"{summary.avg_plies:.1f}",
        "avg_moves": f"{summary.avg_full_moves:.1f}",
        "cov": "--" if summary.legal_moves_total == 0 else f"{live_coverage:.3f}",
        "no_map": summary.turns_with_no_vocab_legal_move,
        "srW": "--" if white_completed == 0 else f"{white_score:.2f}",
        "srB": "--" if black_completed == 0 else f"{black_score:.2f}",
    }


def _record_inference(executor, stats, kind):
    """Count actual calls and merged rows where the existing executor runs."""

    def measured(payloads):
        rows = (
            len(payloads)
            if kind == "root"
            else sum((len(payload[1][1]) for payload in payloads))
        )
        started = time.perf_counter()
        result = executor(payloads)
        search.merge_search_stats(
            stats,
            {
                f"{kind}_calls": 1,
                f"{kind}_requests": len(payloads),
                f"{kind}_rows": rows,
                f"{kind}_batch_size_{rows}": 1,
                f"{kind}_seconds": time.perf_counter() - started,
            },
        )
        return result

    return measured


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
    search_lambda: float,
    opening_random_plies: int,
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
    stockfish_label: str,
    save_games_dir: Path | None,
    concurrent_games: int,
    halving_config: "HalvingConfig | None" = None,
    runtime,
) -> EvalSummary:
    """Run all games through shared inference and concurrent Stockfish calls."""
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
                    search_lambda=search_lambda,
                    opening_random_plies=opening_random_plies,
                    debug_trace_games=debug_trace_games,
                    debug_trace_max_plies=debug_trace_max_plies,
                    debug_topk=debug_topk,
                    stockfish_label=stockfish_label,
                    save_games_dir=save_games_dir,
                    halving_config=halving_config,
                    runtime=runtime,
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
                assert rows is not None
                _accumulate_summary(summary, rows)
                progress.update(1)
                progress.set_postfix(_progress_postfix(summary))

            def _on_game_error(game_id: str, exc: BaseException) -> None:
                raise exc

            scheduler = BatchScheduler(
                game_factory=_game_factory(),
                executors={
                    "root_eval": _record_inference(
                        runtime.executors["root_eval"], summary.inference_stats, "root"
                    ),
                    "decode_wave": _record_inference(
                        runtime.executors["decode_wave"],
                        summary.inference_stats,
                        "decode",
                    ),
                    "sf_move": make_sf_move_executor(pool_threads=concurrent_games),
                },
                concurrent_games=concurrent_games,
                on_game_done=_on_game_done,
                on_game_error=_on_game_error,
            )
            scheduler.run()
    finally:
        runtime.clear_caches()
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
    target.model_selection_seconds += fragment.model_selection_seconds
    search.merge_search_stats(target.search_stats, fragment.search_stats)
    target.game_records.extend(fragment.game_records)
    search.merge_search_stats(target.inference_stats, fragment.inference_stats)


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
    args.model_move_policy = str(
        eval_cfg.model_move_policy
        if args.model_move_policy is None
        else args.model_move_policy
    )
    if args.model_move_policy == "gumbel":
        for flag in (
            "search_budget",
            "search_lambda",
            "search_top_m",
            "halving_rounds",
            "search_refutation_top_r",
            "search_expand_top",
            "search_max_depth",
            "search_tactical_coverage",
            "search_quiescence_plies",
        ):
            if getattr(args, flag) is not None:
                raise ValueError(
                    f"--{flag.replace('_', '-')} applies only to value_search_halving"
                )
    elif args.gumbel_simulations is not None:
        raise ValueError("--gumbel-simulations applies only to gumbel")
    args.gumbel_simulations = (
        128 if args.gumbel_simulations is None else args.gumbel_simulations
    )
    args.search_lambda = float(
        eval_cfg.search_lambda if args.search_lambda is None else args.search_lambda
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
    args.search_tactical_coverage = bool(
        eval_cfg.search_tactical_coverage
        if args.search_tactical_coverage is None
        else args.search_tactical_coverage
    )
    args.search_quiescence_plies = int(
        eval_cfg.search_quiescence_plies
        if args.search_quiescence_plies is None
        else args.search_quiescence_plies
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
        eval_cfg.save_games_dir if args.save_games_dir is None else args.save_games_dir
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
    if float(args.search_lambda) < 0.0:
        raise ValueError("--search-lambda must be >= 0")
    if args.model_move_policy not in {"gumbel", "value_search_halving"}:
        raise ValueError("unsupported search algorithm")
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
    if args.search_quiescence_plies < 0:
        raise ValueError("--search-quiescence-plies must be >= 0")
    if (
        args.search_tactical_coverage or args.search_quiescence_plies
    ) and args.model_move_policy != "value_search_halving":
        raise ValueError(
            "Tactical coverage and quiescence require --model-move-policy value_search_halving"
        )
    if args.concurrent_games < 1:
        raise ValueError("--concurrent-games must be >= 1")
    if not args.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish binary not found: {args.stockfish_path}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)
    dtype = torch.float32
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA device requested but not available.")
    runtime, _ = load_runtime(
        repo_config=repo_config,
        checkpoint=args.checkpoint,
        device=device,
        algorithm=args.model_move_policy,
    )
    model, move_vocab, board_state_encoder = (
        runtime.model,
        runtime.move_vocab,
        runtime.encoder,
    )
    compile_enabled = args.model_move_policy == "gumbel"
    engine_limit = _build_engine_limit(args)
    segment_specs = _build_segment_specs(args)
    print("Running model vs Stockfish")
    print(f"  segments={len(segment_specs)}")
    print(f"  stockfish={args.stockfish_path}")
    print(f"  limit={engine_limit}")
    print(f"  device={device}, dtype={dtype}, compile={compile_enabled}")
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
            f"\nRunning segment '{spec.name}' (games={spec.games}, options={segment_options}, concurrent_games={args.concurrent_games})"
        )
        halving_config = HalvingConfig(
            budget=int(args.search_budget),
            top_m=int(args.search_top_m),
            rounds=int(args.halving_rounds),
            refutation_top_r=int(args.search_refutation_top_r),
            expand_top=int(args.search_expand_top),
            max_depth=int(args.search_max_depth),
            lam=float(args.search_lambda),
            tactical_coverage=bool(args.search_tactical_coverage),
            quiescence_plies=int(args.search_quiescence_plies),
        )
        if args.model_move_policy == "gumbel":
            halving_config = GumbelConfig(simulations=args.gumbel_simulations)
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
            search_lambda=float(args.search_lambda),
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
            halving_config=halving_config,
            runtime=runtime,
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
            search_lambda=float(args.search_lambda),
            opening_random_plies=int(args.opening_random_plies),
            search_knobs={
                "gumbel_simulations": int(args.gumbel_simulations),
                "search_budget": int(args.search_budget),
                "search_top_m": int(args.search_top_m),
                "halving_rounds": int(args.halving_rounds),
                "search_refutation_top_r": int(args.search_refutation_top_r),
                "search_expand_top": int(args.search_expand_top),
                "search_max_depth": int(args.search_max_depth),
                "search_tactical_coverage": bool(args.search_tactical_coverage),
                "search_quiescence_plies": int(args.search_quiescence_plies),
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
                {"name": result["name"], "options": result["stockfish"]["options"]}
                for result in segment_results
            ]
        },
        device=device,
        dtype=dtype,
        compile_enabled=compile_enabled,
        seed=args.seed,
        max_plies=args.max_plies,
        model_move_policy=str(args.model_move_policy),
        search_lambda=float(args.search_lambda),
        opening_random_plies=int(args.opening_random_plies),
        search_knobs={
            "gumbel_simulations": int(args.gumbel_simulations),
            "search_budget": int(args.search_budget),
            "search_top_m": int(args.search_top_m),
            "halving_rounds": int(args.halving_rounds),
            "search_refutation_top_r": int(args.search_refutation_top_r),
            "search_expand_top": int(args.search_expand_top),
            "search_max_depth": int(args.search_max_depth),
            "search_tactical_coverage": bool(args.search_tactical_coverage),
            "search_quiescence_plies": int(args.search_quiescence_plies),
        },
    )
    _print_segment_summary(segment_name="aggregate", payload=aggregate_payload)
    if len(segment_results) == 1 and args.ladder_elos is None:
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
    from imba_chess.process import main_with_hard_exit

    main_with_hard_exit(main)


if __name__ == "__main__":
    _main_with_hard_exit_on_crash()
