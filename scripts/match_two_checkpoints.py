#!/usr/bin/env python
"""Head-to-head match between two checkpoints, sharing one batch scheduler.

Why this exists: measuring two checkpoints by playing each against Stockfish
and differencing the two score rates carries BOTH arms' sampling error (SE
~0.042 at 200 games/arm) plus whatever the Elo-limited opponent's own
randomization contributes. Playing them against each other measures the
contrast directly, so one game is one paired observation.

Two variance reductions on top of that:

1. **Paired openings with colour reversal.** Every opening is played twice --
   once with A as White, once with B as White. Opening imbalance and colour
   advantage cancel within the pair instead of being averaged over.
2. **Real openings.** Evaluation search is deterministic,
   so from the initial position every game would be the SAME game. Openings are the first `--opening-plies` moves of
   real Lichess games, which are varied and roughly balanced -- unlike
   uniform-random legal plies, which reach lopsided junk positions.

Batching: the scheduler groups pending requests by an opaque `kind` string,
so registering "A:root_eval"/"A:decode_wave"/"B:root_eval"/"B:decode_wave"
yields one merged forward per model per tick with no scheduler changes.
"""

from __future__ import annotations
from imba_chess.eval.inference_runtime import INFERENCE_DTYPES, SINGLE_LEAF_ALGORITHMS, load_runtime
from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.eval.search_options import PUCT_METHODS, SEARCH_METHODS, puct_config, search_backend
from dataclasses import asdict

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Generator, Iterator

import chess
import torch
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.position_evaluator import (
    _SequenceHistory,
)
from imba_chess.eval.search import HalvingConfig


def _select_move(*, side, board, history, runtime, config, rng):
    gen = runtime.search(
        board=board,
        history=history,
        actor_id=side,
        game_id=str(id(history)),
        config=config,
        rng=rng,
        noise=0.0 if runtime.algorithm in SINGLE_LEAF_ALGORITHMS else None,
    )
    try:
        request = next(gen)
        while True:
            answer = yield WorkRequest(f"{side}:{request.kind}", request.payload)
            request = gen.send(answer)
    except StopIteration as stop:
        return stop.value.move_uci
    finally:
        gen.close()


def _play_game(
    *,
    opening_ucis: list[str],
    a_is_white: bool,
    models: dict[str, Any],
    move_vocab,
    board_state_encoder,
    device,
    dtype,
    halving_config: HalvingConfig,
    max_plies: int,
    seed: int,
    game_key: str,
    search_configs=None,
) -> Generator[WorkRequest, Any, dict[str, Any]]:
    """Plays one full game. Returns a result dict scored from A's perspective."""
    board = chess.Board()
    history = _SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=board_state_encoder
    )
    for uci in opening_ucis:
        history.append_observed_position(board)
        history.record_played_move(uci)
        board.push(chess.Move.from_uci(uci))

    adjudicated = False
    while True:
        # claim_draw=True: Gumbel search treats threefold/fifty-move positions as terminal
        # (as eval_vs_stockfish and self-play do) and refuses to search them.
        if board.is_game_over(claim_draw=True):
            break
        if len(board.move_stack) >= max_plies:
            adjudicated = True
            break
        side = "A" if (board.turn == chess.WHITE) == a_is_white else "B"
        uci = yield from _select_move(
            side=side,
            board=board,
            history=history,
            runtime=models[side],
            config=search_configs[side] if search_configs is not None else halving_config,
            rng=random.Random(f"{seed}:{game_key}:{len(board.move_stack)}"),
        )
        history.append_observed_position(board)
        history.record_played_move(uci)
        board.push(chess.Move.from_uci(uci))

    if adjudicated:
        a_score = 0.5
        result = "adjudicated-draw"
    else:
        result = board.result(claim_draw=True)
        if result == "1/2-1/2":
            a_score = 0.5
        else:
            white_won = result == "1-0"
            a_score = 1.0 if (white_won == a_is_white) else 0.0
    return {
        "game_key": game_key,
        "a_is_white": a_is_white,
        "result": result,
        "a_score": a_score,
        "plies": len(board.move_stack),
        "adjudicated": adjudicated,
    }


def _opening_iter(lichess_dataset, *, opening_plies: int, num_openings: int):
    """First `opening_plies` UCIs of real games, skipping games that are too short."""
    out = []
    for game in lichess_dataset.stream():
        plays = game["plays"]
        if len(plays) < opening_plies + 10:
            continue
        out.append([p["move_uci"] for p in plays[:opening_plies]])
        if len(out) >= num_openings:
            break
    if len(out) < num_openings:
        raise RuntimeError(f"only {len(out)} openings available, need {num_openings}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--checkpoint-a", type=Path, required=True)
    p.add_argument("--checkpoint-b", type=Path, required=True)
    p.add_argument("--inference-dtype-a", choices=sorted(INFERENCE_DTYPES), default="float32")
    p.add_argument("--inference-dtype-b", choices=sorted(INFERENCE_DTYPES), default="float32")
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument(
        "--games",
        type=int,
        default=200,
        help="total games; rounded down to an even number (paired)",
    )
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--concurrent-games", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--model-move-policy",
        choices=SEARCH_METHODS,
        default="value_search_halving",
    )
    # Each side can select a traversal, budget and PUCT target configuration.
    for side in ("a", "b"):
        p.add_argument(
            f"--model-move-policy-{side}",
            choices=SEARCH_METHODS,
            default=None,
            help=f"Move choice for side {side.upper()} (default: --model-move-policy).",
        )
        p.add_argument(f"--search-config-{side}", type=Path, help="PUCT TOML [search] parameters.")
        p.add_argument(f"--simulations-{side}", type=int, default=None)
    p.add_argument("--simulations", type=int, default=None, help="Shared Gumbel/PUCT budget; per-side flags override.")
    p.add_argument("--search-config", type=Path, help="Shared PUCT TOML [search] parameters.")
    p.add_argument("--gumbel-simulations", type=int, default=None)
    p.add_argument("--search-budget", type=int, default=None)
    p.add_argument("--search-max-depth", type=int, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def _search_budget(config, method):
    if method in ("policy", "policy_q"):
        return 0
    return config.simulations if search_backend(method) in SINGLE_LEAF_ALGORITHMS else config.budget


def _search_configs(args, eval_cfg, side_policies):
    if args.simulations is not None and args.gumbel_simulations is not None:
        raise ValueError("use only one of --simulations and --gumbel-simulations")
    if args.gumbel_simulations is not None and "gumbel" not in side_policies.values():
        raise ValueError("--gumbel-simulations applies only to gumbel")
    configs = {}
    for side, method in side_policies.items():
        side_budget = getattr(args, f"simulations_{side.lower()}")
        budget = side_budget if side_budget is not None else args.simulations
        path = getattr(args, f"search_config_{side.lower()}") or args.search_config
        if path is not None and method not in PUCT_METHODS:
            raise ValueError("--search-config applies only to PUCT methods; use per-side flags for mixed matches")
        if method in PUCT_METHODS:
            if args.search_budget is not None or args.search_max_depth is not None:
                raise ValueError("--search-budget and --search-max-depth apply only to value_search_halving")
            configs[side] = puct_config(method, path=path, simulations=budget)
        elif method == "gumbel":
            if args.search_budget is not None or args.search_max_depth is not None:
                raise ValueError("--search-budget and --search-max-depth apply only to value_search_halving")
            configs[side] = GumbelConfig(simulations=budget if budget is not None else (
                args.gumbel_simulations if args.gumbel_simulations is not None else 128))
        else:
            if budget is not None:
                raise ValueError("--simulations applies only to Gumbel/PUCT")
            configs[side] = HalvingConfig(
                budget=args.search_budget if args.search_budget is not None else eval_cfg.search_budget,
                top_m=eval_cfg.search_top_m, rounds=eval_cfg.halving_rounds,
                refutation_top_r=eval_cfg.search_refutation_top_r, expand_top=eval_cfg.search_expand_top,
                max_depth=args.search_max_depth if args.search_max_depth is not None else eval_cfg.search_max_depth,
                lam=eval_cfg.search_lambda,
            )
    return configs


def main() -> None:
    args = _parse_args()
    repo_config = load_repo_config(args.config)
    eval_cfg = repo_config.eval_vs_stockfish

    side_policies = {
        "A": args.model_move_policy_a or args.model_move_policy,
        "B": args.model_move_policy_b or args.model_move_policy,
    }
    search_configs = _search_configs(args, eval_cfg, side_policies)
    device_arg = args.device or eval_cfg.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    dtype = torch.float32
    models = {}
    for side, checkpoint, side_dtype in (
        ("A", args.checkpoint_a, args.inference_dtype_a),
        ("B", args.checkpoint_b, args.inference_dtype_b),
    ):
        models[side], _ = load_runtime(
            repo_config=repo_config,
            checkpoint=checkpoint,
            device=device,
            algorithm=search_backend(side_policies[side]),
            dtype=INFERENCE_DTYPES[side_dtype],
        )
    move_vocab = models["A"].move_vocab
    board_state_encoder = models["A"].encoder

    halving_config = search_configs["A"]
    max_plies = int(
        args.max_plies if args.max_plies is not None else eval_cfg.max_plies
    )

    num_pairs = args.games // 2
    dataset_cfg = repo_config.dataset
    lichess_dataset = LichessDataset(
        min_avg_elo=dataset_cfg.min_avg_elo,
        min_time_control_sec=dataset_cfg.min_time_control_sec,
        split="train",
        dataset_name=dataset_cfg.dataset_name,
        train_start_month=dataset_cfg.train_start_month,
        train_end_month=dataset_cfg.train_end_month,
        cache_dir=dataset_cfg.cache_dir,
        parquet_batch_size=dataset_cfg.parquet_batch_size,
        max_seq_len=dataset_cfg.max_seq_len,
        shuffle_train_month_files_on_start=dataset_cfg.shuffle_train_month_files_on_start,
        train_month_shuffle_seed=dataset_cfg.train_month_shuffle_seed,
        train_shuffle_buffer_size=dataset_cfg.train_shuffle_buffer_size,
        board_state_config=repo_config.board_state,
    )
    print(
        f"collecting {num_pairs} openings ({args.opening_plies} plies each)...",
        flush=True,
    )
    openings = _opening_iter(
        lichess_dataset, opening_plies=args.opening_plies, num_openings=num_pairs
    )
    print(f"collected {len(openings)} openings", flush=True)

    def _game_factory() -> Iterator[tuple[str, Generator]]:
        for i, opening in enumerate(openings):
            for a_is_white in (True, False):
                key = f"{i}:{'AW' if a_is_white else 'BW'}"
                yield (
                    key,
                    _play_game(
                        opening_ucis=opening,
                        a_is_white=a_is_white,
                        models=models,
                        move_vocab=move_vocab,
                        board_state_encoder=board_state_encoder,
                        device=device,
                        dtype=dtype,
                        halving_config=halving_config,
                        search_configs=search_configs,
                        max_plies=max_plies,
                        seed=args.seed,
                        game_key=key,
                    ),
                )

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    bar = tqdm(
        total=len(openings) * 2, unit="game", desc=f"{args.label_a} vs {args.label_b}"
    )

    def _on_done(game_key: str, value: Any) -> None:
        if value is not None:
            results.append(value)
        wins = sum(1 for r in results if r["a_score"] == 1.0)
        draws = sum(1 for r in results if r["a_score"] == 0.5)
        losses = sum(1 for r in results if r["a_score"] == 0.0)
        n = len(results)
        bar.set_postfix_str(
            f"A: W{wins}/D{draws}/L{losses} score={(wins + 0.5 * draws) / n:.4f}"
            if n
            else ""
        )
        bar.update(1)

    def _on_error(game_key: str, exc: BaseException) -> None:
        raise RuntimeError(f"match game {game_key} failed") from exc

    try:
        BatchScheduler(
            game_factory=_game_factory(),
            executors={
                f"{side}:{kind}": execute
                for side, runtime in models.items()
                for kind, execute in runtime.executors.items()
            },
            concurrent_games=args.concurrent_games,
            on_game_done=_on_done,
            on_game_error=_on_error,
        ).run()
    finally:
        for runtime in models.values():
            runtime.clear_caches()
        bar.close()

    n = len(results)
    wins = sum(1 for r in results if r["a_score"] == 1.0)
    draws = sum(1 for r in results if r["a_score"] == 0.5)
    losses = sum(1 for r in results if r["a_score"] == 0.0)
    score = (wins + 0.5 * draws) / n if n else float("nan")
    var = (wins + 0.25 * draws) / n - score * score if n else float("nan")
    se = (var / n) ** 0.5 if n else float("nan")

    summary = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "inference_dtype_a": args.inference_dtype_a,
        "inference_dtype_b": args.inference_dtype_b,
        "games_completed": n,
        "games_requested": len(openings) * 2,
        "errors": errors,
        "a_wins": wins,
        "a_draws": draws,
        "a_losses": losses,
        "a_score_rate": score,
        "a_score_se": se,
        "adjudicated_draws": sum(1 for r in results if r["adjudicated"]),
        "algorithm": side_policies["A"] if side_policies["A"] == side_policies["B"] else "mixed",
        "algorithm_a": side_policies["A"],
        "algorithm_b": side_policies["B"],
        "search_a": asdict(search_configs["A"]),
        "search_b": asdict(search_configs["B"]),
        "budget_a": _search_budget(search_configs["A"], side_policies["A"]),
        "budget_b": _search_budget(search_configs["B"], side_policies["B"]),
        "budget": _search_budget(halving_config, side_policies["A"]),
        "exploration": "zero_noise" if search_backend(side_policies["A"]) in SINGLE_LEAF_ALGORITHMS else "deterministic",
        "budget_unit": "simulations" if search_backend(side_policies["A"]) in SINGLE_LEAF_ALGORITHMS else "neural_evaluations",
        "budget_unit_a": "simulations" if search_backend(side_policies["A"]) in SINGLE_LEAF_ALGORITHMS else "neural_evaluations",
        "budget_unit_b": "simulations" if search_backend(side_policies["B"]) in SINGLE_LEAF_ALGORITHMS else "neural_evaluations",
        "precision": args.inference_dtype_a if args.inference_dtype_a == args.inference_dtype_b else "mixed",
        "precision_a": args.inference_dtype_a,
        "precision_b": args.inference_dtype_b,
        "exploration_a": "zero_noise" if search_backend(side_policies["A"]) in SINGLE_LEAF_ALGORITHMS else "deterministic",
        "exploration_b": "zero_noise" if search_backend(side_policies["B"]) in SINGLE_LEAF_ALGORITHMS else "deterministic",
        "tf32": False,
        "runtime_revision": models["A"].options["runtime_revision"],
        "search_max_depth": halving_config.max_depth,
        "opening_plies": args.opening_plies,
        "games": results,
    }
    print(
        f"\n{args.label_a} vs {args.label_b}: {n} games  "
        f"W{wins}/D{draws}/L{losses}  score={score:.4f} +/- {se:.4f} (1 SE)  "
        f"errors={len(errors)}"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output_json}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # same hard exit as the other drivers (see 590838a)


if __name__ == "__main__":
    main()
