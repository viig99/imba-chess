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
2. **Real openings.** Both players run deterministic `value_search_halving`
   (`gumbel_root_sampling=False`), so from the initial position every game
   would be the SAME game. Openings are the first `--opening-plies` moves of
   real Lichess games, which are varied and roughly balanced -- unlike
   uniform-random legal plies, which reach lopsided junk positions.

Batching: the scheduler groups pending requests by an opaque `kind` string,
so registering "A:root_eval"/"A:decode_wave"/"B:root_eval"/"B:decode_wave"
yields one merged forward per model per tick with no scheduler changes.
"""

from __future__ import annotations

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
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import load_or_create_static_move_vocab
from imba_chess.eval import search
from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
from imba_chess.eval.merged_executors import (
    _make_decode_wave_executor,
    _make_root_eval_executor,
)
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _SequenceHistory,
    _project_legal_logits,
    load_hstu_checkpoint,
)
from imba_chess.eval.search import HalvingConfig


def _select_move(
    *,
    side: str,
    board: chess.Board,
    history: _SequenceHistory,
    model,
    move_vocab,
    board_state_encoder,
    device,
    dtype,
    halving_config: HalvingConfig,
    rng: random.Random,
) -> Generator[WorkRequest, Any, str]:
    """One side's move: mirrors generate_search_rollouts._generate_rollout_row's
    coroutine core, but returns the chosen UCI instead of building a row.

    Requests are tagged with `side` so the scheduler merges each model's work
    separately.
    """
    batch = history.build_batch_for_current_position(board)
    output = yield WorkRequest(f"{side}:root_eval", batch)

    logits = output["logits"][-1]
    legal_logits, legal_moves, _, mapped_legal = _project_legal_logits(
        logits=logits, board=board, move_vocab=move_vocab
    )
    if mapped_legal == 0:
        # Fail loudly: a position with no vocabulary-representable legal move
        # would silently corrupt the match result if adjudicated instead.
        raise RuntimeError(f"no legal move mapped to vocab at fen={board.fen()}")
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()

    evaluator = CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        device=device,
        dtype=dtype,
        prefix_kv=output["kv_caches"],
        prefix_len=int(batch["total_tokens"]),
    )

    gen = search._halving_stepwise(
        extend=evaluator.extend,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=halving_config,
        rng=rng,
    )
    try:
        request = gen.send(None)
        while True:
            position_evals = yield WorkRequest(
                f"{side}:decode_wave", (evaluator, request.batch)
            )
            request = gen.send(position_evals)
    except StopIteration as stop:
        best_local_idx, _rows = stop.value
    return legal_moves[best_local_idx].uci()


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
        if board.is_game_over(claim_draw=False):
            break
        if len(board.move_stack) >= max_plies:
            adjudicated = True
            break
        side = "A" if (board.turn == chess.WHITE) == a_is_white else "B"
        uci = yield from _select_move(
            side=side,
            board=board,
            history=history,
            model=models[side],
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            device=device,
            dtype=dtype,
            halving_config=halving_config,
            rng=random.Random(f"{seed}:{game_key}:{len(board.move_stack)}"),
        )
        history.append_observed_position(board)
        history.record_played_move(uci)
        board.push(chess.Move.from_uci(uci))

    if adjudicated:
        a_score = 0.5
        result = "adjudicated-draw"
    else:
        result = board.result(claim_draw=False)
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
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--games", type=int, default=200, help="total games; rounded down to an even number (paired)")
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--concurrent-games", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--search-budget", type=int, default=None)
    p.add_argument("--search-max-depth", type=int, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_config = load_repo_config(args.config)
    eval_cfg = repo_config.eval_vs_stockfish

    device_arg = args.device or eval_cfg.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype or eval_cfg.dtype]

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path, include_unk=repo_config.vocab.include_unk
    )
    board_state_encoder = BoardStateEncoder(repo_config.board_state)

    models: dict[str, Any] = {}
    for side, ckpt in (("A", args.checkpoint_a), ("B", args.checkpoint_b)):
        models[side], _ = load_hstu_checkpoint(
            checkpoint_path=ckpt,
            repo_config=repo_config,
            move_vocab=move_vocab,
            device=device,
            compile_model=False,
            require_value_head=True,
        )

    halving_config = HalvingConfig(
        budget=int(args.search_budget if args.search_budget is not None else eval_cfg.search_budget),
        top_m=int(eval_cfg.search_top_m),
        rounds=int(eval_cfg.halving_rounds),
        refutation_top_r=int(eval_cfg.search_refutation_top_r),
        expand_top=int(eval_cfg.search_expand_top),
        max_depth=int(args.search_max_depth if args.search_max_depth is not None else eval_cfg.search_max_depth),
        lam=float(eval_cfg.value_rerank_lambda),
        gumbel_root_sampling=False,
    )
    max_plies = int(args.max_plies if args.max_plies is not None else eval_cfg.max_plies)

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
    print(f"collecting {num_pairs} openings ({args.opening_plies} plies each)...", flush=True)
    openings = _opening_iter(
        lichess_dataset, opening_plies=args.opening_plies, num_openings=num_pairs
    )
    print(f"collected {len(openings)} openings", flush=True)

    def _game_factory() -> Iterator[tuple[str, Generator]]:
        for i, opening in enumerate(openings):
            for a_is_white in (True, False):
                key = f"{i}:{'AW' if a_is_white else 'BW'}"
                yield key, _play_game(
                    opening_ucis=opening,
                    a_is_white=a_is_white,
                    models=models,
                    move_vocab=move_vocab,
                    board_state_encoder=board_state_encoder,
                    device=device,
                    dtype=dtype,
                    halving_config=halving_config,
                    max_plies=max_plies,
                    seed=args.seed,
                    game_key=key,
                )

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    bar = tqdm(total=len(openings) * 2, unit="game", desc=f"{args.label_a} vs {args.label_b}")

    def _on_done(game_key: str, value: Any) -> None:
        if value is not None:
            results.append(value)
        wins = sum(1 for r in results if r["a_score"] == 1.0)
        draws = sum(1 for r in results if r["a_score"] == 0.5)
        losses = sum(1 for r in results if r["a_score"] == 0.0)
        n = len(results)
        bar.set_postfix_str(
            f"A: W{wins}/D{draws}/L{losses} score={(wins + 0.5 * draws) / n:.4f}" if n else ""
        )
        bar.update(1)

    def _on_error(game_key: str, exc: BaseException) -> None:
        errors.append(f"{game_key}: {exc!r}")
        tqdm.write(f"[error] game {game_key}: {exc!r}")
        bar.update(1)

    BatchScheduler(
        game_factory=_game_factory(),
        executors={
            "A:root_eval": _make_root_eval_executor(model=models["A"], device=device, dtype=dtype, stats=None),
            "A:decode_wave": _make_decode_wave_executor(model=models["A"], device=device, dtype=dtype, stats=None),
            "B:root_eval": _make_root_eval_executor(model=models["B"], device=device, dtype=dtype, stats=None),
            "B:decode_wave": _make_decode_wave_executor(model=models["B"], device=device, dtype=dtype, stats=None),
        },
        concurrent_games=args.concurrent_games,
        on_game_done=_on_done,
        on_game_error=_on_error,
    ).run()
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
        "games_completed": n,
        "games_requested": len(openings) * 2,
        "errors": errors,
        "a_wins": wins,
        "a_draws": draws,
        "a_losses": losses,
        "a_score_rate": score,
        "a_score_se": se,
        "adjudicated_draws": sum(1 for r in results if r["adjudicated"]),
        "search_budget": halving_config.budget,
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
