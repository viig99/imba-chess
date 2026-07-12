#!/usr/bin/env python3
"""Generate search-backed value-target rollouts from the training split.

Replays games from LichessDataset(split="train") and, at a sampled subset of
plies per game, calls the same value_search_halving search used at inference
to record a position-resolved value estimate. Writes one row per sampled
position to a rollout parquet consumed by scripts/train.py via
[expert_iteration].rollout_path. Generates data only; trains nothing (mirrors
scripts/train_value_net.py's role as a small, focused, non-Ignite script).

Usage: python scripts/generate_search_rollouts.py --checkpoint <path> --output-path <path>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import torch
from tqdm.auto import tqdm

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import load_or_create_static_move_vocab
from imba_chess.data.rollout_store import RolloutRow, write_rollout_parquet
from imba_chess.eval.position_evaluator import (
    CachedPositionEvaluator,
    _SequenceHistory,
    _forward_model,
    _project_legal_logits,
    load_hstu_checkpoint,
)
from imba_chess.eval.search import HalvingConfig, select_value_search_halving

_RESULT_TO_WHITE_OUTCOME = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _result_to_white_outcome(result: str) -> int:
    outcome = _RESULT_TO_WHITE_OUTCOME.get(result)
    if outcome is None:
        raise ValueError(f"Unsupported game result: {result!r}")
    return outcome


def _sample_ply_indices(num_plies: int, *, every_n: int, seed: int, game_id: str) -> list[int]:
    if num_plies <= 0:
        return []
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    rng = random.Random(f"{seed}:{game_id}")
    offset = rng.randrange(0, every_n)
    return list(range(offset, num_plies, every_n))


def _pad_or_truncate_arms(rows: list[dict], *, top_m: int) -> list[dict]:
    selected = list(rows[:top_m])
    while len(selected) < top_m:
        selected.append(
            {"move_uci": "", "backed_value": 0.0, "evals_spent": 0, "policy_log_prob": 0.0}
        )
    return [
        {
            "move_uci": row["move_uci"],
            "backed_value": (
                float(row["backed_value"]) if row["backed_value"] is not None else 0.0
            ),
            "evals_spent": int(row["evals_spent"]),
            "policy_log_prob": float(row["policy_log_prob"]),
        }
        for row in selected
    ]


def _generate_rollout_row(
    *,
    model,
    board: chess.Board,
    history: _SequenceHistory,
    move_vocab,
    board_state_encoder: BoardStateEncoder,
    device: torch.device,
    dtype: torch.dtype,
    halving_config: HalvingConfig,
    game_id: str,
    ply: int,
    human_move_uci: str,
    real_outcome_stm: int,
    checkpoint_path: str,
) -> RolloutRow | None:
    batch = history.build_batch_for_current_position(board)
    output = _forward_model(model=model, batch=batch, device=device, dtype=dtype, return_kv=True)

    logits = output["logits"][-1]
    try:
        legal_logits, legal_moves, _, mapped_legal = _project_legal_logits(
            logits=logits, board=board, move_vocab=move_vocab
        )
    except RuntimeError:
        return None
    if mapped_legal == 0:
        return None
    legal_log_priors = torch.log_softmax(legal_logits.float(), dim=0).tolist()
    root_wdl_unsearched = tuple(
        float(v) for v in torch.softmax(output["value_logits"][-1].float(), dim=-1).tolist()
    )

    evaluator = CachedPositionEvaluator(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        device=device,
        dtype=dtype,
        prefix_kv=output["kv_caches"],
        prefix_len=int(batch["total_tokens"]),
    )
    best_local_idx, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=halving_config,
    )
    best_move_uci = legal_moves[best_local_idx].uci()
    best_row = next((row for row in rows if row["move_uci"] == best_move_uci), None)
    if best_row is None or best_row["backed_value"] is None:
        return None  # budget-starved fallback with no scored arm; skip this position

    human_row = next((row for row in rows if row["move_uci"] == human_move_uci), None)
    human_move_backed_value = (
        float(human_row["backed_value"])
        if human_row is not None and human_row["backed_value"] is not None
        else None
    )

    arms = _pad_or_truncate_arms(rows, top_m=halving_config.top_m)
    return RolloutRow(
        game_id=game_id,
        ply=ply,
        human_move_uci=human_move_uci,
        human_move_backed_value=human_move_backed_value,
        real_outcome_stm=real_outcome_stm,
        best_arm_move_uci=best_move_uci,
        best_arm_backed_value=float(best_row["backed_value"]),
        root_wdl_unsearched=root_wdl_unsearched,
        arm_move_uci=tuple(arm["move_uci"] for arm in arms),
        arm_backed_value=tuple(arm["backed_value"] for arm in arms),
        arm_evals_spent=tuple(arm["evals_spent"] for arm in arms),
        arm_log_prior=tuple(arm["policy_log_prob"] for arm in arms),
        search_budget=halving_config.budget,
        search_top_m=halving_config.top_m,
        search_max_depth=halving_config.max_depth,
        checkpoint=checkpoint_path,
    )


def _process_game(
    game: dict,
    *,
    model,
    move_vocab,
    board_state_encoder: BoardStateEncoder,
    device: torch.device,
    dtype: torch.dtype,
    halving_config: HalvingConfig,
    every_n_plies: int,
    sample_seed: int,
    checkpoint_path: str,
) -> list[RolloutRow]:
    plays = game["plays"]
    sampled = set(
        _sample_ply_indices(
            len(plays), every_n=every_n_plies, seed=sample_seed, game_id=game["game_id"]
        )
    )
    if not sampled:
        return []

    game_result_white = _result_to_white_outcome(game["result"])
    board = chess.Board()
    history = _SequenceHistory(move_vocab=move_vocab, board_state_encoder=board_state_encoder)
    rows: list[RolloutRow] = []

    for ply_idx, play in enumerate(plays):
        if ply_idx in sampled:
            real_outcome_stm = game_result_white if board.turn == chess.WHITE else -game_result_white
            row = _generate_rollout_row(
                model=model,
                board=board,
                history=history,
                move_vocab=move_vocab,
                board_state_encoder=board_state_encoder,
                device=device,
                dtype=dtype,
                halving_config=halving_config,
                game_id=game["game_id"],
                ply=ply_idx,
                human_move_uci=play["move_uci"],
                real_outcome_stm=real_outcome_stm,
                checkpoint_path=checkpoint_path,
            )
            if row is not None:
                rows.append(row)
        move = chess.Move.from_uci(play["move_uci"])
        history.append_observed_position(board)
        history.record_played_move(play["move_uci"])
        board.push(move)

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--sample-every-n-plies", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    # Search knobs default to [eval_vs_stockfish] so a rollout run matches
    # whatever the live eval protocol currently uses; override per-run if needed.
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--search-top-m", type=int, default=None)
    parser.add_argument("--search-max-depth", type=int, default=None)
    parser.add_argument("--halving-rounds", type=int, default=None)
    parser.add_argument("--search-refutation-top-r", type=int, default=None)
    parser.add_argument("--search-expand-top", type=int, default=None)
    parser.add_argument("--search-lam", type=float, default=None)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="This process's shard index, for running N parallel processes "
        "over disjoint game files. Requires --num-shards. Sharding happens "
        "at the parquet-file level (LichessDataset._shard_data_files), so "
        "the achievable parallelism is capped by how many source files "
        "cover the requested month range -- a single-month window with few "
        "underlying files won't split evenly across many shards.",
    )
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--skip-games",
        type=int,
        default=0,
        help="Skip this many games at the front of the (deterministic, "
        "unshuffled) stream before recording rollouts -- lets a later "
        "invocation continue past games an earlier one already covered, "
        "e.g. for a multi-session run stopped and resumed across days.",
    )
    parser.add_argument(
        "--flush-every-games",
        type=int,
        default=200,
        help="Write accumulated rows to --output-path every N processed "
        "games, not just once at the end -- so a kill mid-run (e.g. "
        "stopping an overnight session) only loses games since the last "
        "flush, not the whole run. Set to 0 to disable and write once at "
        "the end (the old behavior).",
    )
    return parser.parse_args()


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
    model, _ = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=False,
        require_value_head=True,
    )
    board_state_encoder = BoardStateEncoder(repo_config.board_state)

    halving_config = HalvingConfig(
        budget=int(args.search_budget if args.search_budget is not None else eval_cfg.search_budget),
        top_m=int(args.search_top_m if args.search_top_m is not None else eval_cfg.search_top_m),
        rounds=int(args.halving_rounds if args.halving_rounds is not None else eval_cfg.halving_rounds),
        refutation_top_r=int(
            args.search_refutation_top_r
            if args.search_refutation_top_r is not None
            else eval_cfg.search_refutation_top_r
        ),
        expand_top=int(
            args.search_expand_top if args.search_expand_top is not None else eval_cfg.search_expand_top
        ),
        max_depth=int(
            args.search_max_depth if args.search_max_depth is not None else eval_cfg.search_max_depth
        ),
        lam=float(args.search_lam if args.search_lam is not None else eval_cfg.value_rerank_lambda),
    )

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
        board_state_config=repo_config.board_state,
    )

    if (args.shard_id is None) != (args.num_shards is None):
        raise ValueError("--shard-id and --num-shards must both be set or both be None")

    all_rows: list[RolloutRow] = []
    games_processed = 0
    games_skipped = 0
    game_stream = lichess_dataset.stream(shard_id=args.shard_id, num_shards=args.num_shards)
    for game in tqdm(game_stream, desc="rollout-generation", unit="game"):
        if games_skipped < args.skip_games:
            # Cheap skip: never calls _process_game (the expensive search
            # path), so resuming past already-covered games is fast rather
            # than re-running search on them just to discard the result.
            games_skipped += 1
            continue

        rows = _process_game(
            game,
            model=model,
            move_vocab=move_vocab,
            board_state_encoder=board_state_encoder,
            device=device,
            dtype=dtype,
            halving_config=halving_config,
            every_n_plies=args.sample_every_n_plies,
            sample_seed=args.sample_seed,
            checkpoint_path=str(args.checkpoint),
        )
        all_rows.extend(rows)
        games_processed += 1
        if (
            args.flush_every_games > 0
            and games_processed % args.flush_every_games == 0
        ):
            write_rollout_parquet(all_rows, args.output_path)
            tqdm.write(
                f"[checkpoint] flushed {len(all_rows)} rollout rows from "
                f"{games_processed} games to {args.output_path}"
            )
        if args.max_games is not None and games_processed >= args.max_games:
            break

    write_rollout_parquet(all_rows, args.output_path)
    print(
        f"wrote {len(all_rows)} rollout rows from {games_processed} games "
        f"(skipped {games_skipped}) to {args.output_path}"
    )


if __name__ == "__main__":
    main()
