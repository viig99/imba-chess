#!/usr/bin/env python3
"""Generate search-backed rollouts from the training split.

Replays games from LichessDataset(split="train") and, at a sampled subset of
plies per game, calls the same value_search_halving search used at inference.
Each row records both the scalar value-distillation target inputs
(best/human-move backed_value, root_wdl_unsearched, real_outcome_stm) and the
full per-arm data needed for policy distillation (every searched arm's
move_uci, negamax-backed q-hat, evals spent, and policy log-prior) -- which
target(s) a training run actually uses is a downstream config choice
([expert_iteration].beta for value, a future policy-KL weight for policy),
not a generation-time one, so this script always records both. Writes one
row per sampled position to a rollout parquet consumed by scripts/train.py.
Generates data only; trains nothing (mirrors scripts/train_value_net.py's
role as a small, focused, non-Ignite script).

Usage: python scripts/generate_search_rollouts.py --checkpoint <path> --output-path <path>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
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


def _progress_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def _write_progress_sidecar(
    output_path: Path, *, games_skipped: int, games_processed: int, rows: int
) -> None:
    """Atomically records exactly how far this run got.

    A process killed mid-run (e.g. a scheduled overnight stop) never gets to
    print or return its final summary -- external tooling that needs to know
    the correct --skip-games value for the next session reads this file
    instead of parsing logs or the process's exit state.
    """
    payload = {
        "games_skipped": games_skipped,
        "games_processed": games_processed,
        "total_games_covered": games_skipped + games_processed,
        "rows": rows,
    }
    sidecar_path = _progress_sidecar_path(output_path)
    tmp_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    os.replace(tmp_path, sidecar_path)


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


def _arm_rows_to_dicts(rows: list[dict]) -> list[dict]:
    """Project every searched root arm to its minimal storage fields.

    Stores all arms search actually produced -- up to top_m by policy prior
    plus any forcing (capture/check/promotion) moves the root-level floor in
    select_value_search_halving appended beyond that cut. A fixed top_m
    truncation here would silently drop exactly those forcing arms (they're
    appended to `rows` after the top_m prior-ranked ones), discarding the
    tactical candidates a future policy-distillation target most needs.
    Rollout parquet columns are variable-length lists, so no padding is
    needed either -- a fake all-zero placeholder arm has no real move_uci
    and would corrupt any softmax-over-arms target built from this data.
    """
    return [
        {
            "move_uci": row["move_uci"],
            "backed_value": (
                float(row["backed_value"]) if row["backed_value"] is not None else 0.0
            ),
            "evals_spent": int(row["evals_spent"]),
            "policy_log_prob": float(row["policy_log_prob"]),
        }
        for row in rows
    ]


class _TimingStats:
    """Cumulative wall-clock breakdown, coarse enough to say which phase to
    attack next without touching the hot path's actual logic.

    Buckets:
      ply_bookkeeping   -- python-chess + board_state_encoder + history
                           updates, EVERY ply (sampled or not).
      batch_build       -- history.build_batch_for_current_position's tensor
                           construction, sampled plies only (pure CPU/Python,
                           no model call -- separated from ply_bookkeeping
                           since it's specific to the eval path, not every ply).
      root_eval         -- the root position's own _forward_model GPU call.
      search_gpu        -- cumulative time inside CachedPositionEvaluator.
                           evaluate() (the search's batched forward_decode
                           GPU calls across all waves of one search).
      search_bookkeeping -- select_value_search_halving's own time MINUS
                           search_gpu -- heapq/tree management, board copies
                           for candidate moves, python-chess calls inside the
                           search (_is_forcing, terminal_value_for_color, etc).
    """

    def __init__(self) -> None:
        self.games = 0
        self.positions = 0
        self.ply_bookkeeping = 0.0
        self.batch_build = 0.0
        self.root_eval = 0.0
        self.search_gpu = 0.0
        self.search_bookkeeping = 0.0
        self.search_eval_calls = 0
        self.search_eval_items = 0

    def total(self) -> float:
        return (
            self.ply_bookkeeping
            + self.batch_build
            + self.root_eval
            + self.search_gpu
            + self.search_bookkeeping
        )

    def report(self) -> str:
        total = self.total()
        if total <= 0:
            return "no timing data yet"
        buckets = [
            ("ply_bookkeeping (chess+encode, every ply)", self.ply_bookkeeping),
            ("batch_build (tensor construction, sampled plies)", self.batch_build),
            ("root_eval (root forward, GPU)", self.root_eval),
            ("search_gpu (search forward_decode waves, GPU)", self.search_gpu),
            ("search_bookkeeping (heap/tree mgmt, CPU)", self.search_bookkeeping),
        ]
        lines = [
            f"timing after {self.games} games / {self.positions} positions "
            f"({self.search_eval_calls} search waves, {self.search_eval_items} search evals, "
            f"total {total:.1f}s):"
        ]
        for name, seconds in sorted(buckets, key=lambda item: item[1], reverse=True):
            pct = 100.0 * seconds / total
            lines.append(f"  {name}: {seconds:.1f}s ({pct:.1f}%)")
        return "\n".join(lines)


class _TimedEvaluator(CachedPositionEvaluator):
    """CachedPositionEvaluator that times its own evaluate() calls, so the
    search's GPU time can be separated from its CPU tree/heap bookkeeping
    without instrumenting search.py (which stays torch-free by design)."""

    def __init__(self, *args, stats: _TimingStats, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stats = stats

    def evaluate(self, batch):
        start = time.perf_counter()
        result = super().evaluate(batch)
        self._stats.search_gpu += time.perf_counter() - start
        self._stats.search_eval_calls += 1
        self._stats.search_eval_items += len(batch)
        return result


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
    rng: random.Random,
    stats: _TimingStats | None = None,
) -> RolloutRow | None:
    t_batch_start = time.perf_counter()
    batch = history.build_batch_for_current_position(board)
    t_batch_end = time.perf_counter()
    output = _forward_model(model=model, batch=batch, device=device, dtype=dtype, return_kv=True)
    t_root_eval_end = time.perf_counter()
    if stats is not None:
        stats.batch_build += t_batch_end - t_batch_start
        stats.root_eval += t_root_eval_end - t_batch_end

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

    evaluator_kwargs = dict(
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        device=device,
        dtype=dtype,
        prefix_kv=output["kv_caches"],
        prefix_len=int(batch["total_tokens"]),
    )
    evaluator = (
        _TimedEvaluator(**evaluator_kwargs, stats=stats)
        if stats is not None
        else CachedPositionEvaluator(**evaluator_kwargs)
    )
    t_search_start = time.perf_counter()
    gpu_before = stats.search_gpu if stats is not None else 0.0
    best_local_idx, rows = select_value_search_halving(
        evaluator=evaluator,
        root_handle=None,
        board=board,
        legal_moves=legal_moves,
        legal_log_priors=legal_log_priors,
        config=halving_config,
        rng=rng,
    )
    if stats is not None:
        search_total = time.perf_counter() - t_search_start
        stats.search_bookkeeping += search_total - (stats.search_gpu - gpu_before)
        stats.positions += 1
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

    arms = _arm_rows_to_dicts(rows)
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
        search_refutation_top_r=halving_config.refutation_top_r,
        search_expand_top=halving_config.expand_top,
        search_lam=halving_config.lam,
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
    stats: _TimingStats | None = None,
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
                rng=random.Random(f"{sample_seed}:{game['game_id']}:{ply_idx}:gumbel"),
                stats=stats,
            )
            if row is not None:
                rows.append(row)
        t_bookkeeping_start = time.perf_counter()
        move = chess.Move.from_uci(play["move_uci"])
        history.append_observed_position(board)
        history.record_played_move(play["move_uci"])
        board.push(move)
        if stats is not None:
            stats.ply_bookkeeping += time.perf_counter() - t_bookkeeping_start

    if stats is not None:
        stats.games += 1
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
        "--gumbel-root-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample root arms via the Gumbel-Top-k trick instead of a "
        "deterministic top-m-by-prior cut (Danihelka et al., ICLR 2022). "
        "Default True for rollout generation, since a deterministic cut can "
        "permanently exclude a genuinely good but low-prior move from ever "
        "being searched -- exactly the blind spot a future policy-"
        "distillation target would inherit. Live eval play (scripts/"
        "eval_vs_stockfish.py) does not expose this flag and keeps the "
        "validated deterministic behavior unchanged.",
    )
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
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track and periodically print a wall-clock breakdown (ply "
        "bookkeeping / batch build / root eval / search GPU / search "
        "bookkeeping) via time.perf_counter() -- negligible overhead, off "
        "by default to keep normal run logs uncluttered.",
    )
    parser.add_argument(
        "--profile-every-games",
        type=int,
        default=25,
        help="Print the running timing breakdown every N processed games "
        "when --profile is set.",
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
        gumbel_root_sampling=bool(args.gumbel_root_sampling),
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
    stats = _TimingStats() if args.profile else None
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
            stats=stats,
        )
        all_rows.extend(rows)
        games_processed += 1
        if (
            stats is not None
            and args.profile_every_games > 0
            and games_processed % args.profile_every_games == 0
        ):
            tqdm.write(stats.report())
        if (
            args.flush_every_games > 0
            and games_processed % args.flush_every_games == 0
        ):
            write_rollout_parquet(all_rows, args.output_path)
            _write_progress_sidecar(
                args.output_path,
                games_skipped=games_skipped,
                games_processed=games_processed,
                rows=len(all_rows),
            )
            tqdm.write(
                f"[checkpoint] flushed {len(all_rows)} rollout rows from "
                f"{games_processed} games to {args.output_path}"
            )
        if args.max_games is not None and games_processed >= args.max_games:
            break

    write_rollout_parquet(all_rows, args.output_path)
    _write_progress_sidecar(
        args.output_path,
        games_skipped=games_skipped,
        games_processed=games_processed,
        rows=len(all_rows),
    )
    print(
        f"wrote {len(all_rows)} rollout rows from {games_processed} games "
        f"(skipped {games_skipped}) to {args.output_path}"
    )
    if stats is not None:
        print(stats.report())


if __name__ == "__main__":
    main()
