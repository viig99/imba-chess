#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch

from imba_chess.config import DEFAULT_CONFIG_PATH, load_repo_config
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.event_builder import (
    BOS_TOKEN_ID,
    EVENT_TOKEN_ID,
    TARGET_IGNORE_INDEX,
)
from imba_chess.data.move_vocab import MoveVocab, load_or_create_static_move_vocab
from imba_chess.model import (
    HSTUChessModel,
    build_hstu_chess_config,
    create_batch_block_mask,
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


class _SequenceHistory:
    """Incrementally builds the BOS+event sequence used for model inference."""

    def __init__(
        self, *, move_vocab: MoveVocab, board_state_encoder: BoardStateEncoder
    ) -> None:
        self._move_vocab = move_vocab
        self._board_state_encoder = board_state_encoder

        self.seq_token_id: list[int] = [BOS_TOKEN_ID]
        self.piece_ids: list[list[int]] = [[0] * 64]
        self.turn_id: list[int] = [0]
        self.castle_id: list[int] = [0]
        self.ep_file_id: list[int] = [0]
        self.halfmove_bucket_id: list[int] = [0]
        self.fullmove_bucket_id: list[int] = [0]
        self.prev_move_id: list[int] = [self._move_vocab.start_id]
        self.target_move_id: list[int] = [TARGET_IGNORE_INDEX]
        self.played_by_elo: list[int] = [0]

        self._prev_move_id_for_next_token = self._move_vocab.start_id

    def append_observed_position(self, board: chess.Board) -> None:
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)

    def record_played_move(self, move_uci: str) -> None:
        self._prev_move_id_for_next_token = int(self._move_vocab.encode(move_uci))

    def _append_from_state(self, state) -> None:
        self.seq_token_id.append(EVENT_TOKEN_ID)
        self.piece_ids.append(list(state.piece_ids))
        self.turn_id.append(int(state.turn_id))
        self.castle_id.append(int(state.castle_id))
        self.ep_file_id.append(int(state.ep_file_id))
        self.halfmove_bucket_id.append(int(state.halfmove_bucket_id))
        self.fullmove_bucket_id.append(int(state.fullmove_bucket_id))
        self.prev_move_id.append(int(self._prev_move_id_for_next_token))
        self.target_move_id.append(TARGET_IGNORE_INDEX)
        self.played_by_elo.append(0)

    def _pop_last(self) -> None:
        self.seq_token_id.pop()
        self.piece_ids.pop()
        self.turn_id.pop()
        self.castle_id.pop()
        self.ep_file_id.pop()
        self.halfmove_bucket_id.pop()
        self.fullmove_bucket_id.pop()
        self.prev_move_id.pop()
        self.target_move_id.pop()
        self.played_by_elo.pop()

    def _build_single_batch(self) -> dict[str, Any]:
        # Single-sequence jagged batch; avoids collate list-copy overhead.
        total_tokens = len(self.seq_token_id)
        return {
            "game_id": ["stockfish_eval"],
            "num_games": 1,
            "total_tokens": total_tokens,
            "seq_lens": torch.tensor([total_tokens], dtype=torch.long),
            "seq_offsets": torch.tensor([0, total_tokens], dtype=torch.long),
            "piece_ids": torch.tensor(self.piece_ids, dtype=torch.long),
            "seq_token_id": torch.tensor(self.seq_token_id, dtype=torch.long),
            "turn_id": torch.tensor(self.turn_id, dtype=torch.long),
            "castle_id": torch.tensor(self.castle_id, dtype=torch.long),
            "ep_file_id": torch.tensor(self.ep_file_id, dtype=torch.long),
            "halfmove_bucket_id": torch.tensor(
                self.halfmove_bucket_id, dtype=torch.long
            ),
            "fullmove_bucket_id": torch.tensor(
                self.fullmove_bucket_id, dtype=torch.long
            ),
            "prev_move_id": torch.tensor(self.prev_move_id, dtype=torch.long),
            "target_move_id": torch.tensor(self.target_move_id, dtype=torch.long),
            "played_by_elo": torch.tensor(self.played_by_elo, dtype=torch.long),
        }

    def build_batch_for_current_position(self, board: chess.Board) -> dict[str, Any]:
        # Add transient current-position token for next-move prediction only.
        state = self._board_state_encoder.encode(board)
        self._append_from_state(state)
        try:
            return self._build_single_batch()
        finally:
            self._pop_last()


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
        choices=["greedy", "sample"],
        default=None,
        help="Model move selection on legal moves.",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help="Softmax temperature for sampled policy.",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=None,
        help="Top-k truncation before sampling (0 disables).",
    )
    parser.add_argument(
        "--sample-top-p",
        type=float,
        default=None,
        help="Top-p nucleus truncation before sampling in (0, 1].",
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
    parser.add_argument("--output-json", type=Path, default=None)
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


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _load_model(
    *,
    checkpoint_path: Path,
    repo_config,
    move_vocab: MoveVocab,
    device: torch.device,
    compile_model: bool,
) -> tuple[torch.nn.Module, bool]:
    model_cfg = build_hstu_chess_config(
        repo_config.model,
        move_vocab_size=len(move_vocab),
    )
    model: torch.nn.Module = HSTUChessModel(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint must be a model state_dict or Ignite checkpoint containing key 'model'."
        )
    normalized_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint state_dict keys must be strings")
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        normalized_state_dict[new_key] = value
    model.load_state_dict(normalized_state_dict, strict=True)
    model.eval()
    compile_enabled = False
    if compile_model:
        attention_dim = int(model_cfg.attention_dim)
        if not _is_power_of_two(attention_dim):
            print(
                "torch.compile disabled for eval: "
                f"model attention_dim={attention_dim} is not a power of two; "
                "this can fail Triton codegen in inference kernels."
            )
        else:
            model = torch.compile(model, dynamic=True, fullgraph=False)
            compile_enabled = True
    return model, compile_enabled


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
    device: torch.device,
    dtype: torch.dtype,
    policy: str,
    sample_temperature: float,
    sample_top_k: int,
    sample_top_p: float,
    debug_topk: int = 0,
) -> tuple[chess.Move, dict[str, Any]]:
    seq_offsets = batch["seq_offsets"].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    block_mask = create_batch_block_mask(
        seq_offsets=seq_offsets,
        total_tokens=int(batch["total_tokens"]),
        device=device,
    )
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_amp
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        output = model(batch, block_mask=block_mask, return_loss=False)

    logits = output["logits"][-1]
    legal_moves = list(board.legal_moves)
    legal_move_ids: list[int] = []
    legal_moves_with_ids: list[chess.Move] = []
    for move in legal_moves:
        move_id = move_vocab.token_to_id.get(move.uci())
        if move_id is not None:
            legal_move_ids.append(int(move_id))
            legal_moves_with_ids.append(move)
    total_legal = len(legal_moves)
    mapped_legal = len(legal_move_ids)
    if not legal_move_ids:
        raise RuntimeError(
            "No legal moves mapped to vocab ids for current board "
            f"(total legal={total_legal})."
        )

    legal_ids_tensor = torch.tensor(
        legal_move_ids, device=logits.device, dtype=torch.long
    )
    legal_logits = logits.index_select(0, legal_ids_tensor)
    probs_for_debug: torch.Tensor | None = None
    if policy == "greedy":
        chosen_index = int(torch.argmax(legal_logits).item())
    else:
        filtered_logits = legal_logits / float(sample_temperature)
        if sample_top_k > 0 and sample_top_k < mapped_legal:
            topk_values, _ = torch.topk(filtered_logits, k=int(sample_top_k))
            kth_value = topk_values[-1]
            filtered_logits = torch.where(
                filtered_logits >= kth_value,
                filtered_logits,
                torch.full_like(filtered_logits, float("-inf")),
            )
        if sample_top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(filtered_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=0)
            cumsum_probs = torch.cumsum(sorted_probs, dim=0)
            remove_mask = cumsum_probs > float(sample_top_p)
            remove_mask[0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            filtered = torch.full_like(filtered_logits, float("-inf"))
            filtered.scatter_(0, sorted_idx, sorted_logits)
            filtered_logits = filtered
        probs = torch.softmax(filtered_logits, dim=0)
        if not bool(torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0.0:
            chosen_index = int(torch.argmax(legal_logits).item())
        else:
            chosen_index = int(torch.multinomial(probs, num_samples=1).item())
            probs_for_debug = probs
    debug: dict[str, Any] = {
        "total_legal_moves": total_legal,
        "mapped_legal_moves": mapped_legal,
        "coverage": (mapped_legal / total_legal) if total_legal > 0 else float("nan"),
        "policy": policy,
    }
    if debug_topk > 0:
        k = min(int(debug_topk), mapped_legal)
        top_values, top_indices = torch.topk(legal_logits, k=k, largest=True)
        debug["topk_legal"] = [
            {
                "move_uci": legal_moves_with_ids[int(local_idx)].uci(),
                "logit": float(value.item()),
                "prob": (
                    float(probs_for_debug[int(local_idx)].item())
                    if probs_for_debug is not None
                    else None
                ),
            }
            for value, local_idx in zip(top_values, top_indices)
        ]
    debug["selected_prob"] = (
        float(probs_for_debug[chosen_index].item())
        if probs_for_debug is not None
        else None
    )
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
    sample_temperature: float,
    sample_top_k: int,
    sample_top_p: float,
    opening_random_plies: int,
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
            "sample_temperature": float(sample_temperature),
            "sample_top_k": int(sample_top_k),
            "sample_top_p": float(sample_top_p),
            "opening_random_plies": int(opening_random_plies),
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


def _run_segment(
    *,
    engine: chess.engine.SimpleEngine,
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
    sample_temperature: float,
    sample_top_k: int,
    sample_top_p: float,
    opening_random_plies: int,
    debug_trace_games: int,
    debug_trace_max_plies: int,
    debug_topk: int,
) -> EvalSummary:
    summary = EvalSummary()
    with tqdm(
        total=games,
        desc=f"stockfish-eval[{segment_name}]",
        unit="game",
        dynamic_ncols=True,
    ) as progress:
        for game_idx in range(games):
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
                    move, debug_info = _select_model_move(
                        model=model,
                        batch=batch,
                        board=board,
                        move_vocab=move_vocab,
                        device=device,
                        dtype=dtype,
                        policy=model_move_policy,
                        sample_temperature=sample_temperature,
                        sample_top_k=sample_top_k,
                        sample_top_p=sample_top_p,
                        debug_topk=debug_topk,
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
                        selected_prob = debug_info.get("selected_prob")
                        prob_text = (
                            f" selected_prob={selected_prob:.4f}"
                            if isinstance(selected_prob, float)
                            else ""
                        )
                        tqdm.write(
                            f"[debug][{segment_name}] game={game_idx + 1} ply={plies + 1} turn={turn} "
                            f"coverage={coverage:.3f} selected={move.uci()}{prob_text}"
                        )
                        topk = debug_info.get("topk_legal")
                        if isinstance(topk, list) and topk:
                            topk_str = ", ".join(
                                (
                                    f"{entry['move_uci']}:{entry['logit']:.3f}"
                                    if entry.get("prob") is None
                                    else f"{entry['move_uci']}:{entry['logit']:.3f}|p={entry['prob']:.3f}"
                                )
                                for entry in topk
                            )
                            tqdm.write(f"[debug][{segment_name}]   topk={topk_str}")
                else:
                    result = engine.play(board, engine_limit)
                    if result.move is None:
                        raise RuntimeError("Stockfish returned no move.")
                    move = result.move

                history.append_observed_position(board)
                history.record_played_move(move.uci())
                board.push(move)
                plies += 1

            result = board.result(claim_draw=True) if completed else "*"
            _update_summary(
                summary,
                result=result,
                model_color=model_color,
                completed=completed,
                plies=plies,
            )

            progress.update(1)
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
    return summary


def _merge_summaries(summaries: list[EvalSummary]) -> EvalSummary:
    merged = EvalSummary()
    for summary in summaries:
        merged.games += summary.games
        merged.completed_games += summary.completed_games
        merged.wins += summary.wins
        merged.losses += summary.losses
        merged.draws += summary.draws
        merged.games_as_white += summary.games_as_white
        merged.games_as_black += summary.games_as_black
        merged.wins_as_white += summary.wins_as_white
        merged.losses_as_white += summary.losses_as_white
        merged.draws_as_white += summary.draws_as_white
        merged.wins_as_black += summary.wins_as_black
        merged.losses_as_black += summary.losses_as_black
        merged.draws_as_black += summary.draws_as_black
        merged.incomplete_games += summary.incomplete_games
        merged.total_plies += summary.total_plies
        merged.model_turns += summary.model_turns
        merged.legal_moves_total += summary.legal_moves_total
        merged.legal_moves_mapped_total += summary.legal_moves_mapped_total
        merged.turns_with_no_vocab_legal_move += summary.turns_with_no_vocab_legal_move
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
    args.sample_temperature = float(
        eval_cfg.sample_temperature
        if args.sample_temperature is None
        else args.sample_temperature
    )
    args.sample_top_k = int(
        eval_cfg.sample_top_k if args.sample_top_k is None else args.sample_top_k
    )
    args.sample_top_p = float(
        eval_cfg.sample_top_p if args.sample_top_p is None else args.sample_top_p
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
    if args.sample_top_k < 0:
        raise ValueError("--sample-top-k must be >= 0")
    if float(args.sample_temperature) <= 0.0:
        raise ValueError("--sample-temperature must be > 0")
    if not (0.0 < float(args.sample_top_p) <= 1.0):
        raise ValueError("--sample-top-p must be in (0, 1]")
    if args.model_move_policy not in {"greedy", "sample"}:
        raise ValueError("--model-move-policy must be one of: greedy, sample")
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

    model, compile_enabled = _load_model(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=bool(args.compile),
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
        f"{args.model_move_policy}, temp={args.sample_temperature}, "
        f"top_k={args.sample_top_k}, top_p={args.sample_top_p}, "
        f"opening_random_plies={args.opening_random_plies}"
    )

    segment_results: list[dict[str, Any]] = []
    segment_summaries: list[EvalSummary] = []

    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish_path)) as engine:
        for spec in segment_specs:
            segment_options = _build_segment_options(
                base_threads=args.stockfish_threads,
                base_hash_mb=args.stockfish_hash_mb,
                spec=spec,
            )
            engine.configure(segment_options)
            print(
                f"\nRunning segment '{spec.name}' "
                f"(games={spec.games}, options={segment_options})"
            )
            segment_summary = _run_segment(
                engine=engine,
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
                sample_temperature=float(args.sample_temperature),
                sample_top_k=int(args.sample_top_k),
                sample_top_p=float(args.sample_top_p),
                opening_random_plies=int(args.opening_random_plies),
                debug_trace_games=max(0, int(args.debug_trace_games)),
                debug_trace_max_plies=max(0, int(args.debug_trace_max_plies)),
                debug_topk=max(0, int(args.debug_topk)),
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
                sample_temperature=float(args.sample_temperature),
                sample_top_k=int(args.sample_top_k),
                sample_top_p=float(args.sample_top_p),
                opening_random_plies=int(args.opening_random_plies),
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
        sample_temperature=float(args.sample_temperature),
        sample_top_k=int(args.sample_top_k),
        sample_top_p=float(args.sample_top_p),
        opening_random_plies=int(args.opening_random_plies),
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


if __name__ == "__main__":
    main()
