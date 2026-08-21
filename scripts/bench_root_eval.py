"""Microbenchmark the root-eval forward that dominates rollout generation.

Motivation: the search-budget sweep (artifacts/budget_sweep) showed root_eval
is budget-INDEPENDENT (33.5s -> 23.9s while search evals fell 64x), so it is
the floor on rollout throughput -- 80% of all time at budget 32. Before
optimizing it we need to know where the time goes, because the last attempt
at this (the reverted _IncrementalRootCache, 2026-07-15) optimized an assumed
cost without profiling and lost.

Splits one root forward into:
  blockmask -- create_batch_block_mask alone (compiled flex block-mask build)
  model     -- the model forward alone, block mask prebuilt
and sweeps the merged batch shapes the G-game scheduler actually submits
(G games x each game's current ply count).

Run: .venv/bin/python scripts/bench_root_eval.py [--compile]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from imba_chess.config import load_repo_config
from imba_chess.data.move_vocab import load_or_create_static_move_vocab
from imba_chess.eval.position_evaluator import _autocast_context, load_hstu_checkpoint
from imba_chess.model.hstu_model import create_batch_block_mask

SHAPES = [
    (1, 40),
    (1, 80),
    (1, 160),
    (8, 40),
    (8, 80),
    (8, 160),
    (32, 80),
    (64, 80),
]


def _timeit(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _fake_batch(*, games: int, plies: int, device: torch.device) -> dict[str, Any]:
    """A jagged batch of `games` sequences of `plies` tokens each.

    Mirrors _SequenceHistory._build_single_batch's keys/dtypes, with `games`
    sequences merged the way _merge_root_batches does for the G-game
    scheduler. Token *values* are irrelevant to timing (the forward has no
    data-dependent control flow), so zeros are used.
    """
    total = games * plies

    def z() -> torch.Tensor:
        return torch.zeros(total, dtype=torch.long, device=device)

    return {
        "game_id": ["bench"] * games,
        "game_result_white": torch.zeros(games, dtype=torch.long, device=device),
        "num_games": games,
        "total_tokens": total,
        "seq_lens": torch.full((games,), plies, dtype=torch.long, device=device),
        "seq_offsets": torch.arange(
            0, total + 1, plies, dtype=torch.long, device=device
        ),
        "piece_ids": torch.zeros((total, 64), dtype=torch.long, device=device),
        "seq_token_id": z(),
        "turn_id": z(),
        "castle_id": z(),
        "ep_file_id": z(),
        "halfmove_bucket_id": z(),
        "fullmove_bucket_id": z(),
        "prev_move_id": z(),
        "target_move_id": z(),
        "played_by_elo": z(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/imba_chess_exit_seeded_rollout.toml"),
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/best_hr10_checkpoint_23_hr10=0.9564.pt"),
    )
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load with compile_model=True. The rollout generator hardcodes "
        "False (scripts/generate_search_rollouts.py:563), so this measures "
        "what that choice costs.",
    )
    args = ap.parse_args()

    repo_config = load_repo_config(args.config)
    device = torch.device("cuda")
    dtype = torch.float32

    move_vocab = load_or_create_static_move_vocab(
        path=repo_config.vocab.path, include_unk=repo_config.vocab.include_unk
    )
    model, _ = load_hstu_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_config=repo_config,
        move_vocab=move_vocab,
        device=device,
        compile_model=bool(args.compile),
        require_value_head=True,
    )
    model.eval()

    print(f"device={device} dtype={dtype} compile={bool(args.compile)}")
    print(
        f"{'games':>6} {'plies':>6} {'tokens':>7} {'blockmask_ms':>13} "
        f"{'model_ms':>9} {'total_ms':>9} {'ms/game':>8} {'bm_share':>9}"
    )

    for games, plies in SHAPES:
        batch = _fake_batch(games=games, plies=plies, device=device)
        total = int(batch["total_tokens"])
        seq_offsets = batch["seq_offsets"]

        bm_ms = _timeit(
            lambda: create_batch_block_mask(
                seq_offsets=seq_offsets, total_tokens=total, device=device
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        block_mask = create_batch_block_mask(
            seq_offsets=seq_offsets, total_tokens=total, device=device
        )

        def model_only(batch=batch, block_mask=block_mask):
            with torch.inference_mode(), _autocast_context(device, dtype):
                return model(
                    batch, block_mask=block_mask, return_loss=False, return_kv=True
                )

        md_ms = _timeit(model_only, warmup=args.warmup, iters=args.iters)

        tot = bm_ms + md_ms
        print(
            f"{games:6d} {plies:6d} {total:7d} {bm_ms:13.2f} "
            f"{md_ms:9.2f} {tot:9.2f} {tot / games:8.2f} {bm_ms / tot:8.1%}"
        )


if __name__ == "__main__":
    main()
