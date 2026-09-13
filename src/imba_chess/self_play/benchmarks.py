"""Component benchmarks using the same search, inference, replay and trainer APIs."""

from dataclasses import asdict, replace
import math
import random
import resource
import time

import chess
import torch

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.eval import cozy_bridge
from imba_chess.eval.batch_scheduler import BatchScheduler
from imba_chess.eval.gumbel_search import select_gumbel
from imba_chess.eval.position_evaluator import _SequenceHistory
from imba_chess.eval.search import PositionEval
from .dataset import reconstruct
from .trainer import Stage2Trainer


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def histories(seeds, runtime):
    for seed in seeds:
        history = _SequenceHistory(
            move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder
        )
        board = chess.Board()
        for uci in seed.prefix_moves:
            history.append_observed_position(board)
            history.record_played_move(uci)
            board.push_uci(uci)
        yield seed, board, history


class UniformEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab

    def extend(self, handle, uci, move_vocab_id=None):
        return (handle or ()) + (uci,)

    def evaluate(self, batch):
        rows = []
        for _, board in batch:
            ids, moves, ucis, forcing, _ = cozy_bridge.project_legal_moves(
                board, self.vocab
            )
            rows.append(
                PositionEval(
                    0.0, moves, ucis, [-math.log(len(ids))] * len(ids), forcing, ids
                )
            )
        return rows


def benchmark_component(args, cfg, runtime, seeds, max_positions, actor_id):
    """Every component writes raw trials. No component result adopts an optimization."""
    rows = []
    selected = seeds[: args.games]
    if args.component in ("replay", "training"):
        if args.replay is None:
            raise ValueError("--replay is required for replay/training components")
        store = SelfPlayStore(args.replay, read_only=True, **asdict(cfg.replay))
        ids = store.game_ids("train")
        if not ids:
            raise ValueError("replay has no active training trajectories")
        initial_weights = (
            {
                key: value.detach().cpu().clone()
                for key, value in runtime.model.state_dict().items()
            }
            if args.component == "training"
            else None
        )
        for repeat in range(args.repeats + 1):
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            if args.component == "replay":
                destination = SelfPlayStore(
                    args.output / f"replay-r{repeat}", **asdict(cfg.replay)
                )
                start = time.perf_counter()
                positions = 0
                for gid in ids:
                    game = store.read_game(gid)
                    destination.add(game)
                    positions += len(game["moves"])
                destination.flush()
                write_seconds = time.perf_counter() - start
                start = time.perf_counter()
                for gid in destination.game_ids("train"):
                    reconstruct(
                        destination.read_game(gid),
                        move_vocab=runtime.move_vocab,
                        encoder=runtime.encoder,
                        max_positions=max_positions,
                    )
                seconds = time.perf_counter() - start
                row = dict(
                    component="replay",
                    repeat=repeat,
                    warmed=repeat > 0,
                    positions=positions,
                    write_seconds=write_seconds,
                    read_reconstruct_seconds=seconds,
                    bytes=sum(
                        p.stat().st_size
                        for p in destination.directory.glob("*.parquet")
                    ),
                )
            else:
                runtime.model.load_state_dict(initial_weights)
                trainer = Stage2Trainer(
                    model=runtime.model,
                    config=cfg.learning,
                    move_vocab=runtime.move_vocab,
                    encoder=runtime.encoder,
                    device=runtime.device,
                    max_positions=max_positions,
                    run_seed=cfg.run.seed,
                )
                trainer.begin_phase(store)
                steps = []
                sync(runtime.device)
                start = time.perf_counter()
                trainer.train(
                    store,
                    exposure_budget=args.exposures,
                    should_stop=lambda: time.perf_counter() - start >= args.seconds,
                    on_step=steps.append,
                )
                sync(runtime.device)
                seconds = time.perf_counter() - start
                row = dict(
                    component="training",
                    repeat=repeat,
                    warmed=repeat > 0,
                    seconds=seconds,
                    exposures=trainer.exposures,
                    supervised_positions_per_second=trainer.exposures / seconds,
                    context_tokens_per_second=sum(s["context_tokens"] for s in steps)
                    / seconds,
                    steps=steps,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(runtime.device)
                    if runtime.device.type == "cuda"
                    else None,
                )
                del trainer
            row["peak_host_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            rows.append(row)
            atomic_json(args.output / "components.json", rows)
        return
    budgets = (
        [1, 2, 3, 16, 32, 64, 128, 256]
        if args.component == "controller"
        else list(map(int, args.simulations.split(",")))
    )
    concurrencies = (
        [1]
        if args.component == "controller"
        else list(map(int, args.concurrency.split(",")))
    )
    for budget in budgets:
        for concurrency in concurrencies:
            for repeat in range(args.repeats + 1):
                search_config = replace(
                    cfg.search,
                    simulations=budget,
                    top_m=int(args.candidates.split(",")[0]),
                )
                sync(runtime.device)
                start = time.perf_counter()
                result_rows = []
                lengths = []
                if args.component == "controller":
                    boards = [
                        chess.Board(),
                        chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
                    ]
                    for board in boards:
                        result = select_gumbel(
                            evaluator=UniformEvaluator(runtime.move_vocab),
                            board=board,
                            config=search_config,
                            rng=random.Random(cfg.run.seed),
                        )
                        result_rows.append(asdict(result))
                elif args.component == "root":
                    payloads = []
                    for seed, board, history in histories(selected, runtime):
                        batch = history.build_batch_for_current_position(board)
                        if batch["total_tokens"] + cfg.search.max_depth > max_positions:
                            raise ValueError("seed exceeds context guard")
                        lengths.append(batch["total_tokens"])
                        payloads.append(((actor_id, seed.seed_id), batch))
                    for offset in range(0, len(payloads), concurrency):
                        output = runtime.executors["root_eval"](
                            payloads[offset : offset + concurrency]
                        )
                        del output
                else:
                    # Search and leaf use actual sequential requests and real prefixes.
                    # For leaf attribution, synchronize each service call; do not
                    # compare these intrusive component timings to collector throughput.
                    seconds_by_kind = {}

                    def timed(kind, execute):
                        def call(payloads):
                            sync(runtime.device)
                            t = time.perf_counter()
                            out = execute(payloads)
                            sync(runtime.device)
                            seconds_by_kind[kind] = (
                                seconds_by_kind.get(kind, 0) + time.perf_counter() - t
                            )
                            return out

                        return call

                    executors = (
                        {
                            kind: timed(kind, fn)
                            for kind, fn in runtime.executors.items()
                        }
                        if args.component == "leaf"
                        else runtime.executors
                    )

                    def factory():
                        for seed, board, history in histories(selected, runtime):
                            if (
                                len(history.seq_token_id) + 1 + cfg.search.max_depth
                                > max_positions
                            ):
                                raise ValueError("seed exceeds context guard")
                            lengths.append(len(history.seq_token_id) + 1)
                            yield (
                                seed.seed_id,
                                runtime.search(
                                    board=board,
                                    history=history,
                                    actor_id=actor_id,
                                    game_id=seed.seed_id,
                                    config=search_config,
                                    rng=random.Random(f"{cfg.run.seed}:{seed.seed_id}"),
                                    should_stop=lambda: time.perf_counter() - start
                                    >= args.seconds,
                                ),
                            )

                    errors = []
                    BatchScheduler(
                        game_factory=iter(factory()),
                        executors=executors,
                        concurrent_games=concurrency,
                        completion_order=True,
                        on_game_done=lambda gid, r: result_rows.append(
                            dict(seed_id=gid, **asdict(r))
                        )
                        if r is not None
                        else None,
                        on_game_error=lambda gid, exc: errors.append(f"{gid}: {exc}"),
                    ).run()
                sync(runtime.device)
                elapsed = time.perf_counter() - start
                row = dict(
                    component=args.component,
                    budget=budget,
                    concurrency=concurrency,
                    repeat=repeat,
                    warmed=repeat > 0,
                    seconds=elapsed,
                    root_lengths=lengths,
                    positions=len(result_rows)
                    if args.component != "root"
                    else len(lengths),
                    simulations=sum(r["simulations"] for r in result_rows),
                    neural_evaluations=sum(
                        r["neural_evaluations"] for r in result_rows
                    ),
                    terminal_hits=sum(r["terminal_hits"] for r in result_rows),
                    maximum_depth=max(
                        (r["maximum_depth"] for r in result_rows), default=0
                    ),
                )
                if args.component in ("leaf", "search"):
                    row.update(service_seconds=seconds_by_kind, errors=errors)
                if getattr(args, "save_search_results", False):
                    row["results"] = result_rows
                if runtime.device.type == "cuda":
                    row["peak_vram_bytes"] = torch.cuda.max_memory_allocated(
                        runtime.device
                    )
                rows.append(row)
                atomic_json(args.output / "components.json", rows)
                print({k: v for k, v in row.items() if k != "results"}, flush=True)
