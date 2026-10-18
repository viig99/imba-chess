"""Evaluate raw policy/value learning, without search or Stockfish supervision.

Identical root batches are used for the initial network and each saved learning
milestone. Reports are paired by source game and plotted against actual training
exposures and collection-plus-training wall time. Engine WDL remains a proxy.
"""

import argparse
import json
from pathlib import Path
import time

from imba_chess.data.self_play_store import atomic_json
from imba_chess.eval.golden import digest, file_sha256, read_labels, validate_labels
from imba_chess.eval.search_study import summarize, paired_regret_interval


def evaluate(checkpoint, positions, args, directory):
    import chess
    import torch
    from imba_chess.config import load_repo_config
    from imba_chess.eval import cozy_bridge
    from imba_chess.eval.batch_scheduler import BatchScheduler, WorkRequest
    from imba_chess.eval.inference_runtime import load_runtime, INFERENCE_DTYPES
    from imba_chess.eval.position_evaluator import _SequenceHistory

    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=file_sha256(checkpoint), positions_sha256=digest(positions),
                    dtype=args.dtype, concurrent_games=args.concurrent_games,
                    reference_manifest_sha256=file_sha256(args.reference / "manifest.json"),
                    evaluator_sha256=file_sha256(__file__))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("raw evaluation identity changed")
    atomic_json(manifest_path, identity)
    path = directory / "predictions.json"
    if path.exists():
        return json.loads(path.read_text())
    runtime, _ = load_runtime(repo_config=load_repo_config(args.base_config), checkpoint=checkpoint,
                              device="cuda", algorithm="policy", dtype=INFERENCE_DTYPES[args.dtype])
    rows = []

    def play(position):
        board = chess.Board()
        history = _SequenceHistory(move_vocab=runtime.move_vocab, board_state_encoder=runtime.encoder)
        for uci in position["prefix_moves"]:
            history.append_observed_position(board)
            history.record_played_move(uci)
            board.push_uci(uci)
        key = position["position_id"]
        batch = history.build_batch_for_current_position(board)
        batch["game_id"] = [key]
        owner = ("raw-policy-value", key)
        identity, output = yield WorkRequest("root_eval", (owner, batch))
        if identity != owner:
            raise RuntimeError("raw evaluation ownership mismatch")
        legal = cozy_bridge.project_legal_lean(cozy_bridge.board_to_cozy(board), runtime.move_vocab)
        logits = output["logits"][-1]
        indices = torch.tensor(legal.ids, device=logits.device)
        host = torch.cat((torch.log_softmax(logits.index_select(0, indices).float(), 0),
                          torch.softmax(output["value_logits"][-1].float(), -1))).cpu().tolist()
        logs, wdl = host[:-3], host[-3:]
        probabilities = torch.softmax(torch.tensor(logs, dtype=torch.float64), 0).tolist()
        action = max(range(len(logs)), key=logs.__getitem__)
        return dict(position_id=key, variant="raw", legal_ucis=legal.ucis,
                    move_uci=legal.ucis[action], policy=probabilities,
                    root_log_priors=logs, root_wdl=wdl, search_wdl=None,
                    visits=[0] * len(logs))

    def fail(key, error):
        raise error

    started = time.perf_counter()
    with runtime:
        for i in range(0, len(positions), args.concurrent_games):
            chunk = positions[i:i + args.concurrent_games]
            BatchScheduler(game_factory=iter((p["position_id"], play(p)) for p in chunk),
                           executors=runtime.executors, concurrent_games=args.concurrent_games,
                           on_game_done=lambda key, row: rows.append(row), on_game_error=fail,
                           completion_order=True).run()
    atomic_json(path, rows)
    atomic_json(directory / "timing.json", dict(seconds=time.perf_counter() - started, positions=len(rows)))
    del runtime
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--learning", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("tune", "test"), default="test")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrent-games", type=int, default=32)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--base-config", type=Path, default=Path("config/imba_chess_v4_aux3.toml"))
    args = p.parse_args()
    import torch

    torch.set_num_threads(1)
    manifest = json.loads(args.positions.read_text())
    labels = read_labels(args.reference / "labels.jsonl")
    validate_labels(manifest, labels)
    positions = [p for p in manifest["positions"] if p["split"] == args.split][:args.limit]
    if not positions:
        raise ValueError("empty raw-policy evaluation")
    initial = evaluate(args.initial, positions, args, args.out / "initial")
    report = dict(split=args.split, positions=len(positions),
                  initial=summarize(initial, positions, labels)["raw"], arms={},
                  interpretation="Raw network predictions without MCTS. Stockfish scores are reference estimates; one training seed does not establish long-run learning superiority.")
    completed = []
    for arm in sorted(args.learning.iterdir()):
        if not (arm / "summary.json").exists():
            continue
        study = json.loads((arm / "summary.json").read_text())
        milestones = []
        wall = 0
        for phase in study["phases"]:
            wall += phase["collection_seconds"] + phase["training_seconds"]
            checkpoint = Path(phase["checkpoint"])
            rows = evaluate(checkpoint, positions, args, args.out / arm.name / checkpoint.stem)
            entry = dict(exposures=phase["cumulative_exposures"], steps=phase["cumulative_steps"],
                         collect_train_seconds=wall, checkpoint=str(checkpoint),
                         metrics=summarize(rows, positions, labels)["raw"],
                         versus_initial={metric: paired_regret_interval(initial, rows, labels, positions,
                            metric=metric, variant="raw") for metric in ("target_regret", "move_regret")})
            milestones.append(entry)
            print(json.dumps(dict(arm=arm.name, exposures=entry["exposures"], metrics=entry["metrics"]["all"])), flush=True)
        report["arms"][arm.name] = milestones
        if study.get("complete") and milestones:
            completed.append((arm.name, rows))
        atomic_json(args.out / "summary.json", report)
    baseline = dict(completed).get("gumbel")
    if baseline is not None:
        report["final_vs_gumbel"] = {name: {metric: paired_regret_interval(baseline, rows, labels, positions,
                metric=metric, variant="raw") for metric in ("target_regret", "move_regret")}
                for name, rows in completed if name != "gumbel"}
    atomic_json(args.out / "summary.json", report)


if __name__ == "__main__":
    main()
