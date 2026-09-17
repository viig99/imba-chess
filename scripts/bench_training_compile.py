"""Compare stage-2 eager/compiled updates using an immutable replay and checkpoint.

Run eager first, then compiled with the same arguments/output directory. Eager
is the reference; the candidate compiles model/loss and builds masks separately.
"""

import argparse
from dataclasses import asdict
from functools import partial
import gc
import json
from pathlib import Path
import time

import torch

from imba_chess.data.self_play_store import SelfPlayStore, atomic_json
from imba_chess.self_play.config import load_config
from imba_chess.self_play.runtime import load_runtime
from imba_chess.self_play.seeds import file_hash
from imba_chess.self_play.trainer import Stage2Trainer, _model_loss, _training_loss


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_state(v) for v in value]
    return value


def compare(actual, expected, *, atol, rtol, path="state"):
    """Check every tensor, including relative-bias gradients and Adam moments."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                                   msg=lambda detail: f"{path}: {detail}")
        return float((actual - expected).abs().max()) if actual.numel() else 0.0
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        return max((compare(v, expected[k], atol=atol, rtol=rtol,
                            path=f"{path}.{k}") for k, v in actual.items()), default=0)
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected), path
        return max((compare(a, b, atol=atol, rtol=rtol, path=f"{path}.{i}")
                    for i, (a, b) in enumerate(zip(actual, expected))), default=0)
    assert actual == expected, path
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--exposures", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.exposures, args.repeats, args.threads) < 1:
        parser.error("exposures, repeats and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    runtime, max_positions = load_runtime(cfg, args.checkpoint, "cuda", optimized=False)
    store = SelfPlayStore(args.replay, read_only=True, **asdict(cfg.replay))
    initial = cpu_state(runtime.model.state_dict())
    inputs = dict(config=asdict(cfg), checkpoint_sha256=file_hash(args.checkpoint),
                  base_config_sha256=file_hash(Path(cfg.base_config)),
                  source_sha256={str(p): file_hash(p) for p in sorted({
                      Path(__file__),
                      *Path("src/imba_chess/model").glob("*.py"),
                      *Path("src/imba_chess/self_play").glob("*.py"),
                  })},
                  replay_shards={s: file_hash(args.replay / s) for s in store.active_shards()},
                  game_ids=store.game_ids("train"), exposures=args.exposures,
                  threads=args.threads)
    report = dict(mode=args.mode, inputs=inputs, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), dtype="float32",
                  matmul_precision=torch.get_float32_matmul_precision(),
                  allow_tf32=torch.backends.cuda.matmul.allow_tf32, trials=[])

    def trainer():
        runtime.model.load_state_dict(initial)
        torch.manual_seed(cfg.run.seed)
        torch.cuda.manual_seed_all(cfg.run.seed)
        t = Stage2Trainer(model=runtime.model, config=cfg.learning,
                          move_vocab=runtime.move_vocab, encoder=runtime.encoder,
                          device=runtime.device, max_positions=max_positions,
                          run_seed=cfg.run.seed)
        if args.mode == "compiled":
            t._loss_fn = partial(_training_loss, model_loss=torch.compile(
                _model_loss, fullgraph=True, dynamic=True))
        else:
            t._loss_fn = _training_loss
        t.begin_phase(store)
        return t

    # Correctness is separate from throughput: CPU snapshots synchronize and
    # distort timings. Three identical updates test accumulated optimizer drift.
    t = trainer()
    records, batches, gradients = [], [], {}
    next_batch = t._next_batch

    def record_batch(source):
        batch, gids = next_batch(source)
        batches.append(dict(gids=gids, tokens=batch["total_tokens"],
                            supervised=len(batch["supervised_indices"]),
                            legal_width=batch["legal_ids"].shape[1]))
        return batch, gids

    t._next_batch = record_batch

    def capture_gradients(optimizer, a, kw):
        if not gradients:
            gradients.update({n: cpu_state(p.grad) for n, p in t.model.named_parameters()
                              if p.grad is not None})

    handle = t.optimizer.register_step_pre_hook(capture_gradients)
    start = time.perf_counter()
    t.train(store, exposure_budget=10**9, should_stop=lambda: len(records) >= 3,
            on_step=lambda m: (records.append(m), print("correctness", m, flush=True)))
    torch.cuda.synchronize()
    report["correctness_seconds_including_compile_and_snapshots"] = time.perf_counter() - start
    handle.remove()
    state = dict(inputs=inputs, batches=batches, gradients=gradients,
                 model=cpu_state(t.model.state_dict()), optimizer=cpu_state(t.optimizer.state_dict()),
                 metrics=[{k: m[k] for k in ("loss", "policy_loss", "value_loss", "gradient_norm")}
                          for m in records])
    oracle = args.output / "eager-reference.pt"
    if args.mode == "eager":
        torch.save(state, oracle)
    else:
        reference = torch.load(oracle, map_location="cpu", weights_only=False)
        assert inputs == reference["inputs"]
        assert batches == reference["batches"]
        # FP32 fused reductions need not be bit-identical. These bounds are
        # fixed before the comparison, not relaxed in response to a failure.
        checks, failures = {}, []
        for key in ("gradients", "model", "optimizer"):
            try:
                checks[key] = compare(state[key], reference[key], atol=2e-5, rtol=2e-4, path=key)
            except AssertionError as error:
                failures.append(str(error))
        # Preserve the original stricter delta gate, and record every outlier.
        # Failure prevents promotion but need not prevent throughput measurement.
        maximum, delta_error_sq, delta_reference_sq, count = 0.0, 0.0, 0.0, 0
        outliers = []
        for name, value in state["model"].items():
            if not value.is_floating_point():
                continue
            actual = value - initial[name]
            expected = reference["model"][name] - initial[name]
            diff = actual - expected
            maximum = max(maximum, float(diff.abs().max()))
            delta_error_sq += float(diff.double().square().sum())
            delta_reference_sq += float(expected.double().square().sum())
            mismatches = int((~torch.isclose(actual, expected, atol=2e-6, rtol=2e-3)).sum())
            count += mismatches
            if mismatches:
                outliers.append(dict(parameter=name, count=mismatches,
                                     max_absolute_error=float(diff.abs().max())))
        if count:
            failures.append(f"Update delta gate: {count} elements outside atol=2e-6, rtol=2e-3")
        try:
            torch.testing.assert_close(torch.tensor([list(m.values()) for m in state["metrics"]]),
                                       torch.tensor([list(m.values()) for m in reference["metrics"]]),
                                       atol=2e-5, rtol=2e-4)
        except AssertionError as error:
            failures.append(str(error))
        checks.update(passed=not failures, failures=failures, update_outliers=outliers,
                      update_max_absolute_error=maximum,
                      update_relative_l2_error=(delta_error_sq / max(delta_reference_sq, 1e-300))**0.5)
        report["correctness"] = checks
        torch.save(state, args.output / "compiled-reference.pt")
        atomic_json(args.output / f"{args.mode}.json", report)
        del reference
    del state, gradients, t, next_batch
    gc.collect()
    torch.cuda.empty_cache()
    from torch._dynamo.utils import counters

    for repeat in range(args.repeats):
        t = trainer()
        records = []
        torch.cuda.reset_peak_memory_stats()
        graphs_before = counters["stats"]["unique_graphs"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        t.train(store, exposure_budget=args.exposures, on_step=records.append)
        torch.cuda.synchronize()
        row = dict(repeat=repeat, seconds=time.perf_counter() - start,
                   exposures=t.exposures, steps=records,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   new_graphs=counters["stats"]["unique_graphs"] - graphs_before)
        report["trials"].append(row)
        report["compiler_counters"] = {k: dict(v) for k, v in counters.items()}
        atomic_json(args.output / f"{args.mode}.json", report)
        print(json.dumps({k: v for k, v in row.items() if k != "steps"}), flush=True)
        del t
        gc.collect()
    print(args.output / f"{args.mode}.json", flush=True)


if __name__ == "__main__":
    main()
