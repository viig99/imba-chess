"""KLENT loop: collect an iteration with frozen weights, fit one epoch, checkpoint.

Resumable at iteration boundaries; rollout sampling and batch packing are
reseeded from (seed, iteration). Execution and fitting settings may be overridden.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from imba_chess.config import load_repo_config
from imba_chess.data.move_vocab import MoveVocab
from imba_chess.eval.inference_runtime import INFERENCE_DTYPES
from imba_chess.model import HSTUChessModel, build_hstu_chess_config
from imba_chess.model.checkpoint import load_initial_weights
from imba_chess.self_play.runtime import run_lock
from imba_chess.self_play.trainer import atomic_checkpoint

from .engine import SlotEngine
from .selfplay import BoardCodec, SelfPlay
from .targets import lambda_from_tau
from .train import KlentTrainer


# These overrides affect execution or fitting, without reinterpreting stored
# model weights or changing the return/policy target definitions.
_RESUME_OVERRIDES = frozenset({
    "init", "total_positions", "batch_tokens", "slots", "compile",
    "inference_dtype", "lr", "warmup_lr", "weight_decay", "grad_clip",
})


def _validate_resume_config(cfg, state):
    current, saved = asdict(cfg), state["config"]
    changed = sorted(
        key for key in current.keys() | saved.keys()
        if key not in _RESUME_OVERRIDES and current.get(key) != saved.get(key)
    )
    if changed:
        raise ValueError(f"incompatible KLENT resume config: {', '.join(changed)}")


def build_model(cfg, repo, vocab, *, resume_state=None):
    model_config = replace(
        build_hstu_chess_config(repo.model, move_vocab_size=len(vocab)),
        enable_value_head=True,
        enable_auxiliary_value_head=False,
        enable_action_value_head=True,
        tie_policy_embeddings=cfg.init != "scratch",
    )
    if resume_state is not None:
        # Legacy KLENT checkpoints always used tied policy/input weights.
        saved_config = resume_state.get("model_config")
        model_config = replace(
            model_config,
            tie_policy_embeddings=(
                saved_config.get("tie_policy_embeddings", True) if saved_config else True
            ),
        )
        if saved_config is not None and saved_config != asdict(model_config):
            raise ValueError("checkpoint model architecture does not match the base config")
    model = HSTUChessModel(model_config)
    if resume_state is None and cfg.init == "scratch":
        # Untying first keeps the previous-move embeddings informative while
        # starting with Q=0 and a uniform policy, as in the KLENT reference.
        torch.nn.init.zeros_(model.prediction_head.weight)
    elif resume_state is None:
        checkpoint = torch.load(cfg.init, map_location="cpu", weights_only=False)
        load_initial_weights(model, checkpoint, allow_missing_prefixes=("action_value_head.",))
    return model


class KlentRun:
    def __init__(self, cfg, *, device, resume_state=None):
        self.cfg, self.device = cfg, torch.device(device)
        if resume_state is not None:
            _validate_resume_config(cfg, resume_state)
        self.repo = load_repo_config(cfg.base_config)
        self.vocab = MoveVocab.load(self.repo.vocab.path)
        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg, self.repo, self.vocab, resume_state=resume_state).to(self.device)
        self.dtype = INFERENCE_DTYPES[cfg.inference_dtype]
        self.inference = HSTUChessModel(self.model.config).to(self.device, self.dtype).eval()
        self.codec = BoardCodec(self.vocab, self.repo.board_state)
        self.trainer = KlentTrainer(
            self.model,
            device=self.device,
            start_id=self.vocab.start_id,
            batch_tokens=cfg.batch_tokens,
            grad_clip=cfg.grad_clip,
            compile_model=cfg.compile,
        )
        self.iteration = self.positions = 0
        if resume_state is not None:
            self.load_state_dict(resume_state)

    def _phase(self):
        warmup = self.iteration < self.cfg.q_warmup_iterations
        lr = self.cfg.warmup_lr if warmup else self.cfg.lr
        self.trainer.set_phase(warmup=warmup, lr=lr, weight_decay=self.cfg.weight_decay)
        return warmup, lr

    def collect(self, positions, *, bootstrap):
        self.inference.load_state_dict(self.model.state_dict())
        engine = SlotEngine(
            self.inference,
            slots=self.cfg.slots,
            start_id=self.vocab.start_id,
            device=self.device,
            dtype=self.dtype,
        )
        generator = torch.Generator(self.device).manual_seed(
            self.cfg.seed * 1_000_003 + self.iteration
        )
        selfplay = SelfPlay(
            engine,
            self.codec,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            lam=lambda_from_tau(self.cfg.tau),
            max_plies=self.cfg.max_plies,
            start_id=self.vocab.start_id,
            generator=generator,
        )
        try:
            return selfplay.collect(positions, bootstrap=bootstrap)
        finally:
            del engine, selfplay
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def run_iteration(self):
        warmup, lr = self._phase()
        start = time.perf_counter()
        games, metrics = self.collect(
            self.cfg.positions_per_iteration,
            bootstrap="value" if warmup else self.cfg.bootstrap,
        )
        metrics["time/selfplay"] = time.perf_counter() - start
        metrics.update(
            self.trainer.train_epoch(
                games,
                np.random.default_rng([self.cfg.seed, self.iteration]),
                policy_weight=0.0 if warmup else 1.0,
                value_weight=self.cfg.value_weight,
            )
        )
        self.iteration += 1
        self.positions += metrics["selfplay/positions_played"]
        metrics.update(
            iteration=self.iteration,
            positions=self.positions,
            warmup=warmup,
            learning_rate=lr,
            positions_per_second=metrics["selfplay/positions_played"] / metrics["time/selfplay"],
        )
        return metrics

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            model_config=asdict(self.model.config),
            trainer=self.trainer.state_dict(),
            iteration=self.iteration,
            positions=self.positions,
            config=asdict(self.cfg),
        )

    def load_state_dict(self, state):
        _validate_resume_config(self.cfg, state)
        self.model.load_state_dict(state["model"])
        self.iteration, self.positions = state["iteration"], state["positions"]
        warmup = self.iteration < self.cfg.q_warmup_iterations
        lr = self.cfg.warmup_lr if warmup else self.cfg.lr
        self.trainer.load_state_dict(state["trainer"], lr=lr, weight_decay=self.cfg.weight_decay)


def run(cfg, *, output, device, save_every=5):
    output = Path(output)
    with run_lock(output):
        state_path = output / "checkpoint.pt"
        state = (
            torch.load(state_path, map_location="cpu", weights_only=False)
            if state_path.exists() else None
        )
        klent = KlentRun(cfg, device=device, resume_state=state)
        if state is not None:
            print(f"resumed at iteration {klent.iteration}, positions {klent.positions}")
        del state
        while klent.positions < cfg.total_positions:
            metrics = klent.run_iteration()
            atomic_checkpoint(state_path, klent.state_dict())
            if klent.iteration % save_every == 0 or klent.positions >= cfg.total_positions:
                # Model-only snapshot in the format the eval scripts load.
                atomic_checkpoint(
                    output / f"actor-{klent.iteration:04d}.pt",
                    dict(model=klent.model.state_dict(), model_config=asdict(klent.model.config)),
                )
            line = json.dumps(metrics, allow_nan=False, sort_keys=True)
            print(line, flush=True)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")
