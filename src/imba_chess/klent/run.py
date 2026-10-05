"""KLENT loop: collect an iteration with frozen weights, fit one epoch, checkpoint.

Resumable at iteration boundaries; every RNG is reseeded from (seed, iteration),
so a resumed run replays exactly what an uninterrupted one would have.
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


def build_model(cfg, repo, vocab):
    model_config = replace(
        build_hstu_chess_config(repo.model, move_vocab_size=len(vocab)),
        enable_value_head=True,
        enable_auxiliary_value_head=False,
        enable_action_value_head=True,
    )
    model = HSTUChessModel(model_config)
    if cfg.init != "scratch":
        checkpoint = torch.load(cfg.init, map_location="cpu", weights_only=False)
        load_initial_weights(model, checkpoint, allow_missing_prefixes=("action_value_head.",))
    return model


class KlentRun:
    def __init__(self, cfg, *, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.repo = load_repo_config(cfg.base_config)
        self.vocab = MoveVocab.load(self.repo.vocab.path)
        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg, self.repo, self.vocab).to(self.device)
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
            trainer=self.trainer.state_dict(),
            iteration=self.iteration,
            positions=self.positions,
            config=asdict(self.cfg),
        )

    def load_state_dict(self, state):
        if state["config"] != asdict(self.cfg):
            raise ValueError("checkpoint was written by a different KLENT config")
        self.model.load_state_dict(state["model"])
        self.iteration, self.positions = state["iteration"], state["positions"]
        warmup = self.iteration < self.cfg.q_warmup_iterations
        lr = self.cfg.warmup_lr if warmup else self.cfg.lr
        self.trainer.load_state_dict(state["trainer"], lr=lr, weight_decay=self.cfg.weight_decay)


def run(cfg, *, output, device, save_every=5):
    output = Path(output)
    with run_lock(output):
        klent = KlentRun(cfg, device=device)
        state_path = output / "checkpoint.pt"
        if state_path.exists():
            klent.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
            print(f"resumed at iteration {klent.iteration}, positions {klent.positions}")
        while klent.positions < cfg.total_positions:
            metrics = klent.run_iteration()
            atomic_checkpoint(state_path, klent.state_dict())
            if klent.iteration % save_every == 0 or klent.positions >= cfg.total_positions:
                # Model-only snapshot in the format the eval scripts load.
                atomic_checkpoint(
                    output / f"actor-{klent.iteration:04d}.pt", dict(model=klent.model.state_dict())
                )
            line = json.dumps(metrics, allow_nan=False, sort_keys=True)
            print(line, flush=True)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")
