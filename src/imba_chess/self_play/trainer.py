"""Stage-2 trainer with whole-trajectory sampling and explicit resumable state."""

from dataclasses import asdict
import os
from pathlib import Path
import random
import time

import torch
from optimi import StableAdamW

from imba_chess.model import create_batch_dense_mask, create_batch_block_mask
from imba_chess.optim import build_decay_param_groups
from imba_chess.data.self_play_store import sync_directory
from .dataset import reconstruct, collate_self_play
from .losses import self_play_loss


def _model_loss(model, batch, mask, value_weight):
    output = model(batch, block_mask=mask, return_loss=False)
    return self_play_loss(output, batch, value_weight=value_weight)


def _training_loss(model, batch, value_weight, *, model_loss=_model_loss):
    device = model.piece_square_embedding.weight.device
    mask_factory = (
        create_batch_block_mask if device.type == "cuda" else create_batch_dense_mask
    )
    with torch.no_grad():
        mask = mask_factory(
            batch["seq_offsets"].to(device),
            total_tokens=batch["total_tokens"],
            device=device,
        )
    return model_loss(model, batch, mask, value_weight)


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


class Stage2Trainer:
    def __init__(
        self, *, model, config, move_vocab, encoder, device, max_positions, run_seed=42
    ):
        self.model, self.config, self.move_vocab, self.encoder = (
            model,
            config,
            move_vocab,
            encoder,
        )
        self.device, self.max_positions = device, max_positions
        # Keep the model shared with collection and plain checkpoint keys. The
        # benchmark wraps this tensor-only callable without replacing the model.
        self._loss_fn = _training_loss
        self.optimizer = StableAdamW(
            build_decay_param_groups(model, weight_decay=config.weight_decay),
            lr=config.lr,
            triton=False,
            kahan_sum=True,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )
        self.rng = random.Random(run_seed)
        self.steps = self.exposures = self.phase_exposures = 0
        self.queue = []
        self.reuse_counts = {}
        self.sample_ids = []

    def begin_phase(self, store):
        self.phase_exposures = 0
        self.sample_ids = store.game_ids("train")
        self.reuse_counts = {
            gid: self.reuse_counts.get(gid, 0) for gid in self.sample_ids
        }
        self.queue = []
        if not self.sample_ids:
            raise ValueError("no completed training trajectories")

    def _next_batch(self, store):
        if not self.queue:
            self.queue = list(self.sample_ids)
            self.rng.shuffle(self.queue)
        samples, gids, tokens = [], [], 0
        while self.queue:
            gid = self.queue[-1]
            sample = reconstruct(
                store.read_game(gid),
                move_vocab=self.move_vocab,
                encoder=self.encoder,
                max_positions=self.max_positions,
            )
            size = len(sample["seq_token_id"])
            if size > self.config.microbatch_tokens:
                raise ValueError("trajectory exceeds training microbatch token limit")
            if samples and tokens + size > self.config.microbatch_tokens:
                break
            self.queue.pop()
            gids.append(gid)
            samples.append(sample)
            tokens += size
        return collate_self_play(samples), gids

    def train(
        self,
        store,
        *,
        exposure_budget,
        should_stop=lambda: False,
        on_step=lambda metrics: None,
    ):
        self.model.train()
        try:
            while self.phase_exposures < exposure_budget and not should_stop():
                step_start = time.perf_counter()
                batch, gids = self._next_batch(store)
                self.optimizer.zero_grad(set_to_none=True)
                losses = self._loss_fn(self.model, batch, self.config.value_weight)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("nonfinite stage-2 loss")
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.scheduler.step()
                positions = len(batch["supervised_indices"])
                self.steps += 1
                self.exposures += positions
                self.phase_exposures += positions
                for gid in gids:
                    self.reuse_counts[gid] = (
                        self.reuse_counts.get(gid, 0) + store.index[gid][1]["positions"]
                    )
                metrics = dict(
                    {key: float(value.detach()) for key, value in losses.items()},
                    gradient_norm=float(norm),
                    steps=self.steps,
                    exposures=self.exposures,
                    phase_exposures=self.phase_exposures,
                    supervised_positions=positions,
                    context_tokens=batch["total_tokens"],
                    replay_unique_positions=sum(
                        store.index[g][1]["positions"] for g in self.sample_ids
                    ),
                )
                elapsed = time.perf_counter() - step_start
                metrics.update(
                    step_seconds=elapsed,
                    supervised_positions_per_second=positions / elapsed,
                    context_tokens_per_second=batch["total_tokens"] / elapsed,
                )
                on_step(metrics)
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()

    def checkpoint(self, path, *, progress, store, config_id):
        atomic_checkpoint(
            path,
            dict(
                stage2_schema=1,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
                python_rng=random.getstate(),
                sampler_rng=self.rng.getstate(),
                steps=self.steps,
                exposures=self.exposures,
                phase_exposures=self.phase_exposures,
                queue=self.queue,
                sample_ids=self.sample_ids,
                reuse_counts=self.reuse_counts,
                replay_active=list(store.manifest["active"]),
                replay_shards=store.active_shards(),
                progress=progress,
                config_id=config_id,
                learning_config=asdict(self.config),
            ),
        )

    def resume(self, path, *, store, config_id):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("stage2_schema") != 1:
            raise ValueError(
                "resume requires a stage-2 checkpoint; use initialize for stage-1"
            )
        if state["config_id"] != config_id or state["learning_config"] != asdict(
            self.config
        ):
            raise ValueError("resume configuration changed")
        for name in state["replay_shards"]:
            if not (store.directory / name).exists():
                raise FileNotFoundError(f"checkpoint replay shard missing: {name}")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        self.rng.setstate(state["sampler_rng"])
        for key in (
            "steps",
            "exposures",
            "phase_exposures",
            "queue",
            "sample_ids",
            "reuse_counts",
        ):
            setattr(self, key, state[key])
        return state["progress"]
