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
from .config import LearningConfig
from .losses import self_play_loss


class ContinuedOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """Finish an imported schedule, then hold each group's terminal minimum LR."""

    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [group["min_lr"] for group in self.optimizer.param_groups]
        return super().get_lr()


def _model_loss(model, batch, mask, value_weight, auxiliary_value_weight=1.0):
    output = model(batch, block_mask=mask, return_loss=False)
    return self_play_loss(output, batch, value_weight=value_weight,
                          auxiliary_value_weight=auxiliary_value_weight)


def _training_loss(model, batch, value_weight, auxiliary_value_weight=1.0, *, model_loss=_model_loss):
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
    return model_loss(model, batch, mask, value_weight, auxiliary_value_weight)


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
        auxiliary = getattr(model, "auxiliary_value_head", None)
        if config.auxiliary_value_weight > 0 and auxiliary is None:
            raise ValueError("auxiliary value loss requires an enabled auxiliary value head")
        if (auxiliary is not None and config.auxiliary_value_weight > 0
                and auxiliary.out_features != 3 * len(config.auxiliary_value_lambdas)):
            raise ValueError("model auxiliary head count must match auxiliary_value_lambda")
        if auxiliary is not None and config.auxiliary_value_weight == 0:
            auxiliary.requires_grad_(False)
        # Zero value weight means policy-only training. Freeze before grouping
        # so the head receives neither optimizer moments nor weight decay.
        if config.value_weight == 0 and config.auxiliary_value_weight == 0 and getattr(model, "value_head", None) is not None:
            model.value_head.requires_grad_(False)
        head = getattr(model, "value_head", None)
        self.value_parameters = list(head.parameters()) if head is not None else []
        if auxiliary is not None:
            self.value_parameters.extend(auxiliary.parameters())
        value_ids = {id(p) for p in self.value_parameters}
        self.policy_parameters = [p for p in model.parameters() if id(p) not in value_ids]
        self.optimizer = StableAdamW(
            build_decay_param_groups(model, weight_decay=config.weight_decay),
            lr=config.lr,
            triton=False,
            kahan_sum=True,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )
        # Parameters (deduplicated, so tied weights average once) and their
        # moving averages; None when ema_decay is 0.
        self._ema_parameters = list(model.parameters())
        self.ema = self._fresh_ema() if config.ema_decay > 0 else None
        self.rng = random.Random(run_seed)
        self.steps = self.exposures = self.phase_exposures = 0
        self.queue = []
        self.reuse_counts = {}
        self.sample_ids = []

    def _fresh_ema(self):
        return [p.detach().clone() for p in self._ema_parameters]

    def ema_state_dict(self):
        """The model's state_dict with every parameter replaced by its moving average."""
        average = {id(p): e for p, e in zip(self._ema_parameters, self.ema)}
        return {
            key: average.get(id(value), value).detach().cpu().clone()
            for key, value in self.model.state_dict(keep_vars=True).items()
        }

    def _load_ema(self, saved):
        names = {id(v): k for k, v in self.model.state_dict(keep_vars=True).items()}
        self.ema = [saved[names[id(p)]].to(p.device, p.dtype).clone() for p in self._ema_parameters]

    def _clip_gradients(self):
        policy_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_parameters, self.config.grad_clip, error_if_nonfinite=True
        )
        value_norm = torch.nn.utils.clip_grad_norm_(
            self.value_parameters, self.config.grad_clip, error_if_nonfinite=True
        )
        return policy_norm, value_norm

    def _restore_optimization(self, state):
        schedule = state["scheduler"]
        kind = state.get("scheduler_type", "LambdaLR")
        if kind in ("OneCycleLR", "ContinuedOneCycleLR"):
            # Construction changes optimizer rates; restore its saved state after
            # constructing the correct scheduler, then restore the schedule clock.
            if schedule.get("cycle_momentum", False):
                raise ValueError("momentum-cycling schedules are not supported")
            self.scheduler = ContinuedOneCycleLR(
                self.optimizer,
                max_lr=[g["max_lr"] for g in state["optimizer"]["param_groups"]],
                total_steps=schedule["total_steps"],
                cycle_momentum=False,
            )
        elif kind != "LambdaLR":
            raise ValueError(f"unsupported scheduler: {kind}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(schedule)
        if isinstance(self.scheduler, ContinuedOneCycleLR) and self.scheduler.last_epoch >= self.scheduler.total_steps:
            for group in self.optimizer.param_groups:
                group["lr"] = group["min_lr"]
            self.scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def initialize_optimization(self, path):
        """Carry supervised optimizer/schedule into a fresh self-play dataset."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("stage2_schema") or "_schedule_phases" not in state["scheduler"]:
            raise ValueError("expected a supervised OneCycleLR checkpoint")
        source = {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()}
        current = self.model.state_dict()
        if source.keys() != current.keys() or any(
            not torch.equal(v.detach().cpu(), source[k]) for k, v in current.items()
        ):
            raise ValueError("optimizer source must match initialized model weights")
        groups = state["optimizer"]["param_groups"]
        if len(groups) != len(self.optimizer.param_groups):
            raise ValueError("optimizer parameter groups differ")
        for saved, actual in zip(groups, self.optimizer.param_groups):
            if len(saved["params"]) != len(actual["params"]):
                raise ValueError("optimizer parameter counts differ")
            if saved["lr"] != self.config.lr or saved["weight_decay"] != actual["weight_decay"]:
                raise ValueError("configuration must match saved LR and weight decay")
            for key, param in zip(saved["params"], actual["params"]):
                moments = state["optimizer"]["state"].get(key, {})
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in moments and moments[name].shape != param.shape:
                        raise ValueError("optimizer parameter shapes differ")
        self._restore_optimization(dict(state, scheduler_type="OneCycleLR"))

    def begin_phase(self, store, *, iteration=None, exposure_budget=None):
        """Queue every game collected in `iteration`, then older replay games up to
        `exposure_budget` positions in total, shuffled together. Without an
        iteration the phase samples the whole window uniformly."""
        self.phase_exposures = 0
        self.sample_ids = store.game_ids("train")
        self.reuse_counts = {
            gid: self.reuse_counts.get(gid, 0) for gid in self.sample_ids
        }
        self.queue = []
        if not self.sample_ids:
            raise ValueError("no completed training trajectories")
        if iteration is None:
            return
        if exposure_budget is None:
            raise ValueError("fresh-first sampling needs the phase exposure budget")
        fresh = [g for g in self.sample_ids if store.index[g][1].get("iteration") == iteration]
        if not fresh:
            raise ValueError(f"no training trajectories from iteration {iteration}")
        old = [g for g in self.sample_ids if store.index[g][1].get("iteration") != iteration]
        self.rng.shuffle(old)
        remaining = exposure_budget - sum(store.index[g][1]["positions"] for g in fresh)
        chosen = []
        while remaining > 0 and old:
            gid = old.pop()
            chosen.append(gid)
            remaining -= store.index[gid][1]["positions"]
        self.queue = fresh + chosen
        self.rng.shuffle(self.queue)

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
                learning=self.config,
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
                self.optimizer.zero_grad(set_to_none=True)
                # An optimizer step always completes all microbatches, so
                # checkpoints never land mid-accumulation. Loss metrics are
                # means over the step's equally weighted microbatches.
                accumulation = self.config.gradient_accumulation
                loss_sums, positions, tokens = {}, 0, 0
                for _ in range(accumulation):
                    batch, gids = self._next_batch(store)
                    losses = self._loss_fn(self.model, batch, self.config.value_weight,
                                          self.config.auxiliary_value_weight)
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError("nonfinite stage-2 loss")
                    (losses["loss"] / accumulation).backward()
                    for key, value in losses.items():
                        loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach())
                    positions += len(batch["supervised_indices"])
                    tokens += batch["total_tokens"]
                    for gid in gids:
                        self.reuse_counts[gid] = (
                            self.reuse_counts.get(gid, 0) + store.index[gid][1]["positions"]
                        )
                norm, value_norm = self._clip_gradients()
                learning_rate = self.optimizer.param_groups[0]["lr"]
                self.optimizer.step()
                self.scheduler.step()
                if self.ema is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(
                            self.ema, [p.detach() for p in self._ema_parameters],
                            1 - self.config.ema_decay,
                        )
                self.steps += 1
                self.exposures += positions
                self.phase_exposures += positions
                metrics = dict(
                    {key: value / accumulation for key, value in loss_sums.items()},
                    gradient_norm=float(norm),
                    **({"policy_gradient_norm": float(norm),
                        "value_gradient_norm": float(value_norm)} if value_norm is not None else {}),
                    learning_rate=learning_rate,
                    steps=self.steps,
                    exposures=self.exposures,
                    phase_exposures=self.phase_exposures,
                    supervised_positions=positions,
                    context_tokens=tokens,
                    microbatches=accumulation,
                    replay_unique_positions=sum(
                        store.index[g][1]["positions"] for g in self.sample_ids
                    ),
                )
                elapsed = time.perf_counter() - step_start
                metrics.update(
                    step_seconds=elapsed,
                    supervised_positions_per_second=positions / elapsed,
                    context_tokens_per_second=tokens / elapsed,
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
                scheduler_type=type(self.scheduler).__name__,
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
                gradient_clipping="separate_value_head_v1",
                **({"ema_model": self.ema_state_dict()} if self.ema is not None else {}),
            ),
        )

    def resume(self, path, *, store, config_id, previous_learning=None):
        """Restore a checkpoint. previous_learning, when given, is the learning
        config the checkpoint was written under and may differ from this trainer's;
        an EMA switched on by the change starts from the checkpoint's weights."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("stage2_schema") != 1:
            raise ValueError(
                "resume requires a stage-2 checkpoint; use initialize for stage-1"
            )
        if state.get("gradient_clipping") != "separate_value_head_v1":
            raise ValueError("resume configuration changed: gradient clipping mode")
        if "detach_value_features" in state["learning_config"]:
            raise ValueError("resume configuration changed: detached-value checkpoint")
        saved_learning = {"auxiliary_value_weight": 0.0, "ema_decay": 0.0, **state["learning_config"]}
        expected = self.config if previous_learning is None else previous_learning
        if state["config_id"] != config_id or asdict(LearningConfig(**saved_learning)) != asdict(
            expected
        ):
            raise ValueError("resume configuration changed")
        for name in state["replay_shards"]:
            if not (store.directory / name).exists():
                raise FileNotFoundError(f"checkpoint replay shard missing: {name}")
        self.model.load_state_dict(state["model"], strict=True)
        if self.ema is not None:
            if "ema_model" in state:
                self._load_ema(state["ema_model"])
            elif previous_learning is None or previous_learning.ema_decay != 0:
                raise ValueError("checkpoint has no EMA weights")
            else:
                self.ema = self._fresh_ema()
        self._restore_optimization(state)
        if previous_learning is not None and previous_learning.lr != self.config.lr:
            # The restored optimizer and schedule carry the old rate.
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
                raise ValueError("an lr change on resume needs the constant schedule")
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] = self.config.lr
            self.scheduler.base_lrs = [self.config.lr] * len(self.optimizer.param_groups)
            self.scheduler._last_lr = [self.config.lr] * len(self.optimizer.param_groups)
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
