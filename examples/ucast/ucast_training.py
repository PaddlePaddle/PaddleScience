# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""U-Cast's hybrid optimizer, autoregressive validation and resumable Solver."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import paddle
from omegaconf import OmegaConf
from ucast_utils import _latitude_cell_weights
from ucast_utils import build_loss_weights
from ucast_utils import weighted_crps_loss
from ucast_utils import weighted_mae_loss

import ppsci
from ppsci.utils import ema
from ppsci.utils import logger
from ppsci.utils import save_load


def validate_training_config(cfg):
    if cfg.DATA.horizon != 1:
        raise ValueError(
            "U-Cast trains on single-step residual targets (DATA.horizon=1)"
        )
    if cfg.DATA.window < 1 or cfg.TRAIN.prediction_horizon < 1:
        raise ValueError("window and prediction_horizon must be positive")
    if cfg.TRAIN.loss not in ("wmae", "wcrps"):
        raise ValueError("TRAIN.loss must be wmae or wcrps")
    if cfg.TRAIN.loss == "wcrps" and cfg.MODEL.training_ensemble_size < 2:
        raise ValueError("Fair CRPS training requires at least two members")
    if cfg.TRAIN.stage == "probabilistic" and cfg.MODEL.eval_ensemble_size < 2:
        raise ValueError("Probabilistic validation requires at least two members")
    for key in ("epochs", "batch_size", "eval_batch_size", "update_freq", "eval_freq"):
        if cfg.TRAIN[key] < 1:
            raise ValueError(f"TRAIN.{key} must be positive")
    if not 0 <= cfg.TRAIN.ema.decay <= 1:
        raise ValueError("TRAIN.ema.decay must be between zero and one")
    if cfg.TRAIN.optimizer.use_muon:
        optimizer = cfg.TRAIN.optimizer
        if optimizer.get("muon_ns_dtype", "bfloat16") not in ("bfloat16", "float32"):
            raise ValueError("muon_ns_dtype must be bfloat16 or float32")
        if optimizer.muon_ns_steps < 1 or not 0 <= optimizer.muon_momentum < 1:
            raise ValueError("Muon requires positive ns_steps and momentum in [0, 1)")
        if optimizer.muon_learning_rate < 0 or optimizer.muon_weight_decay < 0:
            raise ValueError("Muon learning rate and weight decay must be nonnegative")
    if (
        cfg.TRAIN.eval_forward_batch_size is not None
        and cfg.TRAIN.eval_forward_batch_size < 1
    ):
        raise ValueError("TRAIN.eval_forward_batch_size must be null or positive")
    loss_scale = cfg.TRAIN.get("amp_init_loss_scaling", 65536.0)
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("TRAIN.amp_init_loss_scaling must be finite and positive")
    if cfg.amp_level != "O1":
        raise ValueError("U-Cast training uses AMP O1 with FP32 optimizer/EMA weights")


class UCastLRScheduler(paddle.optimizer.lr.LRScheduler):
    """Original linear-warmup/cosine schedule, counted in optimizer updates."""

    by_epoch = False

    def __init__(
        self, learning_rate, warmup_steps, max_steps, warmup_start_lr=1e-8, eta_min=1e-8
    ):
        if warmup_steps < 0 or max_steps < 1:
            raise ValueError("Invalid warmup/max_steps")
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(float(learning_rate))

    def value(self, base_lr):
        step = self.last_epoch
        if self.warmup_steps and step < self.warmup_steps:
            # The source scheduler reaches base_lr at warmup_steps-1, then
            # repeats it at warmup_steps before entering cosine annealing.
            if self.warmup_steps == 1:
                return self.warmup_start_lr
            return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (
                step / (self.warmup_steps - 1)
            )
        span = max(1, self.max_steps - self.warmup_steps)
        progress = min(max(step - self.warmup_steps, 0), span) / span
        return (
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
        )

    def get_lr(self):
        return self.value(self.base_lr)


def muon_momentum(step, total_steps, warmup_steps, maximum):
    minimum = maximum - 0.1
    cooldown = warmup_steps // 10
    if step < warmup_steps:
        return minimum + step / warmup_steps * (maximum - minimum)
    if cooldown and step > total_steps - cooldown:
        return maximum - (step - total_steps + cooldown) / cooldown * (
            maximum - minimum
        )
    return maximum


def muon_update(gradient, momentum, beta=0.95, ns_steps=5, ns_dtype="bfloat16"):
    """Source Muon; optional FP32 NS improves small-gradient stability.

    BF16 remains the source-compatible default. FP32 changes optimizer
    arithmetic and must be compared against an FP32 reference, too.
    """
    if ns_dtype not in ("bfloat16", "float32"):
        raise ValueError("ns_dtype must be bfloat16 or float32")
    momentum.set_value(paddle.lerp(momentum, gradient, 1 - beta))
    update = paddle.lerp(gradient, momentum, beta)
    if update.ndim == 4:
        update = update.reshape([update.shape[0], -1])
    # GPU BF16 matches the source implementation. FP32 also permits CPU tests.
    x = update.astype(ns_dtype if paddle.get_device().startswith("gpu") else "float32")
    transposed = x.shape[-2] > x.shape[-1]
    # Keep a tall matrix in its original storage layout and perform the
    # transposed algebra through GEMM flags. Materializing a transpose selects
    # a different BF16 GEMM path and amplifies rounding through five NS steps.
    # Paddle has no BF16 Frobenius-norm kernel. Accumulate in FP32 and
    # round the norm back to BF16, matching torch.Tensor.norm on BF16.
    x = x / (
        paddle.linalg.norm(
            x.astype("float32"), p="fro", axis=(-2, -1), keepdim=True
        ).astype(x.dtype)
        + 1e-7
    )
    for _ in range(ns_steps):
        a = paddle.matmul(x, x, transpose_x=transposed, transpose_y=not transposed)
        b = -4.7750 * a + (2.0315 * a) @ a
        correction = (
            paddle.matmul(b, x, transpose_y=True).transpose([1, 0])
            if transposed
            else b @ x
        )
        x = 3.4445 * x + correction
    # Deliberately use the original gradient's final axes, even for Conv2D.
    return x * max(1, gradient.shape[-2] / gradient.shape[-1]) ** 0.5


class UCastOptimizer(paddle.optimizer.AdamW):
    """One AMP-aware optimizer: Muon for hidden matrices, AdamW otherwise.

    A flat parameter list ensures global-norm clipping covers both optimizers
    together. State keys use model names so checkpoints survive layer renaming
    by Paddle's global unique-name counter in the same Python process.
    """

    def __init__(self, model, scheduler, cfg, grad_clip):
        named = [
            (name, p) for name, p in model.named_parameters() if not p.stop_gradient
        ]
        self.named = dict(named)
        self.internal_to_model = {p.name: name for name, p in named}
        self.muon_names = {
            p.name
            for name, p in named
            if cfg.use_muon and p.ndim >= 2 and "out_conv" not in name
        }
        self.muon_lr = float(cfg.muon_learning_rate)
        self.muon_wd = float(cfg.muon_weight_decay)
        self.muon_beta = float(cfg.muon_momentum)
        self.ns_steps = int(cfg.muon_ns_steps)
        self.ns_dtype = cfg.get("muon_ns_dtype", "bfloat16")
        if self.ns_dtype not in ("bfloat16", "float32"):
            raise ValueError("muon_ns_dtype must be bfloat16 or float32")
        self.num_updates = 0
        decay_names = {
            p.name for name, p in named if "weight" in name and "norm" not in name
        }
        super().__init__(
            learning_rate=scheduler,
            parameters=[p for _, p in named],
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            epsilon=cfg.epsilon,
            weight_decay=cfg.weight_decay,
            apply_decay_param_fun=lambda name: name in decay_names,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(grad_clip),
        )

    def _create_accumulators(self, block, parameters):
        ordinary = []
        for p in parameters:
            if p.name in self.muon_names:
                if p.name not in self._already_create_accumulator:
                    self._add_accumulator("muon_momentum", p)
                    self._already_create_accumulator.add(p.name)
            else:
                ordinary.append(p)
        if ordinary:
            super()._create_accumulators(block, ordinary)

    def _append_optimize_op(self, block, param_and_grad):
        parameter, gradient = param_and_grad
        if parameter.name not in self.muon_names:
            return super()._append_optimize_op(block, param_and_grad)
        buffer = self._get_accumulator("muon_momentum", parameter)
        schedule = self._learning_rate
        # The first source update uses the configured momentum; subsequent
        # updates use the schedule set after the previous optimizer step.
        beta = (
            self.muon_beta
            if not self.num_updates
            else muon_momentum(
                self.num_updates,
                schedule.max_steps,
                schedule.warmup_steps,
                self.muon_beta,
            )
        )
        update = muon_update(gradient, buffer, beta, self.ns_steps, self.ns_dtype)
        lr = schedule.value(self.muon_lr)
        parameter.set_value(
            parameter * (1 - lr * self.muon_wd)
            - lr * update.reshape(parameter.shape).astype(parameter.dtype)
        )

    @paddle.no_grad()
    def step(self):
        # GradScaler forwards found_inf to Optimizer subclasses. Custom Muon
        # must skip the entire update, including AdamW, on overflow.
        found_inf = self._get_auxiliary_var("found_inf")
        if found_inf is not None and bool(found_inf):
            return
        with paddle.amp.auto_cast(enable=False):
            super().step()
        self.num_updates += 1

    def state_dict(self):
        return {
            "accumulators": {
                kind: {
                    self.internal_to_model[name]: value
                    for name, value in values.items()
                }
                for kind, values in self._accumulators.items()
            },
            "LR_Scheduler": self._learning_rate.state_dict(),
            "num_updates": self.num_updates,
        }

    def set_state_dict(self, state):
        self._create_accumulators(
            paddle.static.default_main_program().global_block(),
            list(self.named.values()),
        )
        expected = {
            (kind, self.internal_to_model[name])
            for kind, values in self._accumulators.items()
            for name in values
        }
        actual = {
            (kind, name)
            for kind, values in state["accumulators"].items()
            for name in values
        }
        if expected != actual:
            raise ValueError(
                "Optimizer checkpoint parameter groups do not match this model/config"
            )
        for kind, values in state["accumulators"].items():
            for name, value in values.items():
                self._accumulators[kind][self.named[name].name].set_value(value)
        self._learning_rate.set_state_dict(state["LR_Scheduler"])
        self.num_updates = int(state["num_updates"])


class UCastEMA(ema.ExponentialMovingAverage):
    """Source LitEma decay ramp; updated after each microbatch as in Lightning."""

    @paddle.no_grad()
    def update(self):
        decay = min(self.decay, (2 + int(self.n_avg)) / (11 + int(self.n_avg)))
        for name, parameter in self.model.named_parameters():
            if not parameter.stop_gradient:
                self.params_shadow[name].set_value(
                    self.params_shadow[name]
                    - (1 - decay) * (self.params_shadow[name] - parameter)
                )
        self.n_avg.set_value(self.n_avg + 1)


@paddle.no_grad()
def rollout(
    model, inputs, normalizer, window, ensemble_size, autocast, forward_batch_size=None
):
    """Stream each lead, preserving member trajectories without ensembling twice."""
    batch_size, _, lon, lat = inputs["input"].shape
    frames = inputs["input"].reshape([batch_size, window, -1, lon, lat])
    frames = frames.unsqueeze(0).tile([ensemble_size, 1, 1, 1, 1, 1])
    static = inputs["static_condition"].unsqueeze(0).tile([ensemble_size, 1, 1, 1, 1])
    static = static.reshape([ensemble_size * batch_size, -1, lon, lat])
    for lead in range(inputs["forecast_forcing"].shape[1]):
        forcing = (
            inputs["forecast_forcing"][:, lead]
            .unsqueeze(0)
            .tile([ensemble_size, 1, 1, 1, 1])
        )
        x = {
            "input": frames.reshape([ensemble_size * batch_size, -1, lon, lat]),
            "dynamical_condition": forcing.reshape(
                [ensemble_size * batch_size, -1, lon, lat]
            ),
            "static_condition": static,
        }
        packed = paddle.concat([x[key] for key in model.input_keys], axis=1)
        chunk_size = forward_batch_size or packed.shape[0]
        residuals = []
        for start in range(0, packed.shape[0], chunk_size):
            with autocast():
                # Bypass the model's ensemble wrapper: each row already
                # belongs to a specific (member, initial condition) trajectory.
                residuals.append(
                    model.forward_tensor(packed[start : start + chunk_size])
                )
        residual = paddle.concat(residuals, axis=0)
        prediction = (
            normalizer.residual_to_normalized(residual).reshape(
                [ensemble_size, batch_size, -1, lon, lat]
            )
            + frames[:, :, -1]
        )
        if not bool(paddle.all(paddle.isfinite(prediction))):
            raise FloatingPointError(
                f"Non-finite validation prediction at lead {lead + 1}"
            )
        yield normalizer.denormalize(prediction)
        frames = paddle.concat([frames[:, :, 1:], prediction.unsqueeze(2)], axis=2)


class ForecastMetrics:
    """Stream area-weighted per-variable metrics, then average leads/variables."""

    def __init__(self, latitude, variable_names, horizon, ensemble_size):
        self.weights = paddle.to_tensor(_latitude_cell_weights(latitude))[
            None, None, None, :
        ]
        self.variable_names = variable_names
        self.ensemble_size = ensemble_size
        self.count = np.zeros(horizon, dtype=np.int64)
        self.sums = {
            key: np.zeros((horizon, len(variable_names)), dtype=np.float64)
            for key in ("mse", "l1", "crps", "variance")
        }

    def update(self, lead, predictions, targets):
        mean = predictions.mean(axis=0)
        diff = mean - targets
        values = {"mse": diff.square(), "l1": diff.abs()}
        if self.ensemble_size > 1:
            skill = (predictions - targets.unsqueeze(0)).abs().mean(axis=0)
            spread = paddle.zeros_like(skill)
            # Avoid materializing an M x M x B x C x H x W temporary.
            for member in range(self.ensemble_size):
                spread += (
                    (predictions[member : member + 1] - predictions).abs().sum(axis=0)
                )
            values["crps"] = skill - spread / (
                2 * self.ensemble_size * (self.ensemble_size - 1)
            )
            values["variance"] = predictions.var(axis=0, unbiased=True)
        for key, value in values.items():
            self.sums[key][lead] += (
                (value * self.weights).mean(axis=(-2, -1)).sum(axis=0).numpy()
            )
        self.count[lead] += targets.shape[0]

    def compute(self):
        result = {}
        means = {key: value / self.count[:, None] for key, value in self.sums.items()}
        values = {"rmse": np.sqrt(means["mse"]), "l1": means["l1"]}
        if self.ensemble_size > 1:
            values["crps"] = means["crps"]
            values["ssr"] = np.sqrt(
                means["variance"] * (self.ensemble_size + 1) / self.ensemble_size
            ) / np.maximum(values["rmse"], 1e-12)
        for metric, value in values.items():
            result[f"val/avg/{metric}"] = float(value.mean())
            for index, name in enumerate(self.variable_names):
                result[f"val/avg/{metric}/{name}"] = float(value[:, index].mean())
                for lead in range(len(self.count)):
                    result[f"val/t{lead + 1}/{metric}/{name}"] = float(
                        value[lead, index]
                    )
        return result


def _loader_config(dataset, batch_size, num_workers, training):
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_shared_memory": False,
        "sampler": {"name": "BatchSampler", "shuffle": training, "drop_last": training},
    }


def _checkpoint_prefix(path):
    path = str(path)
    for suffix in (".pdparams", ".pdopt", ".pdstates", ".pdtrain"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


class UCastSolver(ppsci.solver.Solver):
    """Specialize Solver without changing shared PaddleScience training behavior."""

    def __init__(self, cfg, model, train_dataset, val_dataset, normalizer, latitude):
        self.ucast_cfg = cfg
        self.normalizer = normalizer
        self.val_dataset = val_dataset
        self.latitude = latitude
        weights = paddle.to_tensor(
            build_loss_weights(latitude, train_dataset.variable_names)
        )
        self.loss_function = (
            weighted_mae_loss(weights)
            if cfg.TRAIN.loss == "wmae"
            else weighted_crps_loss(weights)
        )
        constraint = ppsci.constraint.SupervisedConstraint(
            _loader_config(
                train_dataset, cfg.TRAIN.batch_size, cfg.TRAIN.num_workers, True
            ),
            ppsci.loss.FunctionalLoss(self.loss_function),
            name="ucast_train",
        )
        self.train_loader = constraint.data_loader
        self.val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_size=cfg.TRAIN.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.TRAIN.num_workers,
            use_shared_memory=False,
        )
        iterations = len(self.train_loader)
        if not iterations:
            raise ValueError(
                "Training dataset is smaller than TRAIN.batch_size; reduce batch size"
            )
        scheduler = UCastLRScheduler(
            cfg.TRAIN.optimizer.learning_rate,
            cfg.TRAIN.scheduler.warmup_steps,
            cfg.TRAIN.epochs * math.ceil(iterations / cfg.TRAIN.update_freq),
            cfg.TRAIN.scheduler.warmup_start_lr,
            cfg.TRAIN.scheduler.eta_min,
        )
        optimizer = UCastOptimizer(
            model, scheduler, cfg.TRAIN.optimizer, cfg.TRAIN.grad_clip
        )
        # The specialized loop owns EMA/resume and validation, so the base
        # constructor must not load weights before these objects exist.
        super().__init__(
            model=model,
            constraint={constraint.name: constraint},
            output_dir=cfg.output_dir,
            optimizer=optimizer,
            epochs=cfg.TRAIN.epochs,
            iters_per_epoch=iterations,
            update_freq=cfg.TRAIN.update_freq,
            log_freq=cfg.log_freq,
            seed=cfg.seed,
            device=cfg.device,
            use_amp=bool(cfg.use_amp) and cfg.device.startswith("gpu"),
            amp_level=cfg.amp_level,
        )
        if self.use_amp:
            self.scaler = paddle.amp.GradScaler(
                init_loss_scaling=cfg.TRAIN.get("amp_init_loss_scaling", 65536.0)
            )
        if self.world_size != 1:
            raise ValueError(
                "This U-Cast training entry currently supports one GPU per process"
            )
        self.epoch = 0
        self.micro_steps = 0
        self.best_value = float("inf")
        if cfg.TRAIN.pretrained_model_path:
            save_load.load_pretrain(self.model, cfg.TRAIN.pretrained_model_path)
        self.ema_model = (
            UCastEMA(self.model, cfg.TRAIN.ema.decay) if cfg.TRAIN.ema.use_ema else None
        )
        if cfg.TRAIN.checkpoint_path:
            self.restore_training(_checkpoint_prefix(cfg.TRAIN.checkpoint_path))
        logger.info(
            f"Prepared {len(train_dataset)} train / {len(val_dataset)} validation samples; "
            f"{iterations} microbatches/epoch, accumulation={cfg.TRAIN.update_freq}, "
            f"Muon tensors={len(optimizer.muon_names)}"
        )

    def autocast(self):
        return self.autocast_context_manager(self.use_amp, self.amp_level)

    def _signature(self):
        cfg = self.ucast_cfg
        optimizer_signature = OmegaConf.to_container(cfg.TRAIN.optimizer)
        if optimizer_signature.get("muon_ns_dtype", "bfloat16") == "bfloat16":
            optimizer_signature.pop("muon_ns_dtype", None)
        data_signature = OmegaConf.to_container(cfg.DATA)
        # Storage location does not change the ERA5 samples or resume protocol.
        data_signature.pop("cache_dir", None)
        return {
            "seed": cfg.seed,
            "use_amp": cfg.use_amp,
            "amp_level": cfg.amp_level,
            "model": OmegaConf.to_container(cfg.MODEL),
            "optimizer": optimizer_signature,
            "scheduler": OmegaConf.to_container(cfg.TRAIN.scheduler),
            "data": data_signature,
            "train_time_slices": OmegaConf.to_container(cfg.TRAIN.train_time_slices),
            "max_train_samples": cfg.TRAIN.max_train_samples,
            "stage": cfg.TRAIN.stage,
            "loss": cfg.TRAIN.loss,
            "batch_size": cfg.TRAIN.batch_size,
            "update_freq": cfg.TRAIN.update_freq,
            "iters_per_epoch": self.iters_per_epoch,
            "epochs": cfg.TRAIN.epochs,
            "ema": OmegaConf.to_container(cfg.TRAIN.ema),
        }

    def save_training(self, prefix, metric):
        save_load.save_checkpoint(
            self.model,
            self.optimizer,
            {"epoch": self.epoch, "metric": metric},
            self.scaler if self.use_amp else None,
            self.output_dir,
            prefix,
            ema_model=self.ema_model,
        )
        state = {
            "signature": self._signature(),
            "epoch": self.epoch,
            "micro_steps": self.micro_steps,
            "best_value": self.best_value,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "paddle_rng": paddle.get_rng_state(),
            "cuda_rng": paddle.get_cuda_rng_state()
            if self.use_amp or paddle.get_device().startswith("gpu")
            else None,
        }
        paddle.save(
            state, str(Path(self.output_dir) / "checkpoints" / f"{prefix}.pdtrain")
        )

    def restore_training(self, prefix):
        state = paddle.load(prefix + ".pdtrain", return_numpy=True)
        if state["signature"] != self._signature():
            raise ValueError(
                "Resume configuration differs from checkpoint; use pretrained_model_path for warm-start"
            )
        save_load.load_checkpoint(
            prefix,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp else None,
            ema_model=self.ema_model,
        )
        self.epoch = int(state["epoch"])
        self.micro_steps = int(state["micro_steps"])
        self.best_value = float(state["best_value"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        paddle.set_rng_state(state["paddle_rng"])
        if state["cuda_rng"] is not None:
            paddle.set_cuda_rng_state(state["cuda_rng"])
        logger.info(
            f"Resumed epoch {self.epoch}, optimizer step {self.optimizer.num_updates}, "
            f"EMA updates {int(self.ema_model.n_avg) if self.ema_model else 0}"
        )

    @paddle.no_grad()
    def eval(self, epoch_id=None):
        cfg = self.ucast_cfg
        modes = [
            (layer, layer.training) for layer in [self.model, *self.model.sublayers()]
        ]
        self.model.enable_mc_dropout()
        metrics = ForecastMetrics(
            self.latitude,
            self.val_dataset.variable_names,
            cfg.TRAIN.prediction_horizon,
            cfg.MODEL.eval_ensemble_size,
        )
        if self.ema_model:
            self.ema_model.apply_shadow()
        try:
            for inputs, labels, _ in self.val_loader:
                for lead, predictions in enumerate(
                    rollout(
                        self.model,
                        inputs,
                        self.normalizer,
                        cfg.DATA.window,
                        cfg.MODEL.eval_ensemble_size,
                        self.autocast,
                        cfg.TRAIN.eval_forward_batch_size,
                    )
                ):
                    targets = self.normalizer.denormalize(labels["output"][:, lead])
                    metrics.update(lead, predictions, targets)
        finally:
            if self.ema_model:
                self.ema_model.restore()
            for layer, training in modes:
                layer.training = training
        values = metrics.compute()
        key = "val/avg/crps" if cfg.TRAIN.stage == "probabilistic" else "val/avg/rmse"
        logger.info(
            f"[Eval][Epoch {epoch_id}] {key}={values[key]:.8g}, "
            f"horizon={cfg.TRAIN.prediction_horizon}, members={cfg.MODEL.eval_ensemble_size}"
        )
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"validation_epoch_{epoch_id}.json").write_text(
            json.dumps(values, indent=2)
        )
        return values[key], values

    def train(self):
        cfg = self.ucast_cfg
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        last_epoch = min(
            cfg.TRAIN.epochs, cfg.TRAIN.stop_after_epoch or cfg.TRAIN.epochs
        )
        for epoch in range(self.epoch + 1, last_epoch + 1):
            self.model.train()
            self.optimizer.clear_grad()
            running_loss = 0.0
            samples = 0
            # Match Lightning's normalization by configured accumulation factor,
            # including a partial final accumulation group.
            for iteration, (inputs, labels, _) in enumerate(self.train_loader, 1):
                with self.autocast():
                    output = self.model(inputs)
                    loss = next(iter(self.loss_function(output, labels).values()))
                if not bool(paddle.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch}, iteration {iteration}"
                    )
                scaled_loss = loss / cfg.TRAIN.update_freq
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if (
                    iteration % cfg.TRAIN.update_freq == 0
                    or iteration == self.iters_per_epoch
                ):
                    previous_updates = self.optimizer.num_updates
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.clear_grad()
                    if self.optimizer.num_updates > previous_updates:
                        self.lr_scheduler.step()
                if self.ema_model:
                    self.ema_model.update()
                self.micro_steps += 1
                count = labels["output"].shape[0]
                running_loss += float(loss) * count
                samples += count
                if (
                    iteration == 1
                    or iteration % cfg.log_freq == 0
                    or iteration == self.iters_per_epoch
                ):
                    logger.info(
                        f"[Train][Epoch {epoch}][{iteration}/{self.iters_per_epoch}] "
                        f"loss={float(loss):.8g}, optimizer_step={self.optimizer.num_updates}, "
                        f"lr={self.optimizer.get_lr():.8g}"
                    )
            self.epoch = epoch
            metric = float("inf")
            if (
                cfg.TRAIN.eval_during_train
                and epoch >= cfg.TRAIN.start_eval_epoch
                and epoch % cfg.TRAIN.eval_freq == 0
            ):
                metric, _ = self.eval(epoch)
                if metric < self.best_value:
                    self.best_value = metric
                    self.save_training("best_model", metric)
                    if self.ema_model:
                        save_load.save_checkpoint(
                            self.ema_model,
                            None,
                            output_dir=self.output_dir,
                            prefix="best_model_ema",
                        )
            if cfg.TRAIN.save_freq > 0 and epoch % cfg.TRAIN.save_freq == 0:
                self.save_training(f"epoch_{epoch}", metric)
            self.save_training("latest", metric)
            summary = {
                "epoch": epoch,
                "loss": running_loss / samples,
                "optimizer_steps": self.optimizer.num_updates,
                "micro_steps": self.micro_steps,
                "ema_updates": int(self.ema_model.n_avg) if self.ema_model else 0,
                "learning_rate": self.optimizer.get_lr(),
                "validation_metric": metric,
            }
            with (Path(self.output_dir) / "training_history.jsonl").open("a") as stream:
                stream.write(json.dumps(summary) + "\n")
