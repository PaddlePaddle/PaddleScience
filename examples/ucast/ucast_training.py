# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""PaddleScience Solver training, checkpoint and forecast evaluation helpers."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import paddle
from omegaconf import OmegaConf
from ucast_config import normalize_config
from ucast_data import (
    ERA5WindowDataset,
    latitude_area_weights,
    select_initial_conditions,
    training_weights,
)
from ucast_optimization import clip_grad_norm  # noqa: F401
from ucast_optimization import AdaptiveEMA, HybridMuonAdamW
from ucast_utils import (
    VARIABLE_NAMES,
    autoregressive_inference,
    ensure_latitude_is_ascending,
    load_normalizer,
    load_static_conditions,
    open_era5_zarr,
)

import ppsci
from ppsci.utils import logger, save_load


def load_weights(model, path, use_ema=False):
    path = Path(path)
    if not path.is_file() and path.with_suffix(".pdparams").is_file():
        path = path.with_suffix(".pdparams")
    state = paddle.load(str(path), return_numpy=True)
    if state.get("format_version") == 2 and "solver_checkpoint" in state:
        prefix = path.parent / state["solver_checkpoint"]
        if use_ema:
            state = paddle.load(str(prefix) + ".pdstates", return_numpy=True)["ucast"][
                "ema"
            ]["shadow"]
        else:
            state = paddle.load(str(prefix) + ".pdparams", return_numpy=True)
    if "model" in state and "format_version" in state:
        state = state["ema"]["shadow"] if use_ema else state["model"]
    if state and all(name.startswith("network.") for name in state):
        state = {name[len("network.") :]: value for name, value in state.items()}
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"Checkpoint keys differ: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}"
        )
    for name, value in state.items():
        if list(value.shape) != list(expected[name].shape):
            raise ValueError(
                f"Wrong checkpoint shape for {name}: {value.shape} != {expected[name].shape}"
            )
    model.set_state_dict(state)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": paddle.get_rng_state("cpu"),
        "gpu": paddle.get_rng_state("gpu")
        if paddle.get_device().startswith("gpu")
        else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    paddle.set_rng_state(state["cpu"], "cpu")
    if state["gpu"] is not None:
        paddle.set_rng_state(state["gpu"], "gpu")


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    paddle.save(state, str(temporary))
    os.replace(temporary, path)


def forecast_moments(result, latitude):
    prediction = result["predictions"].numpy().astype(np.float64)
    target = result["targets"].numpy().astype(np.float64)
    weights = latitude_area_weights(latitude)[None, None, None, :]
    mean = prediction.mean(axis=0)
    error = mean - target
    moments = {
        "mse": (error**2 * weights).mean(axis=(-2, -1)),
        "l1": (np.abs(error) * weights).mean(axis=(-2, -1)),
    }
    members = len(prediction)
    if members > 1:
        crps = np.abs(prediction - target[None]).mean(axis=0)
        for i in range(members):
            for j in range(i):
                crps -= np.abs(prediction[i] - prediction[j]) / (
                    members * (members - 1)
                )
        moments["crps"] = (crps * weights).mean(axis=(-2, -1))
        moments["variance"] = (prediction.var(axis=0, ddof=1) * weights).mean(
            axis=(-2, -1)
        )
    return moments


@paddle.no_grad()
def evaluate(model, dataset, normalizer, members, use_amp):
    model.enable_mc_dropout()
    totals = {}
    static = paddle.to_tensor(dataset.static_condition)
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = {
            key: paddle.to_tensor(sample[key])
            for key in ("dynamics_normalized", "dynamics_raw", "forcing")
        }
        result = autoregressive_inference(
            model,
            batch,
            static,
            normalizer,
            dataset.window,
            dataset.horizon,
            members,
            use_amp,
        )
        moments = forecast_moments(result, dataset.latitude)
        for key, value in moments.items():
            totals[key] = totals.get(key, 0) + value
    metrics = {key: value / len(dataset) for key, value in totals.items()}
    metrics["rmse"] = np.sqrt(metrics.pop("mse"))
    if "variance" in metrics:
        # Ratio of globally averaged ensemble spread and ensemble-mean RMSE.
        metrics["ssr"] = np.sqrt(
            metrics.pop("variance") * (members + 1) / members
        ) / np.maximum(metrics["rmse"], 1e-12)
    report = {f"val/avg/{key}": float(value.mean()) for key, value in metrics.items()}
    report["per_lead_variable"] = {
        key: {
            name: value[:, i].tolist() for i, name in enumerate(dataset.variable_names)
        }
        for key, value in metrics.items()
    }
    if not all(
        np.isfinite(v) for key, v in report.items() if key != "per_lead_variable"
    ):
        raise FloatingPointError("Non-finite validation metric")
    return report


class UCastTrainer:
    def __init__(self, cfg, model, train_dataset, val_dataset, normalizer):
        cfg = normalize_config(cfg)
        self.cfg, self.model = cfg, model
        self.train_dataset, self.val_dataset, self.normalizer = (
            train_dataset,
            val_dataset,
            normalizer,
        )
        self.rank = paddle.distributed.get_rank()
        self.world_size = paddle.distributed.get_world_size()
        self.train_cfg = cfg.TRAIN
        self.output_dir = Path(cfg.TRAIN.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(cfg.TRAIN.batch_size)
        self.accumulate = int(cfg.TRAIN.accumulate_grad_batches)
        if self.batch_size < 1 or self.accumulate < 1 or cfg.TRAIN.epochs < 1:
            raise ValueError(
                "batch_size, accumulate_grad_batches and epochs must be positive"
            )
        if (
            not math.isfinite(cfg.TRAIN.amp_init_loss_scaling)
            or cfg.TRAIN.amp_init_loss_scaling <= 0
        ):
            raise ValueError("TRAIN.amp_init_loss_scaling must be finite and positive")
        if cfg.TRAIN.loss == "wcrps" and cfg.TRAIN.ensemble_size < 2:
            raise ValueError(
                "Probabilistic training requires at least two dropout members"
            )
        self.use_amp = bool(cfg.TRAIN.use_amp) and paddle.get_device().startswith("gpu")
        self.scaler = paddle.amp.GradScaler(
            enable=self.use_amp, init_loss_scaling=cfg.TRAIN.amp_init_loss_scaling
        )
        self.optimizer = HybridMuonAdamW(
            model,
            gradient_clip_norm=cfg.TRAIN.gradient_clip_norm,
            **cfg.TRAIN.optimizer,
        )
        self.ema = AdaptiveEMA(model, cfg.TRAIN.ema_decay, cfg.TRAIN.ema_enabled)
        self.weights = paddle.to_tensor(
            training_weights(train_dataset.latitude, train_dataset.variable_names)
        )
        per_rank = len(train_dataset) // self.world_size
        self.num_batches = math.ceil(per_rank / self.batch_size)
        if cfg.TRAIN.max_batches_per_epoch is not None:
            self.num_batches = min(
                self.num_batches, int(cfg.TRAIN.max_batches_per_epoch)
            )
        if self.num_batches < 1:
            raise ValueError("Training data has fewer samples than distributed ranks")
        self.total_steps = int(cfg.TRAIN.epochs) * math.ceil(
            self.num_batches / self.accumulate
        )
        self.epoch, self.next_batch, self.micro_step = 0, 0, 0
        self.best_metric, self.resume_rng = float("inf"), None
        self.signature = {
            "model": OmegaConf.to_container(cfg.MODEL, resolve=True),
            "optimizer": OmegaConf.to_container(cfg.TRAIN.optimizer, resolve=True),
            "loss": cfg.TRAIN.loss,
            "ensemble_size": cfg.TRAIN.ensemble_size,
            "batch_size": self.batch_size,
            "accumulate": self.accumulate,
            "num_batches": self.num_batches,
            "total_steps": self.total_steps,
            "warmup_steps": cfg.TRAIN.warmup_steps,
            "lr_start": cfg.TRAIN.lr_start,
            "lr_min": cfg.TRAIN.lr_min,
            "world_size": self.world_size,
            "data": OmegaConf.to_container(cfg.DATA, resolve=True),
            "train_times": [str(t) for t in train_dataset.initial_conditions],
        }
        if cfg.TRAIN.resume_path:
            self.resume(cfg.TRAIN.resume_path)

    def save(self, name="last"):
        rng_states = [capture_rng()]
        if self.world_size > 1:
            paddle.distributed.all_gather_object(rng_states, rng_states[0])
        if self.rank != 0:
            return
        state = {
            "format_version": 2,
            "ema": self.ema.state_dict(),
            "rng": rng_states,
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "micro_step": self.micro_step,
            "best_metric": self.best_metric,
            "signature": self.signature,
        }
        # Native PaddleScience checkpoint files; pdstates also holds U-Cast's
        # EMA, RNG and sampler position for exact mid-epoch resumption.
        prefix = f"{name}_raw"
        save_load.save_checkpoint(
            self.model,
            self.optimizer,
            {"metric": self.best_metric, "epoch": self.epoch, "ucast": state},
            self.scaler if self.use_amp else None,
            str(self.output_dir),
            prefix,
            print_log=False,
        )
        root = self.output_dir / "checkpoints"
        atomic_save(
            {"format_version": 2, "solver_checkpoint": prefix}, root / f"{name}.pdckpt"
        )
        with self.ema.apply(self.model):
            atomic_save(self.model.state_dict(), root / f"{name}.pdparams")

    def resume(self, path):
        path = Path(path)
        if path.suffix == ".pdckpt":
            state = paddle.load(str(path), return_numpy=True)
            if state.get("format_version") == 2:
                path = path.parent / state["solver_checkpoint"]
            elif state.get("format_version") == 1:
                self._restore_training_state(state, legacy=True)
                return
            else:
                raise ValueError("Unsupported U-Cast checkpoint format")
        if path.suffix in (".pdparams", ".pdopt", ".pdstates", ".pdscaler"):
            path = path.with_suffix("")
        states = paddle.load(str(path) + ".pdstates", return_numpy=True)
        if "ucast" not in states:
            raise ValueError(
                "Checkpoint lacks U-Cast EMA/RNG state; use last_raw/best_raw, or warm-start weights"
            )
        state = states["ucast"]
        if state["signature"] != self.signature:
            raise ValueError(
                "Resume configuration/data differs from the checkpoint; use pretrained_model_path for a new stage"
            )
        save_load.load_checkpoint(
            str(path), self.model, self.optimizer, self.scaler if self.use_amp else None
        )
        self._restore_training_state(state)

    def _restore_training_state(self, state, legacy=False):
        if state["signature"] != self.signature:
            raise ValueError(
                "Resume configuration/data differs from the checkpoint; use pretrained_model_path for a new stage"
            )
        if legacy:
            self.model.set_state_dict(state["model"])
            self.optimizer.set_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
        self.ema.set_state_dict(state["ema"])
        self.epoch, self.next_batch = state["epoch"], state["next_batch"]
        self.micro_step, self.best_metric = state["micro_step"], state["best_metric"]
        self.resume_rng = state["rng"][self.rank]
        if self.next_batch and self.next_batch % self.accumulate:
            raise ValueError("Resume checkpoint must be at an optimizer-step boundary")
        restore_rng(self.resume_rng)
        logger.info(
            f"Resumed epoch={self.epoch}, next_batch={self.next_batch}, optimizer_step={self.optimizer.step_count}"
        )

    def validate(self):
        with self.ema.apply(self.model):
            self.solver.eval()
        report = self.solver.ucast_metrics.report
        if self.rank == 0:
            (self.output_dir / "validation.json").write_text(
                json.dumps(report, indent=2)
            )
        metric = report[f"val/avg/{self.train_cfg.monitor}"]
        self.solver.cur_metric = metric
        improved = metric < self.best_metric
        if improved:
            self.best_metric = metric
        self.model.train()
        return improved

    def fit(self):
        from ucast_solver import SolverCallbacks, TrainingComplete, build_solver

        if self.rank == 0:
            (self.output_dir / "config.yaml").write_text(
                OmegaConf.to_yaml(self.cfg, resolve=True)
            )
        self.solver = build_solver(self)
        callbacks = SolverCallbacks(self)
        self.solver.register_callback_on_epoch_begin(callbacks.epoch_begin)
        self.solver.register_callback_on_epoch_end(callbacks.epoch_end)
        self.solver.register_callback_on_iter_begin(callbacks.iter_begin)
        self.solver.register_callback_on_iter_end(callbacks.iter_end)
        self.solver.best_metric["epoch"] = self.epoch
        try:
            if (
                self.train_cfg.max_steps is None
                or self.optimizer.step_count < self.train_cfg.max_steps
            ):
                self.solver.train()
        except TrainingComplete:
            pass
        return self.model


def train(cfg):
    cfg = normalize_config(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.seed(cfg.seed)
    world_size = int(os.environ.get("PADDLE_TRAINERS_NUM", "1"))
    device = cfg.device
    if world_size > 1:
        device = f"gpu:{os.environ.get('FLAGS_selected_gpus', '0').split(',')[0]}"
    paddle.set_device(device)
    if world_size > 1 and not paddle.distributed.is_initialized():
        paddle.distributed.init_parallel_env()
    stage = cfg.TRAIN.stage
    if stage not in ("det", "prob"):
        raise ValueError("TRAIN.stage must be det or prob")
    if stage == "prob" and not (
        cfg.TRAIN.pretrained_model_path or cfg.TRAIN.resume_path
    ):
        raise ValueError(
            "Stage 2 requires TRAIN.pretrained_model_path from stage 1, or resume_path"
        )
    channels = len(VARIABLE_NAMES)
    if (
        cfg.MODEL.out_channels != channels
        or cfg.MODEL.in_channels
        != channels * cfg.DATA.window + 4 + len(cfg.DATA.static_fields)
    ):
        raise ValueError(
            "MODEL channels must match ERA5 variables, time window, four forcings and static fields"
        )
    dataset = ensure_latitude_is_ascending(
        open_era5_zarr(cfg.DATA.data_dir, cfg.DATA.dataset, cfg.DATA.get("cache_dir"))
    )
    normalizer = load_normalizer(cfg.DATA.stats_dir, dataset)
    static = load_static_conditions(dataset, cfg.DATA.static_fields)
    train_times = select_initial_conditions(
        dataset,
        cfg.DATA.train_slices,
        cfg.DATA.window,
        1,
        cfg.DATA.hourly_resolution,
        max_samples=cfg.TRAIN.max_train_samples,
    )
    val_dataset = None
    if cfg.TRAIN.validate:
        val_times = select_initial_conditions(
            dataset,
            cfg.DATA.val_slices,
            cfg.DATA.window,
            cfg.TRAIN.val_horizon,
            cfg.DATA.hourly_resolution,
            cfg.DATA.val_initial_hours,
            cfg.TRAIN.max_val_samples,
        )
        train_end = max(train_times) + np.timedelta64(cfg.DATA.hourly_resolution, "h")
        val_begin = min(val_times) - np.timedelta64(
            (cfg.DATA.window - 1) * cfg.DATA.hourly_resolution, "h"
        )
        if train_end >= val_begin:
            raise ValueError(
                "Train and validation windows overlap; adjust DATA train_slices/val_slices"
            )
        val_dataset = ERA5WindowDataset(
            dataset,
            normalizer,
            static,
            val_times,
            cfg.DATA.window,
            cfg.TRAIN.val_horizon,
            cfg.DATA.hourly_resolution,
        )
    train_dataset = ERA5WindowDataset(
        dataset,
        normalizer,
        static,
        train_times,
        cfg.DATA.window,
        1,
        cfg.DATA.hourly_resolution,
    )
    model = ppsci.arch.UCast(**cfg.MODEL)
    if cfg.TRAIN.pretrained_model_path and not cfg.TRAIN.resume_path:
        load_weights(model, cfg.TRAIN.pretrained_model_path)
    trainer = UCastTrainer(cfg, model, train_dataset, val_dataset, normalizer)
    logger.info(
        f"U-Cast stage={stage}, train={len(train_dataset)}, val={len(val_dataset) if val_dataset is not None else 0}, "
        f"optimizer_steps={trainer.total_steps}, world_size={trainer.world_size}"
    )
    trainer.fit()
    return trainer


def evaluate_checkpoint(cfg):
    """Evaluate a saved EMA model using the original training-area metrics."""
    cfg = normalize_config(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.seed(cfg.seed)
    paddle.set_device(cfg.device)
    dataset = ensure_latitude_is_ascending(
        open_era5_zarr(cfg.DATA.data_dir, cfg.DATA.dataset, cfg.DATA.get("cache_dir"))
    )
    normalizer = load_normalizer(cfg.DATA.stats_dir, dataset)
    static = load_static_conditions(dataset, cfg.DATA.static_fields)
    initial = select_initial_conditions(
        dataset,
        cfg.DATA.val_slices,
        cfg.DATA.window,
        cfg.TRAIN.val_horizon,
        cfg.DATA.hourly_resolution,
        cfg.DATA.val_initial_hours,
        cfg.TRAIN.max_val_samples,
    )
    windows = ERA5WindowDataset(
        dataset,
        normalizer,
        static,
        initial,
        cfg.DATA.window,
        cfg.TRAIN.val_horizon,
        cfg.DATA.hourly_resolution,
    )
    model = ppsci.arch.UCast(**cfg.MODEL)
    load_weights(model, cfg.INFER.pretrained_model_path, use_ema=True)
    use_amp = bool(cfg.INFER.use_amp) and paddle.get_device().startswith("gpu")
    from ucast_solver import build_evaluation_solver

    solver = build_evaluation_solver(cfg, model, windows, normalizer, use_amp)
    solver.eval()
    report = solver.ucast_metrics.report
    output = Path(cfg.TRAIN.output_dir) / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    logger.info(
        f"Evaluation saved to {output}: "
        + json.dumps({k: v for k, v in report.items() if k != "per_lead_variable"})
    )
    return report
