# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""U-Cast adapters for the unchanged PaddleScience Solver training/eval loops."""
from __future__ import annotations

import json

import numpy as np
import paddle
from ucast_optimization import weighted_loss
from ucast_training import forecast_moments, restore_rng
from ucast_utils import autoregressive_inference

import ppsci
from ppsci.utils import logger


class SupervisedWindows(paddle.io.Dataset):
    """Expose ERA5 windows as Solver (input, label, weight) dictionaries."""

    def __init__(self, windows, input_keys, output_key, evaluation=False):
        super().__init__()
        self.windows, self.evaluation = windows, evaluation
        self.input_keys = (
            tuple(input_keys)
            if not evaluation
            else ("dynamics_normalized", "dynamics_raw", "forcing")
        )
        self.label_keys = (output_key,) if not evaluation else ("mse",)
        self.indices = np.arange(len(windows))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.windows[int(self.indices[index])]
        inputs = {key: sample[key] for key in self.input_keys}
        if self.evaluation:
            labels = {
                "mse": np.zeros(
                    (self.windows.horizon, len(self.windows.variable_names)), np.float32
                )
            }
        else:
            labels = {self.label_keys[0]: sample["target"]}
        return inputs, labels, {}


class UCastTask(paddle.nn.Layer):
    """Training ensemble expansion and compact autoregressive validation output.

    The underlying network remains ppsci.arch.UCast. Validation returns spatial
    score moments, so Solver gathers only [initial condition, lead, variable]
    arrays, rather than retaining every ensemble's full weather fields.
    """

    def __init__(self, cfg, network, windows=None, normalizer=None, use_amp=False):
        super().__init__()
        self.network = network
        self.input_keys, self.output_keys = tuple(cfg.MODEL.input_keys), tuple(
            cfg.MODEL.output_keys
        )
        self.members = int(cfg.TRAIN.ensemble_size)
        self.val_members = int(cfg.TRAIN.val_ensemble_size)
        self.windows, self.normalizer, self.use_amp = windows, normalizer, use_amp
        self.eval_forward_batch_size = cfg.TRAIN.get("eval_forward_batch_size")

    def forward(self, inputs):
        if self.training:
            batch_size = inputs[self.input_keys[0]].shape[0]
            expanded = {
                key: inputs[key]
                .unsqueeze(0)
                .expand([self.members, *inputs[key].shape])
                .reshape([-1, *inputs[key].shape[1:]])
                for key in self.input_keys
            }
            # Already expanded by the task; do not ensemble again in UCast.forward.
            packed = paddle.concat([expanded[key] for key in self.input_keys], axis=1)
            output = self.network.forward_tensor(packed)
            output = output.reshape([self.members, batch_size, *output.shape[1:]])
            return {self.output_keys[0]: output.transpose([1, 0, 2, 3, 4])}
        self.network.enable_mc_dropout()
        moments = {}
        for index in range(inputs["dynamics_raw"].shape[0]):
            sample = {
                key: inputs[key][index]
                for key in ("dynamics_normalized", "dynamics_raw", "forcing")
            }
            result = autoregressive_inference(
                self.network,
                sample,
                paddle.to_tensor(self.windows.static_condition),
                self.normalizer,
                self.windows.window,
                self.windows.horizon,
                self.val_members,
                self.use_amp,
                self.eval_forward_batch_size,
            )
            for key, value in forecast_moments(result, self.windows.latitude).items():
                moments.setdefault(key, []).append(value)
        return {
            key: paddle.to_tensor(np.stack(values)) for key, values in moments.items()
        }


class ForecastMetrics:
    """Aggregate MSE/variance before taking RMSE and finite-ensemble SSR."""

    def __init__(self, windows, members, monitor):
        self.windows, self.members, self.monitor = windows, members, monitor
        self.report = None

    def __call__(self, output, labels):
        values = {key: value.numpy().mean(axis=0) for key, value in output.items()}
        values["rmse"] = np.sqrt(values.pop("mse"))
        if "variance" in values:
            values["ssr"] = np.sqrt(
                values.pop("variance") * (self.members + 1) / self.members
            ) / np.maximum(values["rmse"], 1e-12)
        if self.monitor not in values:
            raise ValueError(
                f"Validation monitor {self.monitor} requires more ensemble members"
            )
        averages = {key: float(value.mean()) for key, value in values.items()}
        if not all(np.isfinite(value) for value in averages.values()):
            raise FloatingPointError("Non-finite validation metric")
        self.report = {f"val/avg/{key}": value for key, value in averages.items()}
        self.report["per_lead_variable"] = {
            key: {
                name: value[:, i].tolist()
                for i, name in enumerate(self.windows.variable_names)
            }
            for key, value in values.items()
        }
        # Solver uses the first returned metric for checkpoint selection.
        order = [self.monitor, *[key for key in averages if key != self.monitor]]
        return {key: paddle.to_tensor(averages[key], dtype="float64") for key in order}


def build_validator(cfg, windows):
    adapter = SupervisedWindows(
        windows, cfg.MODEL.input_keys, cfg.MODEL.output_keys[0], True
    )
    metrics = ForecastMetrics(
        windows, int(cfg.TRAIN.val_ensemble_size), cfg.TRAIN.monitor
    )
    keys = ("mse", "l1") + (
        ("crps", "variance") if cfg.TRAIN.val_ensemble_size > 1 else ()
    )
    validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": adapter,
            "batch_size": int(cfg.TRAIN.get("eval_batch_size", 1)),
            "num_workers": 0,
        },
        output_expr={key: (lambda out, k=key: out[k]) for key in keys},
        metric={"forecast": ppsci.metric.FunctionalMetric(metrics)},
        name="ERA5",
    )
    return validator, metrics


def build_evaluation_solver(cfg, model, windows, normalizer, use_amp):
    validator, metrics = build_validator(cfg, windows)
    solver = ppsci.solver.Solver(
        UCastTask(cfg, model, windows, normalizer, use_amp),
        output_dir=cfg.TRAIN.output_dir,
        validator={validator.name: validator},
        device=paddle.get_device(),
        # O1 leaves model parameters unchanged. Solver's constructor assumes
        # amp.decorate(model, None) returns a pair, but Paddle returns only the
        # model. Enable eval autocast below after optimizer-free construction.
        use_amp=False,
        amp_level="O1",
        eval_with_no_grad=True,
        log_freq=cfg.TRAIN.log_freq,
    )
    solver.use_amp = use_amp
    solver.ucast_metrics = metrics
    return solver


def build_solver(trainer):
    cfg = trainer.cfg
    dataset = SupervisedWindows(
        trainer.train_dataset, cfg.MODEL.input_keys, cfg.MODEL.output_keys[0]
    )

    def loss_expr(output, label, weight):
        key = cfg.MODEL.output_keys[0]
        loss = weighted_loss(
            output[key].transpose([1, 0, 2, 3, 4]),
            label[key],
            trainer.weights,
            cfg.TRAIN.loss,
        )
        if not bool(paddle.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite training loss at batch {trainer.micro_step}"
            )
        trainer.last_loss = float(loss)
        return {cfg.TRAIN.loss: loss}

    constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": dataset,
            "batch_size": trainer.batch_size,
            "num_workers": int(cfg.TRAIN.num_workers),
            "seed": int(cfg.seed),
        },
        ppsci.loss.FunctionalLoss(loss_expr),
        name="ERA5",
    )
    validators, metrics = None, None
    if trainer.val_dataset is not None:
        validator, metrics = build_validator(cfg, trainer.val_dataset)
        validators = {validator.name: validator}
    solver = ppsci.solver.Solver(
        UCastTask(
            cfg, trainer.model, trainer.val_dataset, trainer.normalizer, trainer.use_amp
        ),
        constraint={constraint.name: constraint},
        output_dir=str(trainer.output_dir),
        optimizer=trainer.optimizer,
        epochs=int(cfg.TRAIN.epochs),
        save_freq=int(cfg.TRAIN.get("save_freq", 1)),
        iters_per_epoch=trainer.num_batches,
        update_freq=trainer.accumulate,
        log_freq=int(cfg.TRAIN.log_freq),
        validator=validators,
        device=paddle.get_device(),
        use_amp=trainer.use_amp,
        amp_level="O1",
        eval_with_no_grad=True,
        # Validation is invoked through solver.eval() by the epoch callback
        # under the original adaptive EMA. Solver's epoch EMA is disabled.
        eval_during_train=False,
    )
    if trainer.use_amp:
        solver.scaler = trainer.scaler
    solver.ucast_metrics = metrics
    return solver


class TrainingComplete(Exception):
    """Stop at a configured optimizer-step limit after a complete checkpoint."""


class SolverCallbacks:
    """Schedule, microbatch EMA and exact sampler resume around Solver's loop."""

    def __init__(self, trainer):
        self.trainer = trainer

    def epoch_begin(self, solver):
        t = self.trainer
        constraint = next(iter(solver.constraint.values()))
        # Each loader owns an immutable index array: mutating a dataset while
        # Paddle's previous iterator is still prefetching can race its thread.
        dataset = SupervisedWindows(
            t.train_dataset, t.cfg.MODEL.input_keys, t.cfg.MODEL.output_keys[0]
        )
        indices = np.random.default_rng(int(t.cfg.seed) + t.epoch).permutation(
            len(t.train_dataset)
        )
        indices = indices[: len(indices) // t.world_size * t.world_size]
        dataset.indices = indices[: t.num_batches * t.batch_size * t.world_size]
        constraint.data_loader = ppsci.data.build_dataloader(
            dataset,
            {
                "dataset": dataset,
                "batch_size": t.batch_size,
                "num_workers": int(t.train_cfg.num_workers),
                "seed": int(t.cfg.seed),
            },
        )
        constraint.data_iter = iter(constraint.data_loader)
        for _ in range(t.next_batch):
            next(constraint.data_iter)
        if t.resume_rng is not None:
            restore_rng(t.resume_rng)
            t.resume_rng = None
        solver.iters_per_epoch = t.num_batches - t.next_batch
        solver.global_step = t.micro_step
        solver.max_steps = int(t.cfg.TRAIN.epochs) * t.num_batches
        solver.model.train()
        t.optimizer.clear_grad()

    def iter_begin(self, solver):
        t = self.trainer
        cfg = t.train_cfg
        t.optimizer.last_grad_norm = None
        self.previous_step = t.optimizer.step_count
        t.optimizer.schedule(
            int(cfg.warmup_steps), t.total_steps, cfg.lr_start, cfg.lr_min
        )

    def iter_end(self, solver):
        t, cfg = self.trainer, self.trainer.train_cfg
        t.micro_step += 1
        t.next_batch += 1
        t.ema.update(t.model)
        advanced = t.optimizer.step_count != self.previous_step
        if t.rank == 0 and (t.micro_step % cfg.log_freq == 0 or advanced):
            record = {
                "engine": "ppsci.solver.Solver",
                "epoch": t.epoch,
                "micro_step": t.micro_step,
                "optimizer_step": t.optimizer.step_count,
                "loss": t.last_loss,
                "adam_lr": t.optimizer.adam_lr,
                "muon_lr": t.optimizer.muon_lr,
                "muon_momentum": t.optimizer.momentum,
                "grad_norm": t.optimizer.last_grad_norm,
                "ema_updates": t.ema.num_updates,
            }
            with (t.output_dir / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            logger.info(json.dumps(record))
        if t.next_batch < t.num_batches:
            if cfg.max_steps is not None and t.optimizer.step_count >= cfg.max_steps:
                t.save("last")
                raise TrainingComplete
            if (
                advanced
                and cfg.save_freq_steps
                and t.optimizer.step_count % cfg.save_freq_steps == 0
            ):
                t.save("last")

    def epoch_end(self, solver):
        t, cfg = self.trainer, self.trainer.train_cfg
        t.epoch += 1
        t.next_batch = 0
        if (
            t.val_dataset is not None
            and t.epoch >= cfg.get("start_eval_epoch", 1)
            and t.epoch % cfg.eval_freq == 0
        ):
            if t.validate():
                t.save("best")
        t.save("last")
        solver.best_metric.update(metric=t.best_metric, epoch=t.epoch)
        if (cfg.max_steps is not None and t.optimizer.step_count >= cfg.max_steps) or (
            cfg.get("stop_after_epoch") is not None and t.epoch >= cfg.stop_after_epoch
        ):
            raise TrainingComplete
