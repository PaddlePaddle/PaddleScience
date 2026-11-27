# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""U-Cast's hybrid optimizer, fair CRPS, schedule and adaptive EMA.

The explicit optimizer keeps one gradient-scaling boundary for both branches.
Its state is indexed by structured model names, so a restarted process does not
have to reproduce Paddle's generated parameter names.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import paddle


def fair_crps(predictions, target):
    """Fair ensemble CRPS, predictions [E, B, C, lon, lat]."""
    count = predictions.shape[0]
    if count < 2:
        raise ValueError("Fair CRPS training requires at least two members")
    skill = (predictions - target.unsqueeze(0)).abs().mean(axis=0)
    # Accumulate pairs without allocating the E x E x B x C x lon x lat array.
    spread = paddle.zeros_like(skill)
    for i in range(count):
        for j in range(i):
            spread = spread + (predictions[i] - predictions[j]).abs()
    return skill - spread / (count * (count - 1))


def weighted_loss(predictions, target, weights, kind):
    predictions = predictions.astype("float32")
    target = target.astype("float32")
    if kind == "wcrps":
        errors = fair_crps(predictions, target)
    elif kind == "wmae":
        errors = (predictions - target.unsqueeze(0)).abs().mean(axis=0)
    elif kind == "wmse":
        errors = (predictions - target.unsqueeze(0)).square().mean(axis=0)
    else:
        raise ValueError(f"Unknown U-Cast loss: {kind}")
    return (errors * weights).mean()


def warmup_cosine(step, base_lr, warmup_steps, total_steps, start_lr=1e-8, min_lr=1e-8):
    """Match the original scheduler, including its warmup endpoint."""
    if warmup_steps and step == 0:
        return start_lr
    if 0 < step < warmup_steps:
        return start_lr + (base_lr - start_lr) * step / max(1, warmup_steps - 1)
    if step == warmup_steps:
        return base_lr
    progress = min(
        1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    )
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def muon_momentum(step, total_steps, warmup_steps, maximum=0.95):
    minimum = maximum - 0.1
    cooldown = warmup_steps // 10
    if warmup_steps and step < warmup_steps:
        return minimum + (maximum - minimum) * step / warmup_steps
    if cooldown and step > total_steps - cooldown:
        return (
            maximum - (maximum - minimum) * (step - (total_steps - cooldown)) / cooldown
        )
    return maximum


def zeropower_newton_schulz(gradient, steps=5, ns_dtype="bfloat16"):
    """Source Newton-Schulz arithmetic, preserving the PR's BF16 GEMM layout."""
    if ns_dtype not in ("bfloat16", "float32"):
        raise ValueError("ns_dtype must be bfloat16 or float32")
    matrix = gradient.reshape([gradient.shape[0], -1])
    matrix = matrix.astype(ns_dtype if gradient.place.is_gpu_place() else "float32")
    transposed = matrix.shape[0] > matrix.shape[1]
    matrix = matrix / (
        paddle.linalg.norm(
            matrix.astype("float32"), p="fro", axis=(-2, -1), keepdim=True
        ).astype(matrix.dtype)
        + 1e-7
    )
    for _ in range(steps):
        gram = paddle.matmul(
            matrix, matrix, transpose_x=transposed, transpose_y=not transposed
        )
        polynomial = -4.7750 * gram + (2.0315 * gram) @ gram
        correction = (
            paddle.matmul(polynomial, matrix, transpose_y=True).transpose([1, 0])
            if transposed
            else polynomial @ matrix
        )
        matrix = 3.4445 * matrix + correction
    return matrix.reshape(gradient.shape).astype(gradient.dtype)


def clip_grad_norm(parameters, maximum):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return 0.0
    norm = paddle.sqrt(
        paddle.add_n([g.astype("float32").square().sum() for g in gradients])
    )
    factor = paddle.clip(float(maximum) / (norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.set_value(gradient * factor)
    return float(norm)


class HybridMuonAdamW(paddle.optimizer.Optimizer):
    """Muon for hidden matrix/conv weights; AdamW for output/norm/bias.

    Exposes Paddle GradScaler's public optimizer protocol (step, clear_grad)
    and its parameter list. All branches are skipped together on overflow.
    Training uses FP32 master parameters with O1 mixed-precision forward passes.
    """

    def __init__(
        self,
        model,
        adam_lr=3e-4,
        muon_lr=3e-3,
        adam_weight_decay=0.1,
        muon_weight_decay=0.03,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        momentum=0.95,
        use_muon=True,
        ns_steps=5,
        ns_dtype="bfloat16",
        gradient_clip_norm=None,
    ):
        self.parameters = {
            n: p for n, p in model.named_parameters() if not p.stop_gradient
        }
        super().__init__(
            learning_rate=float(adam_lr), parameters=list(self.parameters.values())
        )
        if ns_dtype not in ("bfloat16", "float32"):
            raise ValueError("ns_dtype must be bfloat16 or float32")
        self.ns_dtype = ns_dtype
        self.gradient_clip_norm = gradient_clip_norm
        self.last_grad_norm = None
        self.muon_names = {
            name
            for name, param in self.parameters.items()
            if use_muon
            and param.ndim >= 2
            and not any(
                part in name
                for part in ("output_layer", "out_layer", "out_conv", "output_conv")
            )
        }
        self.base_adam_lr, self.base_muon_lr = float(adam_lr), float(muon_lr)
        self.adam_lr, self.muon_lr = float(adam_lr), float(muon_lr)
        self.adam_weight_decay, self.muon_weight_decay = (
            adam_weight_decay,
            muon_weight_decay,
        )
        self.betas, self.epsilon = tuple(betas), epsilon
        self.maximum_momentum = momentum
        self.momentum, self.ns_steps = momentum, ns_steps
        self.state = {}
        self.step_count = 0

    def schedule(self, warmup_steps, total_steps, start_lr=1e-8, min_lr=1e-8):
        args = (warmup_steps, total_steps, start_lr, min_lr)
        self.adam_lr = warmup_cosine(self.step_count, self.base_adam_lr, *args)
        self.muon_lr = warmup_cosine(self.step_count, self.base_muon_lr, *args)
        # Original Muon uses its configured momentum on the very first update.
        if self.step_count:
            self.momentum = muon_momentum(
                self.step_count, total_steps, warmup_steps, self.maximum_momentum
            )

    @paddle.no_grad()
    def step(self):
        # Paddle GradScaler delegates the finite check to Optimizer subclasses
        # through this auxiliary flag. Skip both branches atomically.
        found_inf = self._get_auxiliary_var("found_inf")
        if found_inf is not None and bool(found_inf):
            return
        if self.gradient_clip_norm is not None:
            self.last_grad_norm = clip_grad_norm(
                self._parameter_list, self.gradient_clip_norm
            )
        beta1, beta2 = self.betas
        for name, param in self.parameters.items():
            if param.grad is None:
                continue
            grad = param.grad.astype("float32")
            state = self.state.setdefault(name, {})
            if name in self.muon_names:
                if not state:
                    state["momentum"] = paddle.zeros_like(param)
                momentum = state["momentum"]
                momentum.set_value(paddle.lerp(momentum, grad, 1 - self.momentum))
                update = paddle.lerp(grad, momentum, self.momentum)
                update = zeropower_newton_schulz(update, self.ns_steps, self.ns_dtype)
                # The released implementation uses the original tensor's last
                # two dimensions here (kernel H/W for convolution weights).
                update = update * math.sqrt(max(1, grad.shape[-2] / grad.shape[-1]))
                param.set_value(
                    param * (1 - self.muon_lr * self.muon_weight_decay)
                    - self.muon_lr * update
                )
            else:
                if not state:
                    state.update(
                        m=paddle.zeros_like(param), v=paddle.zeros_like(param), step=0
                    )
                state["step"] += 1
                state["m"].set_value(beta1 * state["m"] + (1 - beta1) * grad)
                state["v"].set_value(beta2 * state["v"] + (1 - beta2) * grad.square())
                decay = (
                    self.adam_weight_decay
                    if "weight" in name and "norm" not in name
                    else 0.0
                )
                denominator = (
                    state["v"].sqrt() / math.sqrt(1 - beta2 ** state["step"])
                    + self.epsilon
                )
                update = state["m"] / denominator / (1 - beta1 ** state["step"])
                param.set_value(
                    param * (1 - self.adam_lr * decay) - self.adam_lr * update
                )
        self.step_count += 1

    def minimize(self, loss, startup_program=None, parameters=None, no_grad_set=None):
        """Solver/GradScaler calls this after backward and finite-gradient checks."""
        self.step()
        return [], [(p, p.grad) for p in self._parameter_list if p.grad is not None]

    def get_lr(self):
        return self.adam_lr

    def clear_grad(self):
        for param in self._parameter_list:
            param.clear_gradient()

    def state_dict(self):
        return {
            "state": self.state,
            "step_count": self.step_count,
            "muon_names": sorted(self.muon_names),
            "adam_lr": self.adam_lr,
            "muon_lr": self.muon_lr,
            "momentum": self.momentum,
        }

    def set_state_dict(self, state):
        if set(state["muon_names"]) != self.muon_names:
            raise ValueError(
                "Checkpoint optimizer parameter groups differ from the model"
            )
        if set(state["state"]) - set(self.parameters):
            raise ValueError("Checkpoint contains unknown optimizer parameters")
        self.state = {}
        for name, values in state["state"].items():
            self.state[name] = {
                key: paddle.to_tensor(value, place=self.parameters[name].place)
                if isinstance(value, (paddle.Tensor, np.ndarray))
                else value
                for key, value in values.items()
            }
        self.step_count = state["step_count"]
        self.adam_lr, self.muon_lr = state["adam_lr"], state["muon_lr"]
        self.momentum = state["momentum"]


class AdaptiveEMA:
    def __init__(self, model, decay=0.9999, enabled=True):
        self.enabled = bool(enabled)
        self.decay, self.num_updates = float(decay), 0
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if self.enabled and not p.stop_gradient
        }

    @paddle.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in model.named_parameters():
            if name in self.shadow:
                shadow = self.shadow[name]
                shadow.set_value(shadow - (1 - decay) * (shadow - param))

    @contextmanager
    def apply(self, model):
        backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }
        try:
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.set_value(self.shadow[name])
            yield
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.set_value(backup[name])

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def set_state_dict(self, state):
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("Checkpoint EMA parameters differ from the model")
        self.decay, self.num_updates = state["decay"], state["num_updates"]
        for name, shadow in self.shadow.items():
            shadow.set_value(state["shadow"][name])
