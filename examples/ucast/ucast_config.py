# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Translate public PaddleScience options into U-Cast's resumable task options."""
from __future__ import annotations

import copy

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def normalize_config(cfg):
    """Keep public Hydra options and earlier U-Cast aliases compatible."""
    cfg = copy.deepcopy(cfg)
    # Hydra supplies a structured config; aliases are added only to this copy.
    OmegaConf.set_struct(cfg, False)
    train, data = cfg.TRAIN, cfg.DATA

    def value(mapping, name, fallback):
        result = mapping.get(name)
        return fallback if result is None else result

    output = train.get("output_dir")
    if output is None:
        raw = OmegaConf.to_container(cfg, resolve=False).get("output_dir")
        output = (
            cfg.output_dir
            if raw and ("${hydra:" not in str(raw) or HydraConfig.initialized())
            else "./outputs_ucast"
        )
    train.output_dir = cfg.output_dir = str(output)
    train.stage = {"deterministic": "det", "probabilistic": "prob"}.get(
        train.stage, train.stage
    )
    scheduler = train.get("scheduler", {})
    ema = train.get("ema", {})
    aliases = {
        "accumulate_grad_batches": train.get("update_freq", 1),
        "ensemble_size": cfg.MODEL.get("training_ensemble_size", 1),
        "val_ensemble_size": cfg.MODEL.get("eval_ensemble_size", 1),
        "use_amp": cfg.get("use_amp", False),
        "gradient_clip_norm": train.get("grad_clip", 1.0),
        "ema_decay": ema.get("decay", 0.9999),
        "warmup_steps": scheduler.get("warmup_steps", 1500),
        "lr_start": scheduler.get("warmup_start_lr", 1e-8),
        "lr_min": scheduler.get("eta_min", 1e-8),
        "resume_path": train.get("checkpoint_path"),
        "validate": train.get("eval_during_train", True),
        "val_horizon": train.get("prediction_horizon", 4),
        "monitor": "crps" if train.stage == "prob" else "rmse",
        "log_freq": cfg.get("log_freq", 20),
        "max_steps": None,
        "max_batches_per_epoch": None,
        "save_freq_steps": 0,
    }
    for name, fallback in aliases.items():
        train[name] = value(train, name, fallback)
    train.ema_enabled = ema.get("use_ema", True)
    train.amp_init_loss_scaling = train.get("amp_init_loss_scaling", 65536.0)
    if cfg.get("amp_level", "O1") != "O1":
        raise ValueError("U-Cast training uses AMP O1 with FP32 optimizer/EMA weights")
    if data.get("horizon", 1) != 1:
        raise ValueError(
            "U-Cast trains on single-step residual targets (DATA.horizon=1)"
        )
    data.train_slices = value(data, "train_slices", train.get("train_time_slices"))
    data.val_slices = value(data, "val_slices", [train.get("val_time_slice")])
    data.val_initial_hours = value(
        data, "val_initial_hours", train.get("possible_initial_hours_eval")
    )
    opt = train.optimizer
    train.optimizer = {
        "adam_lr": value(opt, "adam_lr", opt.get("learning_rate", 3e-4)),
        "muon_lr": value(opt, "muon_lr", opt.get("muon_learning_rate", 0.003)),
        "adam_weight_decay": value(
            opt, "adam_weight_decay", opt.get("weight_decay", 0.1)
        ),
        "muon_weight_decay": opt.get("muon_weight_decay", 0.03),
        "betas": opt.get("betas", [opt.get("beta1", 0.9), opt.get("beta2", 0.95)]),
        "epsilon": opt.get("epsilon", 1e-8),
        "momentum": value(opt, "momentum", opt.get("muon_momentum", 0.95)),
        "use_muon": opt.get("use_muon", True),
        "ns_steps": value(opt, "ns_steps", opt.get("muon_ns_steps", 5)),
        "ns_dtype": value(opt, "ns_dtype", opt.get("muon_ns_dtype", "bfloat16")),
    }
    return cfg
