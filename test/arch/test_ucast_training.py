# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

import ppsci

# Example modules are imported after adding their directory.
# ruff: noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "ucast"))
from smoke_data import make_smoke_data
from ucast import infer
from ucast_data import (
    latitude_area_weights,
    select_initial_conditions,
    training_weights,
)
from ucast_optimization import fair_crps, warmup_cosine
from ucast_training import evaluate_checkpoint, train
from ucast_utils import ensure_latitude_is_ascending, open_era5_zarr


@pytest.fixture(scope="module")
def fixture_data(tmp_path_factory):
    return make_smoke_data(tmp_path_factory.mktemp("ucast_data"))


def config(data, output):
    source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "ucast"
        / "conf"
        / "ucast.yaml"
    )
    cfg = OmegaConf.load(source)
    cfg.device = os.environ.get("UCAST_TEST_DEVICE", "cpu")
    cfg.mode = "train"
    cfg.MODEL.model_channels = 8
    cfg.MODEL.channel_mult = [1, 2]
    cfg.MODEL.num_blocks = 1
    cfg.MODEL.attn_levels = [1]
    cfg.MODEL.channels_per_head = 4
    cfg.MODEL.image_resolution = 8
    cfg.DATA.data_dir, cfg.DATA.dataset = str(data), "era5.zarr"
    cfg.DATA.stats_dir = str(data / "stats")
    cfg.DATA.train_slices = [["2000-01-01T00", "2000-01-04T00"]]
    cfg.DATA.val_slices = [["2000-01-06T00", "2000-01-10T18"]]
    cfg.TRAIN.output_dir = str(output)
    cfg.TRAIN.epochs = 2
    cfg.TRAIN.batch_size = 1
    cfg.TRAIN.accumulate_grad_batches = 2
    cfg.TRAIN.max_batches_per_epoch = 4
    cfg.TRAIN.max_train_samples = 8
    cfg.TRAIN.max_val_samples = 2
    cfg.TRAIN.val_horizon = 2
    cfg.TRAIN.use_amp = os.environ.get("UCAST_TEST_AMP", "0") == "1"
    cfg.TRAIN.warmup_steps = 0
    return cfg


def test_area_and_pressure_weights():
    lat = np.array([-90, -45, 0, 45, 90])
    area = latitude_area_weights(lat)
    assert np.all(area > 0)
    np.testing.assert_allclose(area.mean(), 1.0)
    np.testing.assert_allclose(area, area[::-1])
    weights = training_weights(
        lat, ["2m_temperature", "sea_surface_temperature", "temperature_500"]
    )
    np.testing.assert_allclose(weights[0, 0], area)
    np.testing.assert_allclose(weights[1, 0], 0.1 * area)
    assert weights[2].mean() > weights[1].mean()


def test_fair_crps_analytic_and_gradients():
    prediction = paddle.to_tensor(
        np.array([0.0, 2.0], np.float32).reshape(2, 1, 1, 1, 1), stop_gradient=False
    )
    truth = paddle.ones([1, 1, 1, 1])
    score = fair_crps(prediction, truth)
    np.testing.assert_allclose(score.numpy(), 0.0)
    score.sum().backward()
    assert np.isfinite(prediction.grad.numpy()).all()
    with pytest.raises(ValueError, match="two members"):
        fair_crps(prediction[:1], truth)


def test_scheduler_original_endpoints():
    assert warmup_cosine(0, 0.1, 4, 10) == 1e-8
    assert warmup_cosine(3, 0.1, 4, 10) == pytest.approx(0.1)
    assert warmup_cosine(4, 0.1, 4, 10) == 0.1
    assert warmup_cosine(10, 0.1, 4, 10) == 1e-8


def test_time_streams_and_missing_times(fixture_data):
    dataset = ensure_latitude_is_ascending(
        open_era5_zarr(str(fixture_data), "era5.zarr")
    )
    slices = [["2000-01-01T00", "2000-01-04T00"], ["2000-01-01T06", "2000-01-04T00"]]
    times = select_initial_conditions(dataset, slices, 2, 1, 12)
    assert {int(t.astype("datetime64[h]").astype(int) % 24) for t in times} == {
        0,
        6,
        12,
        18,
    }
    missing = dataset.time.values[4]
    dataset = dataset.drop_sel(time=missing)
    times = select_initial_conditions(dataset, slices, 2, 1, 12)
    for initial in times:
        assert missing not in initial + np.arange(-1, 2) * np.timedelta64(12, "h")


@pytest.mark.parametrize("batches", [4, 5])
def test_training_resume_matches_uninterrupted(
    fixture_data, tmp_path, monkeypatch, batches
):
    calls = {"train": 0, "eval": 0, "epochs": 0}
    official_train = ppsci.solver.Solver.train
    official_eval = ppsci.solver.Solver.eval
    from ppsci.solver import train as solver_train

    official_epoch = solver_train.train_epoch_func

    def train_entry(solver):
        calls["train"] += 1
        return official_train(solver)

    def eval_entry(solver, *args, **kwargs):
        calls["eval"] += 1
        return official_eval(solver, *args, **kwargs)

    def epoch_entry(*args, **kwargs):
        calls["epochs"] += 1
        return official_epoch(*args, **kwargs)

    monkeypatch.setattr(ppsci.solver.Solver, "train", train_entry)
    monkeypatch.setattr(ppsci.solver.Solver, "eval", eval_entry)
    monkeypatch.setattr(solver_train, "train_epoch_func", epoch_entry)
    cfg = config(fixture_data, tmp_path / "continuous")
    cfg.TRAIN.max_batches_per_epoch = batches
    expected_steps = 2 * ((batches + 1) // 2)
    continuous = train(cfg)
    assert isinstance(continuous.solver, ppsci.solver.Solver)
    assert isinstance(
        next(iter(continuous.solver.constraint.values())),
        ppsci.constraint.SupervisedConstraint,
    )
    assert isinstance(
        next(iter(continuous.solver.validator.values())),
        ppsci.validate.SupervisedValidator,
    )
    assert calls["train"] == 1 and calls["epochs"] == 2 and calls["eval"] == 2
    assert continuous.optimizer.step_count == expected_steps
    assert continuous.ema.num_updates == 2 * batches
    assert (tmp_path / "continuous/checkpoints/best.pdparams").is_file()
    interrupted_cfg = copy.deepcopy(cfg)
    interrupted_cfg.TRAIN.output_dir = str(tmp_path / "interrupted")
    interrupted_cfg.TRAIN.max_steps = 1
    interrupted = train(interrupted_cfg)
    assert interrupted.epoch == 0 and interrupted.next_batch == 2
    resumed_cfg = copy.deepcopy(interrupted_cfg)
    resumed_cfg.TRAIN.max_steps = None
    # Resume directly from the official PaddleScience checkpoint prefix.
    resumed_cfg.TRAIN.resume_path = str(tmp_path / "interrupted/checkpoints/last_raw")
    for suffix in (".pdparams", ".pdopt", ".pdstates"):
        assert Path(resumed_cfg.TRAIN.resume_path + suffix).is_file()
    resumed = train(resumed_cfg)
    assert resumed.optimizer.step_count == expected_steps
    assert resumed.ema.num_updates == 2 * batches
    # AMP/BF16 GPU reductions vary even between uninterrupted repetitions.
    # Keep FP32 strict; the six-update AMP repetition measured < 6e-4 drift.
    tolerance = 1e-3 if continuous.use_amp else 2e-6
    for name, parameter in continuous.model.state_dict().items():
        np.testing.assert_allclose(
            parameter.numpy(),
            resumed.model.state_dict()[name].numpy(),
            atol=tolerance,
            rtol=tolerance,
            err_msg=name,
        )
        np.testing.assert_allclose(
            continuous.ema.shadow[name].numpy(),
            resumed.ema.shadow[name].numpy(),
            atol=tolerance,
            rtol=tolerance,
            err_msg=name,
        )
    # An architecture/schedule change must not silently resume stale state.
    invalid_cfg = copy.deepcopy(resumed_cfg)
    invalid_cfg.TRAIN.accumulate_grad_batches = 1
    with pytest.raises(ValueError, match="differs"):
        train(invalid_cfg)


def test_stage2_and_deep_ensemble(fixture_data, tmp_path):
    cfg = config(fixture_data, tmp_path / "det")
    cfg.TRAIN.max_steps = 1
    stage1 = train(cfg)
    assert stage1.optimizer.step_count == 1
    cfg.TRAIN.stage, cfg.TRAIN.loss = "prob", "wcrps"
    cfg.TRAIN.ensemble_size, cfg.TRAIN.val_ensemble_size = 2, 3
    # PR model also has an ensemble wrapper. Solver must expand members once,
    # while preserving its recomputation/custom attention AMP backward.
    cfg.MODEL.training_ensemble_size, cfg.MODEL.eval_ensemble_size = 2, 3
    cfg.MODEL.use_recompute = True
    cfg.TRAIN.output_dir = str(tmp_path / "prob")
    cfg.TRAIN.pretrained_model_path = str(
        tmp_path / "det/checkpoints/last_raw.pdparams"
    )
    cfg.TRAIN.max_steps = None
    cfg.TRAIN.epochs = 1
    cfg.TRAIN.accumulate_grad_batches = 1
    cfg.TRAIN.max_batches_per_epoch = 1
    cfg.TRAIN.batch_size = 2
    cfg.TRAIN.optimizer.adam_lr = 7e-5
    cfg.TRAIN.optimizer.muon_lr = 0.007
    cfg.TRAIN.optimizer.muon_weight_decay = 0.0
    stage2 = train(cfg)
    assert stage2.optimizer.step_count == 1
    assert np.isfinite(stage2.best_metric)
    assert isinstance(stage2.solver.optimizer, paddle.optimizer.Optimizer)
    # The legacy-style manifest resolves to the same native Solver files.
    restored_cfg = copy.deepcopy(cfg)
    restored_cfg.TRAIN.resume_path = str(tmp_path / "prob/checkpoints/last.pdckpt")
    restored = train(restored_cfg)
    assert restored.optimizer.step_count == 1
    assert restored.ema.num_updates == 1
    cfg.INFER.pretrained_model_paths = [
        str(tmp_path / "det/checkpoints/last.pdparams"),
        str(tmp_path / "prob/checkpoints/last.pdparams"),
    ]
    cfg.INFER.prediction_horizon = 2
    cfg.INFER.ensemble_size, cfg.INFER.ensemble_chunk_size = 7, 2
    cfg.INFER.ic_start_dates = ["2000-01-07T00"]
    cfg.INFER.use_amp = False
    cfg.INFER.output_path = str(tmp_path / "deep.nc")
    cfg.INFER.score_output_path = str(tmp_path / "scores.json")
    forecasts = infer(cfg)
    assert forecasts[0].sizes["ensemble_member"] == 7
    assert forecasts[0].sizes["lead_time"] == 2
    np.testing.assert_array_equal(forecasts[0].source_model, [0, 0, 0, 0, 1, 1, 1])
    assert (tmp_path / "deep.nc").is_file() and (tmp_path / "scores.json").is_file()
    import xarray as xr

    with xr.open_dataset(tmp_path / "deep.nc") as restored:
        assert restored.sizes["ensemble_member"] == 7
        np.testing.assert_array_equal(
            restored.lead_time.values, np.array([12, 24], dtype="timedelta64[h]")
        )
        assert np.isfinite(restored["2m_temperature_predicted"].values).all()
    cfg.INFER.pretrained_model_path = str(tmp_path / "prob/checkpoints/best.pdckpt")
    cfg.TRAIN.val_ensemble_size = 3
    cfg.INFER.use_amp = cfg.TRAIN.use_amp
    report = evaluate_checkpoint(cfg)
    assert np.isfinite(report["val/avg/crps"])
    assert (tmp_path / "prob/evaluation.json").is_file()


@pytest.mark.skipif(not paddle.is_compiled_with_cuda(), reason="CUDA Paddle required")
@pytest.mark.parametrize("api", ["step", "minimize"])
def test_amp_overflow_skips_both_optimizer_branches(api):
    from ucast_optimization import HybridMuonAdamW
    from ucast_training import clip_grad_norm

    if paddle.device.cuda.device_count() == 0:
        pytest.skip("No CUDA device")
    previous = paddle.get_device()
    paddle.set_device("gpu")
    try:

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.hidden = paddle.nn.Conv2D(1, 4, 1)
                self.out_conv = paddle.nn.Conv2D(4, 1, 1)

            def forward(self, value):
                return self.out_conv(self.hidden(value))

        model = Model()
        optimizer = HybridMuonAdamW(model)
        before = {
            name: parameter.numpy().copy()
            for name, parameter in model.named_parameters()
        }
        scaler = paddle.amp.GradScaler(init_loss_scaling=128)
        loss = scaler.scale(model(paddle.ones([1, 1, 4, 4])).mean())
        loss.backward()
        model.hidden.weight.grad.set_value(
            paddle.full_like(model.hidden.weight.grad, float("inf"))
        )
        if api == "step":
            scaler.unscale_(optimizer)
            clip_grad_norm(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            scaler.minimize(optimizer, loss)
        assert optimizer.step_count == 0
        assert not optimizer.state
        for name, parameter in model.named_parameters():
            np.testing.assert_array_equal(parameter.numpy(), before[name])
        optimizer.clear_grad()
        finite_loss = scaler.scale(model(paddle.ones([1, 1, 4, 4])).mean())
        finite_loss.backward()
        scaler.minimize(optimizer, finite_loss)
        assert optimizer.step_count == 1
        assert all(np.isfinite(p.numpy()).all() for p in model.parameters())
    finally:
        paddle.set_device(previous)


def test_forecast_metrics_reduce_moments_before_rmse():
    from types import SimpleNamespace

    from ucast_solver import ForecastMetrics

    windows = SimpleNamespace(variable_names=("temperature",), horizon=1)
    metrics = ForecastMetrics(windows, members=2, monitor="crps")
    output = {
        key: paddle.to_tensor(np.array(values, np.float64).reshape(2, 1, 1))
        for key, values in {
            "mse": [1, 9],
            "l1": [1, 3],
            "variance": [4, 4],
            "crps": [0.5, 1.5],
        }.items()
    }
    result = metrics(output, {})
    assert next(iter(result)) == "crps"
    assert metrics.report["val/avg/rmse"] == pytest.approx(np.sqrt(5))
    assert metrics.report["val/avg/ssr"] == pytest.approx(np.sqrt(6 / 5))
    assert metrics.report["val/avg/crps"] == pytest.approx(1.0)


def test_legacy_checkpoint_can_resume(fixture_data, tmp_path):
    from ucast_training import capture_rng

    cfg = config(fixture_data, tmp_path / "source")
    cfg.TRAIN.max_steps = 1
    cfg.TRAIN.validate = False
    original = train(cfg)
    legacy = {
        "format_version": 1,
        "model": original.model.state_dict(),
        "optimizer": original.optimizer.state_dict(),
        "ema": original.ema.state_dict(),
        "scaler": original.scaler.state_dict(),
        "rng": [capture_rng()],
        "epoch": original.epoch,
        "next_batch": original.next_batch,
        "micro_step": original.micro_step,
        "best_metric": original.best_metric,
        "signature": original.signature,
    }
    path = tmp_path / "legacy.pdckpt"
    paddle.save(legacy, str(path))
    cfg.TRAIN.output_dir = str(tmp_path / "resumed")
    cfg.TRAIN.resume_path = str(path)
    cfg.TRAIN.max_steps = 2
    resumed = train(cfg)
    assert resumed.optimizer.step_count == 2
    assert resumed.ema.num_updates == 4
    assert resumed.epoch == 1


def test_pr_config_epoch_resume(fixture_data, tmp_path):
    """Published PR config names run the official loop and resume exact epochs."""
    cfg = config(fixture_data, tmp_path / "continuous")
    cfg.output_dir, cfg.TRAIN.output_dir = str(tmp_path / "continuous"), None
    cfg.TRAIN.stage = "deterministic"
    cfg.TRAIN.update_freq = cfg.TRAIN.accumulate_grad_batches
    cfg.TRAIN.accumulate_grad_batches = None
    cfg.TRAIN.train_time_slices = cfg.DATA.train_slices
    cfg.TRAIN.val_time_slice = cfg.DATA.val_slices[0]
    cfg.DATA.train_slices = cfg.DATA.val_slices = None
    cfg.TRAIN.prediction_horizon = cfg.TRAIN.val_horizon
    del cfg.TRAIN.val_horizon
    cfg.use_amp = cfg.TRAIN.use_amp
    del cfg.TRAIN.use_amp
    cfg.TRAIN.scheduler.warmup_steps = cfg.TRAIN.warmup_steps
    del cfg.TRAIN.warmup_steps
    cfg.TRAIN.eval_batch_size = 2
    cfg.TRAIN.eval_forward_batch_size = 1
    cfg.MODEL.use_recompute = True
    # Match Hydra CLI struct mode, which rejects undeclared alias fields.
    OmegaConf.set_struct(cfg, True)
    continuous = train(cfg)
    partial_cfg = copy.deepcopy(cfg)
    partial_cfg.output_dir = str(tmp_path / "partial")
    partial_cfg.TRAIN.stop_after_epoch = 1
    partial = train(partial_cfg)
    assert partial.epoch == 1
    resumed_cfg = copy.deepcopy(partial_cfg)
    resumed_cfg.TRAIN.stop_after_epoch = None
    resumed_cfg.TRAIN.checkpoint_path = str(tmp_path / "partial/checkpoints/last_raw")
    resumed = train(resumed_cfg)
    assert resumed.epoch == 2
    assert resumed.optimizer.step_count == continuous.optimizer.step_count
    tolerance = 1e-3 if continuous.use_amp else 2e-6
    for name, param in continuous.model.state_dict().items():
        np.testing.assert_allclose(
            param.numpy(),
            resumed.model.state_dict()[name].numpy(),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            continuous.ema.shadow[name].numpy(),
            resumed.ema.shadow[name].numpy(),
            rtol=tolerance,
            atol=tolerance,
        )
