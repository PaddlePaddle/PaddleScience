"""Train or evaluate a six-hour forecast pair with PaddleScience Solver."""

import dataclasses
import json
import math
from pathlib import Path

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from omegaconf import OmegaConf

import ppsci
from ppsci.arch.aurora import Batch
from ppsci.arch.aurora.normalisation import level_to_str
from ppsci.arch.aurora.normalisation import scales


def spatial_sample(batch, stride):
    """Subsample coordinates and fields together; retain every pressure level."""
    if stride < 1:
        raise ValueError("DATA.spatial_stride must be positive")
    if batch.metadata.lat.ndim != 1 or batch.metadata.lon.ndim != 1:
        raise ValueError("This example requires a regular latitude/longitude grid")
    return dataclasses.replace(
        batch,
        surf_vars={k: v[..., ::stride, ::stride] for k, v in batch.surf_vars.items()},
        atmos_vars={k: v[..., ::stride, ::stride] for k, v in batch.atmos_vars.items()},
        static_vars={k: v[::stride, ::stride] for k, v in batch.static_vars.items()},
        metadata=dataclasses.replace(
            batch.metadata,
            lat=batch.metadata.lat[::stride],
            lon=batch.metadata.lon[::stride],
        ),
    )


def validate_pair(batch, target, timestep, patch_size):
    """Fail before optimization when dates, variables, levels or grids differ."""
    if len(batch.metadata.time) != 1 or len(target.metadata.time) != 1:
        raise ValueError("This example accepts exactly one forecast pair")
    if any(v.shape[1] != 2 for v in batch.surf_vars.values()) or any(
        v.shape[1] != 2 for v in batch.atmos_vars.values()
    ):
        raise ValueError("Input must contain two historical frames")
    if any(v.shape[1] != 1 for v in target.surf_vars.values()) or any(
        v.shape[1] != 1 for v in target.atmos_vars.values()
    ):
        raise ValueError("Target must contain exactly one future frame")
    if target.metadata.time[0] != batch.metadata.time[0] + timestep:
        raise ValueError("Target time must be exactly one model timestep after input")
    if tuple(batch.metadata.atmos_levels) != tuple(target.metadata.atmos_levels):
        raise ValueError("Input and target pressure levels differ")
    cropped = batch.crop(patch_size)
    target = target.crop(patch_size)
    for name in ("lat", "lon"):
        if not np.array_equal(
            getattr(cropped.metadata, name).numpy(),
            getattr(target.metadata, name).numpy(),
        ):
            raise ValueError(f"Input and target {name} coordinates differ")
    for kind in ("surf_vars", "atmos_vars"):
        if set(getattr(batch, kind)) != set(getattr(target, kind)):
            raise ValueError(f"Input and target {kind} differ")
    return target


def normalized_mse(levels):
    """Average nine per-variable MSEs using Aurora's official variable scales."""

    def expression(outputs, labels, weights=None):
        terms = []
        for key, label in labels.items():
            kind, variable = key.split("_", 1)
            if kind == "atmos":
                scale = paddle.to_tensor(
                    [scales[f"{variable}_{level_to_str(level)}"] for level in levels],
                    dtype=outputs[key].dtype,
                ).reshape([1, 1, -1, 1, 1])
            else:
                scale = scales[variable]
            term = ((outputs[key] - label) / scale).square()
            if weights:
                term = term * weights[key]
            terms.append(term.mean())
        return {"normalized_mse": paddle.stack(terms).mean()}

    return expression


@hydra.main(version_base=None, config_path="conf", config_name="aurora")
def main(cfg: DictConfig):
    if cfg.mode not in ("train", "eval"):
        raise ValueError("mode must be train or eval")
    if cfg.MODEL.name != "AuroraArch":
        raise ValueError("MODEL.name must be AuroraArch")
    if cfg.MODEL.model_type not in ("AuroraPretrained", "AuroraSmallPretrained"):
        raise ValueError(
            "This six-hour example supports Pretrained and SmallPretrained"
        )
    if cfg.mode == "eval" and not cfg.MODEL.checkpoint:
        raise ValueError("Evaluation requires MODEL.checkpoint")
    if not cfg.MODEL.checkpoint and not cfg.TRAIN.allow_random_init:
        raise ValueError("Set MODEL.checkpoint to converted native weights")
    if min(cfg.TRAIN.epochs, cfg.TRAIN.iters_per_epoch) < 1:
        raise ValueError("Training epochs and iterations must be positive")
    paddle.set_device(cfg.device)
    ppsci.utils.misc.set_random_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ppsci.utils.logger.init_logger("ppsci", str(output / f"{cfg.mode}.log"), "info")
    batch = spatial_sample(Batch.from_netcdf(cfg.DATA.input), cfg.DATA.spatial_stride)
    target = spatial_sample(Batch.from_netcdf(cfg.DATA.target), cfg.DATA.spatial_stride)
    model_cfg = OmegaConf.to_container(cfg.MODEL, resolve=True)
    checkpoint = model_cfg.pop("checkpoint")
    model_cfg.update(metadata=batch.metadata, static_vars=batch.static_vars)
    model = ppsci.arch.build_model(model_cfg)
    if checkpoint:
        model.model.load_checkpoint_local(str(checkpoint))
    target = validate_pair(batch, target, model.model.timestep, model.model.patch_size)
    inputs = {f"surf_{k}": v.numpy() for k, v in batch.surf_vars.items()}
    inputs.update({f"atmos_{k}": v.numpy() for k, v in batch.atmos_vars.items()})
    labels = {f"surf_{k}": v.numpy() for k, v in target.surf_vars.items()}
    labels.update({f"atmos_{k}": v.numpy() for k, v in target.atmos_vars.items()})
    dataloader = {
        "dataset": {"name": "NamedArrayDataset", "input": inputs, "label": labels},
        "sampler": {"name": "BatchSampler", "shuffle": False},
        "batch_size": 1,
        "num_workers": 0,
    }
    expression = normalized_mse(batch.metadata.atmos_levels)
    loss = ppsci.loss.FunctionalLoss(expression)
    validator = ppsci.validate.SupervisedValidator(
        dataloader,
        loss,
        metric={"normalized": ppsci.metric.FunctionalMetric(expression)},
        name="forecast_pair",
    )
    constraints, optimizer = None, None
    if cfg.mode == "train":
        constraint = ppsci.constraint.SupervisedConstraint(
            dataloader, loss, name="forecast"
        )
        constraints = {constraint.name: constraint}
        optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)
    solver = ppsci.solver.Solver(
        model,
        constraints,
        optimizer=optimizer,
        validator={validator.name: validator},
        output_dir=str(output),
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        save_freq=1,
        log_freq=1,
        device=cfg.device,
        eval_with_no_grad=True,
    )
    if cfg.mode == "train":
        solver.train()
        paddle.save(model.model.state_dict(), str(output / "model.pdparams"))
    value, metrics = solver.eval()
    if not math.isfinite(float(value)):
        raise ValueError("Non-finite evaluation metric")
    summary = {
        "mode": cfg.mode,
        "model": cfg.MODEL.model_type,
        "input_grid": list(batch.spatial_shape),
        "pressure_levels_hpa": [float(level) for level in batch.metadata.atmos_levels],
        "loss": "Mean per-variable official-scale normalized MSE",
        "evaluation": metrics,
        "scope": "One supervised pair; evaluation on the same pair is a workflow check only",
    }
    (output / f"{cfg.mode}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
