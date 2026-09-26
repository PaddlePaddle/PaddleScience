# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-command entry: `python main.py mode=train` / `mode=eval` (cwd = examples/climax).

Set CLIMAX_USE_CINN=1 to run under to_static+CINN: the env var toggles both
the CINN FLAGS below and the `to_static` flag passed to Solver.
"""

import os

# CINN switch: a single env var toggles both the CINN FLAGS below and the
# `to_static` flag passed to Solver. Must run before any `import paddle`
# for the FLAGS to take effect.
_USE_CINN = os.environ.get("CLIMAX_USE_CINN", "0") == "1"
if _USE_CINN:
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"

import hydra  # noqa: E402
import paddle  # noqa: E402
import utils  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import ppsci  # noqa: E402
from ppsci.arch.climax.pos_embed import interpolate_pos_embed  # noqa: E402
from ppsci.data.dataset.climatebench_dataset import ClimateBenchDataset  # noqa: E402
from ppsci.data.dataset.climatebench_dataset import Normalize  # noqa: E402
from ppsci.data.process import batch_transform  # noqa: E402


def _dataset_kwargs(cfg: DictConfig, partition: str) -> dict:
    """Constructor kwargs of ClimateBenchDataset for the given partition."""
    return {
        "root_dir": cfg.DATA.root_dir,
        "input_keys": cfg.MODEL.input_keys,
        "label_keys": cfg.MODEL.output_keys,
        "partition": partition,
        "history": cfg.DATA.history,
        "list_train_simu": cfg.DATA.list_train_simu,
        "list_test_simu": cfg.DATA.list_test_simu,
        "variables": cfg.DATA.variables,
        "out_variables": cfg.DATA.out_variables,
        "train_ratio": cfg.DATA.train_ratio,
        "seed": cfg.seed,
    }


def _make_validator(cfg: DictConfig, dataset, lat, denorm, clim, name: str):
    """Build a SupervisedValidator over an already-normalized dataset."""
    dataloader_cfg = {
        "dataset": dataset,
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": cfg.DATA.num_workers,
        "collate_fn": batch_transform.default_collate_fn,
    }
    return ppsci.validate.SupervisedValidator(
        dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(utils.make_w_mse_loss(lat)),
        metric=utils.make_lat_weighted_metrics(
            lat, transform=denorm, clim=clim, log_postfix=None
        ),
        name=name,
    )


def _make_denorm(dataset_train) -> Normalize:
    """Inverse of the train out_transform."""
    mean_norm = dataset_train.out_transform.mean
    std_norm = dataset_train.out_transform.std
    return Normalize(-mean_norm / std_norm, 1 / std_norm)


def _load_weights_strict(model, path: str):
    """Strictly load weights, detecting the checkpoint structure.

    Handles both the wrapped initial state ``{"state_dict": {"net.*": ...}}``
    and plain state dicts saved by Solver; any missing/unexpected key raises.
    """
    state = paddle.load(path)
    if "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net.") for k in state):
        state = {k[len("net.") :]: v for k, v in state.items() if k.startswith("net.")}
    missing, unexpected = model.set_state_dict(state)
    print(
        "Loaded weights from %s: missing=%s unexpected=%s"
        % (path, sorted(missing), sorted(unexpected))
    )
    if missing or unexpected:
        raise RuntimeError(f"strict weight load failed from {path}")


def _load_mae_weights(model, path: str):
    """Leniently load a pre-trained MAE checkpoint.

    Interpolates the positional embedding to the model grid, renames channel
    embeddings to var embeddings and drops token embedding / head keys and
    shape-mismatched keys so the remaining backbone weights load cleanly.
    """
    checkpoint = paddle.load(path=str(path))
    print("Loading pre-trained checkpoint from: %s" % path)
    checkpoint_model = checkpoint["state_dict"]
    # interpolate positional embedding (expects "net."-prefixed keys)
    interpolate_pos_embed(model, checkpoint_model, new_size=model.img_size)
    # strip the "net." prefix -> ClimaXClimateBench-level keys
    checkpoint_model = {
        k[len("net.") :]: v for k, v in checkpoint_model.items() if k.startswith("net.")
    }
    state_dict = model.state_dict()
    if model.parallel_patch_embed:
        if "token_embeds.proj_weights" not in checkpoint_model.keys():
            raise ValueError(
                "Pretrained checkpoint does not have token_embeds.proj_weights"
                " for parallel processing. Please convert the checkpoints first"
                " or disable parallel patch_embed tokenization."
            )
    for k in list(checkpoint_model.keys()):
        if "channel" in k:
            checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
            del checkpoint_model[k]
        if "token_embeds" in k or "head" in k:  # initialize embedding from scratch
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            continue
    for k in list(checkpoint_model.keys()):
        if (
            k not in state_dict.keys()
            or checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # paddle set_state_dict is lenient by default (torch strict=False)
    print(model.set_state_dict(checkpoint_model))


def _make_optimizer(model, lr_scheduler, cfg: DictConfig) -> paddle.optimizer.AdamW:
    """AdamW with decay/no_decay parameter groups.

    Parameters named var_embed/pos_embed/time_pos_embed get no weight decay;
    frozen parameters (stop_gradient, set by freeze_encoder) are excluded.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.stop_gradient:
            continue
        if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=cfg.TRAIN.beta_1,
        beta2=cfg.TRAIN.beta_2,
        epsilon=1e-8,
        parameters=[
            {"params": decay, "weight_decay": cfg.TRAIN.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
    )


def train(cfg: DictConfig):
    # set train dataloader config (dataset name resolved by build_dataset)
    train_dataloader_cfg = {
        "dataset": {"name": "ClimateBenchDataset", **_dataset_kwargs(cfg, "train")},
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": cfg.DATA.num_workers,
        "collate_fn": batch_transform.default_collate_fn,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(utils.mse_loss),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # val/test datasets share the transforms fitted on the train partition
    dataset_train = sup_constraint.data_loader.dataset
    dataset_val = ClimateBenchDataset(**_dataset_kwargs(cfg, "val"))
    dataset_val.set_normalize(dataset_train.inp_transform, dataset_train.out_transform)
    dataset_test = ClimateBenchDataset(**_dataset_kwargs(cfg, "test"))
    dataset_test.set_normalize(dataset_train.inp_transform, dataset_train.out_transform)

    lat = dataset_val.lat
    denorm = _make_denorm(dataset_train)
    clim = dataset_test.y_normalization

    # set validators (Val during training, Test for mode=eval and after train).
    # Val comes last so the solver's best-model metric tracks Val/w_mse,
    # matching the original implementation's checkpoint monitor.
    sup_validator_val = _make_validator(cfg, dataset_val, lat, denorm, clim, "Val")
    sup_validator_test = _make_validator(cfg, dataset_test, lat, denorm, clim, "Test")
    validator = {
        sup_validator_test.name: sup_validator_test,
        sup_validator_val.name: sup_validator_val,
    }

    # set model
    model = ppsci.arch.ClimaXClimateBench(**cfg.MODEL)

    # load weights BEFORE building the Solver: the full initial state takes
    # precedence over the cleaned MAE checkpoint
    if cfg.TRAIN.init_state_path:
        _load_weights_strict(model, cfg.TRAIN.init_state_path)
    elif cfg.TRAIN.pretrained_path:
        _load_mae_weights(model, cfg.TRAIN.pretrained_path)

    # init lr scheduler and optimizer; the schedule is step-unit and is
    # advanced once per iteration by the solver's by_epoch=False branch
    lr_scheduler = ppsci.optimizer.lr_scheduler.LinearWarmupCosineAnnealingLR(
        **cfg.TRAIN.lr_scheduler
    )
    optimizer = _make_optimizer(model, lr_scheduler, cfg)

    # initialize solver (env CLIMAX_USE_CINN overrides cfg.to_static)
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=ITERS_PER_EPOCH,
        log_freq=cfg.log_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        save_freq=cfg.TRAIN.save_freq,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        to_static=_USE_CINN,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # transforms are fitted on the train partition, same as in training
    dataset_train = ClimateBenchDataset(**_dataset_kwargs(cfg, "train"))
    dataset_test = ClimateBenchDataset(**_dataset_kwargs(cfg, "test"))
    dataset_test.set_normalize(dataset_train.inp_transform, dataset_train.out_transform)

    lat = dataset_test.lat
    denorm = _make_denorm(dataset_train)
    clim = dataset_test.y_normalization

    sup_validator_test = _make_validator(cfg, dataset_test, lat, denorm, clim, "Test")
    validator = {sup_validator_test.name: sup_validator_test}

    # set model
    model = ppsci.arch.ClimaXClimateBench(**cfg.MODEL)

    # load weights manually: load_pretrain cannot unwrap {"state_dict": ...}
    # checkpoints, so the structure is detected here instead
    if cfg.EVAL.pretrained_model_path:
        _load_weights_strict(model, cfg.EVAL.pretrained_model_path)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        validator=validator,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        to_static=_USE_CINN,
    )
    # evaluate
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="climax.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
