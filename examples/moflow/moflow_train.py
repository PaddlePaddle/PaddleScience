# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import path as osp

import hydra
import moflow_transform
import numpy as np
import paddle
from moflow_utils import Hyperparameters
from moflow_utils import check_validity
from omegaconf import DictConfig
from tabulate import tabulate

import ppsci
from ppsci.utils import logger


def infer(model, batch_size=20, temp=0.7, z_mu=None, true_adj=None):
    """generate mols

    Args:
        model (object): Generated eval Moflownet model
        batch_size (int, optional): Batch size during evaling per GPU. Defaults to 20.
        temp (float, optional): temperature of the gaussian distribution. Defaults to 0.7.
        z_mu (int, optional): latent vector of a molecule. Defaults to None.
        true_adj (paddle.Tensor, optional): True Adjacency. Defaults to None.

    Returns:
        Tuple(paddle.Tensor, paddle.Tensor): Adjacency and nodes
    """
    z_dim = model.b_size + model.a_size
    mu = np.zeros(z_dim)
    sigma_diag = np.ones(z_dim)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[: model.b_size] = (
                np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[: model.b_size]
            )
            sigma_diag[model.b_size + 1 :] = (
                np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size + 1 :]
            )
    sigma = temp * sigma_diag
    with paddle.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        z = np.random.normal(mu, sigma, (batch_size, z_dim))
        z = paddle.to_tensor(data=z).astype(paddle.get_default_dtype())
        adj, x = model.reverse(z, true_adj=true_adj)
    return adj, x


class eval_func:
    def __init__(
        self,
        metrics_mode,
        batch_size,
        atomic_num_list,
        *args,
    ):
        super().__init__()
        self.metrics_mode = metrics_mode
        self.batch_size = batch_size
        self.atomic_num_list = atomic_num_list

    def __call__(
        self,
        output_dict,
        label_dict,
    ):
        self.metrics_mode.eval()
        adj, x = infer(self.metrics_mode, self.batch_size)
        validity_info = check_validity(adj, x, self.atomic_num_list)
        self.metrics_mode.train()
        results = dict()
        results["valid"] = validity_info["valid_ratio"]
        results["unique"] = validity_info["unique_ratio"]
        results["abs_unique"] = validity_info["abs_unique_ratio"]
        return results


def train(cfg: DictConfig):
    # set training hyper-parameters
    b_hidden_ch = cfg.get(cfg.data_name).b_hidden_ch
    a_hidden_gnn = cfg.get(cfg.data_name).a_hidden_gnn
    a_hidden_lin = cfg.get(cfg.data_name).a_hidden_lin
    mask_row_size_list = list(cfg.get(cfg.data_name).mask_row_size_list)
    mask_row_stride_list = list(cfg.get(cfg.data_name).mask_row_stride_list)
    a_n_type = len(cfg.get(cfg.data_name).atomic_num_list)
    atomic_num_list = list(cfg.get(cfg.data_name).atomic_num_list)

    model_params = Hyperparameters(
        b_n_type=cfg.get(cfg.data_name).b_n_type,
        b_n_flow=cfg.get(cfg.data_name).b_n_flow,
        b_n_block=cfg.get(cfg.data_name).b_n_block,
        b_n_squeeze=cfg.get(cfg.data_name).b_n_squeeze,
        b_hidden_ch=b_hidden_ch,
        b_affine=True,
        b_conv_lu=cfg.get(cfg.data_name).b_conv_lu,
        a_n_node=cfg.get(cfg.data_name).a_n_node,
        a_n_type=a_n_type,
        a_hidden_gnn=a_hidden_gnn,
        a_hidden_lin=a_hidden_lin,
        a_n_flow=cfg.get(cfg.data_name).a_n_flow,
        a_n_block=cfg.get(cfg.data_name).a_n_block,
        mask_row_size_list=mask_row_size_list,
        mask_row_stride_list=mask_row_stride_list,
        a_affine=True,
        learn_dist=cfg.get(cfg.data_name).learn_dist,
        seed=cfg.seed,
        noise_scale=cfg.get(cfg.data_name).noise_scale,
    )

    logger.info("Model params:\n" + tabulate(model_params.print()))

    # set transforms
    if cfg.data_name == "qm9":
        transform_fn = moflow_transform.transform_fn
    elif cfg.data_name == "zinc250k":
        transform_fn = moflow_transform.transform_fn_zinc250k

    # set select eval data
    valid_idx_path = osp.join(cfg.FILE_PATH, cfg.get(cfg.data_name).valid_idx)
    valid_idx = moflow_transform.get_val_ids(valid_idx_path, cfg.data_name)

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MOlFLOWDataset",
            "file_path": cfg.FILE_PATH,
            "data_name": cfg.data_name,
            "mode": cfg.mode,
            "valid_idx": valid_idx,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.get(cfg.data_name).label_keys,
            "smiles_col": cfg.get(cfg.data_name).smiles_col,
            "transform_fn": transform_fn,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": cfg.TRAIN.num_workers,
    }

    # set model
    model_cfg = dict(cfg.MODEL)
    model_cfg.update({"hyper_params": model_params})
    model = ppsci.arch.MoFlowNet(**model_cfg)

    # set constraint
    output_keys = cfg.MODEL.output_keys
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.FunctionalLoss(model.log_prob_loss),
        {key: (lambda out, k=key: out[k]) for key in output_keys},
        name="Sup_constraint",
    )

    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # init optimizer and lr scheduler
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MOlFLOWDataset",
            "file_path": cfg.FILE_PATH,
            "data_name": cfg.data_name,
            "mode": "eval",
            "valid_idx": valid_idx,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.get(cfg.data_name).label_keys,
            "smiles_col": cfg.get(cfg.data_name).smiles_col,
            "transform_fn": transform_fn,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.FunctionalLoss(model.log_prob_loss),
        {key: (lambda out, k=key: out[k]) for key in output_keys},
        metric={
            "Valid": ppsci.metric.FunctionalMetric(
                eval_func(model, cfg.EVAL.batch_size, atomic_num_list)
            )
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        seed=cfg.seed,
        validator=validator,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # train model
    solver.train()

    # validation for training
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="moflow_train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
