# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""
Reference: https://github.com/zhry10/PhyLSTM.git
"""

from os import path as osp

import functions
import hydra
import numpy as np
import scipy.io
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    mat = scipy.io.loadmat(cfg.data_file)
    ag_data = mat["input_tf"]  # ag, ad, av
    u_data = mat["target_X_tf"]
    ut_data = mat["target_Xd_tf"]
    utt_data = mat["target_Xdd_tf"]
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

    t = mat["time"]
    dt = t[0, 1] - t[0, 0]

    ag_all = ag_data
    u_all = u_data
    u_t_all = ut_data
    u_tt_all = utt_data

    # finite difference
    N = u_data.shape[1]
    phi1 = np.concatenate(
        [
            np.array([-3 / 2, 2, -1 / 2]),
            np.zeros([N - 3]),
        ]
    )
    temp1 = np.concatenate([-1 / 2 * np.identity(N - 2), np.zeros([N - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([N - 2, 2]), 1 / 2 * np.identity(N - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate(
        [
            np.zeros([N - 3]),
            np.array([1 / 2, -2, 3 / 2]),
        ]
    )
    phi_t0 = (
        1
        / dt
        * np.concatenate(
            [
                np.reshape(phi1, [1, phi1.shape[0]]),
                phi2,
                np.reshape(phi3, [1, phi3.shape[0]]),
            ],
            axis=0,
        )
    )
    phi_t0 = np.reshape(phi_t0, [1, N, N])

    ag_star = ag_all[0:10]
    eta_star = u_all[0:10]
    eta_t_star = u_t_all[0:10]
    eta_tt_star = u_tt_all[0:10]
    ag_c_star = ag_all[0:50]
    lift_star = -ag_c_star

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    ag_c = ag_c_star
    g = -eta_tt - ag
    phi_t = np.repeat(phi_t0, ag_c_star.shape[0], axis=0)

    model = ppsci.arch.DeepPhyLSTM(
        cfg.MODEL.phylstm2_net.input_size,
        eta.shape[2],
        cfg.MODEL.phylstm2_net.hidden_size,
        cfg.MODEL.phylstm2_net.model_type,
    )
    model.register_input_transform(functions.transform_in)
    model.register_output_transform(functions.transform_out)

    dataset_obj = functions.Dataset(eta, eta_t, g, ag, ag_c, lift, phi_t)
    (
        input_dict_train,
        label_dict_train,
        input_dict_val,
        label_dict_val,
    ) = dataset_obj.get(cfg.TRAIN.epochs)

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func2),
        {
            "eta_pred": lambda out: out["eta_pred"],
            "eta_dot_pred": lambda out: out["eta_dot_pred"],
            "g_pred": lambda out: out["g_pred"],
            "eta_t_pred_c": lambda out: out["eta_t_pred_c"],
            "eta_dot_pred_c": lambda out: out["eta_dot_pred_c"],
            "lift_pred_c": lambda out: out["lift_pred_c"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func2),
        {
            "eta_pred": lambda out: out["eta_pred"],
            "eta_dot_pred": lambda out: out["eta_dot_pred"],
            "g_pred": lambda out: out["g_pred"],
            "eta_t_pred_c": lambda out: out["eta_t_pred_c"],
            "eta_dot_pred_c": lambda out: out["eta_dot_pred_c"],
            "lift_pred_c": lambda out: out["lift_pred_c"],
        },
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # initialize solver
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator_pde,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    mat = scipy.io.loadmat(cfg.data_file)
    ag_data = mat["input_tf"]  # ag, ad, av
    u_data = mat["target_X_tf"]
    ut_data = mat["target_Xd_tf"]
    utt_data = mat["target_Xdd_tf"]
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

    t = mat["time"]
    dt = t[0, 1] - t[0, 0]

    ag_all = ag_data
    u_all = u_data
    u_t_all = ut_data
    u_tt_all = utt_data

    # finite difference
    N = u_data.shape[1]
    phi1 = np.concatenate(
        [
            np.array([-3 / 2, 2, -1 / 2]),
            np.zeros([N - 3]),
        ]
    )
    temp1 = np.concatenate([-1 / 2 * np.identity(N - 2), np.zeros([N - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([N - 2, 2]), 1 / 2 * np.identity(N - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate(
        [
            np.zeros([N - 3]),
            np.array([1 / 2, -2, 3 / 2]),
        ]
    )
    phi_t0 = (
        1
        / dt
        * np.concatenate(
            [
                np.reshape(phi1, [1, phi1.shape[0]]),
                phi2,
                np.reshape(phi3, [1, phi3.shape[0]]),
            ],
            axis=0,
        )
    )
    phi_t0 = np.reshape(phi_t0, [1, N, N])

    ag_star = ag_all[0:10]
    eta_star = u_all[0:10]
    eta_t_star = u_t_all[0:10]
    eta_tt_star = u_tt_all[0:10]
    ag_c_star = ag_all[0:50]
    lift_star = -ag_c_star

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    ag_c = ag_c_star
    g = -eta_tt - ag
    phi_t = np.repeat(phi_t0, ag_c_star.shape[0], axis=0)

    model = ppsci.arch.DeepPhyLSTM(
        cfg.MODEL.phylstm2_net.input_size,
        eta.shape[2],
        cfg.MODEL.phylstm2_net.hidden_size,
        cfg.MODEL.phylstm2_net.model_type,
    )
    model.register_input_transform(functions.transform_in)
    model.register_output_transform(functions.transform_out)

    dataset_obj = functions.Dataset(eta, eta_t, g, ag, ag_c, lift, phi_t)
    (
        _,
        _,
        input_dict_val,
        label_dict_val,
    ) = dataset_obj.get(1)

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func2),
        {
            "eta_pred": lambda out: out["eta_pred"],
            "eta_dot_pred": lambda out: out["eta_dot_pred"],
            "g_pred": lambda out: out["g_pred"],
            "eta_t_pred_c": lambda out: out["eta_t_pred_c"],
            "eta_dot_pred_c": lambda out: out["eta_dot_pred_c"],
            "lift_pred_c": lambda out: out["lift_pred_c"],
        },
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        validator=validator_pde,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate after finished training
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="phylstm2.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
