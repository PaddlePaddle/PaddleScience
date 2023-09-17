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

import numpy as np
import scipy.io
import util

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output/phylstm3" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # set training hyper-parameters
    EPOCHS = 200 if not args.epochs else args.epochs

    mat = scipy.io.loadmat("data_boucwen.mat")

    t = mat["time"]
    dt = 0.02
    n1 = int(dt / 0.005)
    t = t[::n1]

    ag_data = mat["input_tf"][:, ::n1]  # ag, ad, av
    u_data = mat["target_X_tf"][:, ::n1]
    ut_data = mat["target_Xd_tf"][:, ::n1]
    utt_data = mat["target_Xdd_tf"][:, ::n1]
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

    ag_pred = mat["input_pred_tf"][:, ::n1]  # ag, ad, av
    u_pred = mat["target_pred_X_tf"][:, ::n1]
    ut_pred = mat["target_pred_Xd_tf"][:, ::n1]
    utt_pred = mat["target_pred_Xdd_tf"][:, ::n1]
    ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
    u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
    ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
    utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])

    n = u_data.shape[1]
    phi1 = np.concatenate(
        [
            np.array([-3 / 2, 2, -1 / 2]),
            np.zeros(
                [
                    n - 3,
                ]
            ),
        ]
    )
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate(
        [
            np.zeros(
                [
                    n - 3,
                ]
            ),
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
    phi_t0 = np.reshape(phi_t0, [1, n, n])

    ag_star = ag_data
    eta_star = u_data
    eta_t_star = ut_data
    eta_tt_star = utt_data
    ag_c_star = np.concatenate([ag_data, ag_pred[0:53]])
    lift_star = -ag_c_star
    eta_c_star = np.concatenate([u_data, u_pred[0:53]])
    eta_t_c_star = np.concatenate([ut_data, ut_pred[0:53]])
    eta_tt_c_star = np.concatenate([utt_data, utt_pred[0:53]])

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    g = -eta_tt - ag
    ag_c = ag_c_star

    # Training Data
    eta_train = eta
    ag_train = ag
    lift_train = lift
    eta_t_train = eta_t
    eta_tt_train = eta_tt
    g_train = g
    ag_c_train = ag_c

    loss_bfgs = np.empty([0])
    loss_val_bfgs = np.empty([0])
    phi_t = np.repeat(phi_t0, ag_c_star.shape[0], axis=0)

    dataset_obj = util.Dataset(eta, eta_t, g, ag, ag_c, lift, phi_t)

    input_dict, label_dict, input_dict_val, label_dict_val = dataset_obj.get(EPOCHS)

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict,
                "label": label_dict,
            },
        },
        ppsci.loss.FunctionalLoss(util.pde_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        name="loss",
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
        ppsci.loss.FunctionalLoss(util.pde_loss_val_func),
        {
            "loss": lambda out: out["loss"].detach().clone(),
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="loss_val",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # initialize solver
    ITERS_PER_EPOCH = 1
    SAVE_FREQ = 1000
    model = ppsci.arch.DeepPhyLSTM3(eta.shape[2])
    opt = ppsci.optimizer.Adam(1e-3)(model)
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        OUTPUT_DIR,
        opt,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=SAVE_FREQ,
        eval_with_no_grad=True,
        validator=validator_pde,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
