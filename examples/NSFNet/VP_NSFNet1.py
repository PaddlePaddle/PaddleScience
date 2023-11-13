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
Reference: https://github.com/Alexzihaohu/NSFnets/tree/master

Corresponding AIstudio: https://aistudio.baidu.com/studio/project/partial/verify/6832363/c3b2af9c2b8243548daa037ee4342f2e
"""

import ppsci
import paddle

paddle.set_default_dtype("float32")
import numpy as np
from ppsci.utils import logger
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet1.yaml")
def main(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # Set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)

    ITERS_PER_EPOCH = cfg.iters_per_epoch
    # set model
    input_key = ("x", "y")
    output_key = ("u", "v", "p")
    model = ppsci.arch.MLP(
        input_key, output_key, cfg.model.ihlayers, cfg.model.ineurons, "tanh", input_dim=len(input_key), output_dim=len(output_key), Xavier=True
    )

    ## set the number of residual samples
    N_TRAIN = cfg.ntrain

    ## set the number of boundary samples
    Nb_TRAIN = cfg.nb_train

    # generate data

    # set the Reynolds number and the corresponding lambda which is the parameter in the exact solution.
    Re = cfg.re
    lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

    x = np.linspace(-0.5, 1.0, 101)
    y = np.linspace(-0.5, 1.5, 101)

    yb1 = np.array([-0.5] * 100)
    yb2 = np.array([1] * 100)
    xb1 = np.array([-0.5] * 100)
    xb2 = np.array([1.5] * 100)

    y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0).astype('float32')
    x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0).astype('float32')

    xb_train = x_train1.reshape(x_train1.shape[0], 1).astype('float32')
    yb_train = y_train1.reshape(y_train1.shape[0], 1).astype('float32')
    ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
    vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)

    x_train = ((np.random.rand(N_TRAIN, 1) - 1 / 3) * 3 / 2)
    y_train = ((np.random.rand(N_TRAIN, 1) - 1 / 4) * 2)

    # generate test data
    np.random.seed(SEED)
    x_star = ((np.random.rand(1000, 1) - 1 / 3) * 3 / 2).astype('float32')
    y_star = ((np.random.rand(1000, 1) - 1 / 4) * 2).astype('float32')

    u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
    v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
    p_star = 0.5 * (1 - np.exp(2 * lam * x_star))

    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": xb_train, "y": yb_train},
            "label": {"u": ub_train, "v": vb_train}
        },
        "batch_size": Nb_TRAIN,
        'iters_per_epoch': ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    valida_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x_star, "y": y_star},
            "label": {"u": u_star, "v": v_star, "p": p_star}},
        'total_size': u_star.shape[0],
        "batch_size": u_star.shape[0],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    geom = ppsci.geometry.PointCloud({"x": x_train, "y": y_train}, ("x", "y"))

    # supervised constraint s.t ||u-u_0||
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="Sup",
    )

    # set equation constarint s.t. ||F(u)||

    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(nu=1.0 / Re, rho=1.0, dim=2, time=False),
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom,
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": N_TRAIN,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    constraint = {sup_constraint.name: sup_constraint, pde_constraint.name: pde_constraint}

    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # wrap validator
    validator = {residual_validator.name: residual_validator}

    # set learning rate scheduler
    epoch_list = cfg.epoch_list
    new_epoch_list = []
    for i, _ in enumerate(epoch_list):
        new_epoch_list.append(sum(epoch_list[:i+1]))
    EPOCHS = new_epoch_list[-1]
    lr_list = cfg.lr_list

    lr_scheduler = ppsci.optimizer.lr_scheduler.Piecewise(EPOCHS, ITERS_PER_EPOCH, new_epoch_list, lr_list)()

    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    
    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        optimizer=optimizer,
        epochs=EPOCHS,
        lr_scheduler=lr_scheduler,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        log_freq=cfg.log_freq,
        eval_freq=cfg.eval_freq,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=None,
        eval_with_no_grad=False,
        output_dir=OUTPUT_DIR
    )
    # train model
    solver.train()
    
    solver.eval()

    # plot the loss
    solver.plot_loss_history()

    # set optimizer
    EPOCHS = 5000
    optimizer = ppsci.optimizer.LBFGS(max_iter=50000, tolerance_change=np.finfo(float).eps, history_size=50)(model)

    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    
    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        optimizer=optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        log_freq=2000,
        eval_freq=2000,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=None,
        eval_with_no_grad=False,
        output_dir=OUTPUT_DIR
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()
    
if __name__ == "__main__":
    main()
