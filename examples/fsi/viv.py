# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ppsci
from ppsci.utils import logger

if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    output_dir = "./output_viv_refactor"
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("t_f",), ("eta",), 5, 50, "tanh", False, False)
    # set equation
    equation = {"VIV": ppsci.equation.Vibration(2, -4, 0)}

    # set dataloader config
    iters_per_epoch = 1
    train_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": "./VIV_Training_Neta100.mat",
            "input_keys": ("t_f",),
            "label_keys": ("eta", "f"),
            "weight_dict": {"eta": 100},
        },
        "batch_size": 150,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }
    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        name="EQ",
    )
    # wrap constraints together
    constraint = {
        sup_constraint.name: sup_constraint,
    }

    # set training hyper-parameters
    epochs = 100000
    eval_freq = 1000

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(
        epochs, iters_per_epoch, 0.001, step_size=20000, gamma=0.9
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)([model] + list(equation.values()))

    valida_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": "./VIV_Training_Neta100.mat",
            "input_keys": ("t_f",),
            "label_keys": ("eta", "f"),
        },
        "batch_size": 32,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    # set validator
    eta_mse_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        metric={"MSE": ppsci.metric.MSE()},
        name="eta_mse",
    )
    validator = {eta_mse_validator.name: eta_mse_validator}

    # set visualizer(optional)
    visu_mat = ppsci.utils.reader.load_mat_file(
        "./VIV_Training_Neta100.mat", ("t_f", "eta_gt"), alias_dict={"eta_gt": "eta"}
    )
    visualizer = {
        "visulzie_u": ppsci.visualize.VisualizerScatter1D(
            visu_mat,
            ("t_f",),
            {"eta_pred": lambda d: d["eta"], "eta_gt": lambda d: d["eta_gt"]},
            1,
            "viv_pred",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        optimizer,
        lr_scheduler,
        epochs,
        iters_per_epoch,
        eval_during_train=True,
        eval_freq=eval_freq,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    # # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate model from pretrained_model_path(optional)
    logger.init_logger("ppsci", f"{output_dir}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"./{output_dir}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction from pretrained_model_path(optional)
    solver.visualize()
