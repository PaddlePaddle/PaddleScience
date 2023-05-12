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
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_viv" if args.output_dir is None else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("t_f",), ("eta",), 5, 50, "tanh", False, False)

    # set equation
    equation = {"VIV": ppsci.equation.Vibration(2, -4, 0)}

    # set dataloader config
    ITERS_PER_EPOCH = 1
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
        name="Sup",
    )
    # wrap constraints together
    constraint = {
        sup_constraint.name: sup_constraint,
    }

    # set training hyper-parameters
    EPOCHS = 100000 if args.epochs is None else args.epochs
    EVAL_FREQ = 1000

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(
        EPOCHS, ITERS_PER_EPOCH, 0.001, step_size=20000, gamma=0.9
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,) + tuple(equation.values()))

    # set validator
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
        "./VIV_Training_Neta100.mat",
        ("t_f", "eta_gt", "f_gt"),
        alias_dict={"eta_gt": "eta", "f_gt": "f"},
    )
    visualizer = {
        "visulzie_u": ppsci.visualize.VisualizerScatter1D(
            visu_mat,
            ("t_f",),
            {
                r"$\eta$": lambda d: d["eta"],  # plot with latex title
                r"$\eta_{gt}$": lambda d: d["eta_gt"],  # plot with latex title
                r"$f$": equation["VIV"].equations["f"],  # plot with latex title
                r"$f_{gt}$": lambda d: d["f_gt"],  # plot with latex title
            },
            num_timestamps=1,
            prefix="viv_pred",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=EVAL_FREQ,
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
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction from pretrained_model_path(optional)
    solver.visualize()
