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

import numpy as np
import scipy.io

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import reader

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_bubble_pinns_p" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    DATASET_PATH = "bubble_train.mat"
    DATASET_PATH_VALID = "bubble_test.mat"

    # set model
    model_psi = ppsci.arch.MLP(("x", "y", "t"), ("psi",), 9, 30, "tanh")
    model_p = ppsci.arch.MLP(("x", "y", "t"), ("p",), 9, 30, "tanh")
    model_phil = ppsci.arch.MLP(("x", "y", "t"), ("phil",), 9, 30, "tanh")

    def transform_out(in_, out):
        psi_y = out["psi"]
        y = in_["y"]
        x = in_["x"]
        u_out = jacobian(psi_y, y)
        v_out = -jacobian(psi_y, x)
        return {"u": u_out, "v": v_out}

    model_psi.register_output_transform(transform_out)
    model_list = ppsci.arch.ModelList((model_psi, model_p, model_phil))

    # set time-geometry
    # set timestamps(including initial t0)
    timestamps = np.linspace(0, 126, 127, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.PointCloud(
            reader.load_mat_file(
                DATASET_PATH,
                ("t", "x", "y"),
            ),
            ("t", "x", "y"),
        ),
        "time_rect_eval": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(1, 126, timestamps=timestamps),
            ppsci.geometry.Rectangle((0, 0), (15, 5)),
        ),
    }

    # set dataloader config
    ITERS_PER_EPOCH = 1
    train_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("x", "t", "y"),
            "label_keys": ("u", "v", "p", "phil"),
            "timestamps": timestamps,
        },
        "batch_size": 2419,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }

    NTIME_ALL = len(timestamps)
    NPOINT_PDE, NTIME_PDE = 300 * 100, NTIME_ALL - 1

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        {
            "pressure_Poisson": lambda out: hessian(out["p"], out["x"])
            + hessian(out["p"], out["y"])
        },
        {"pressure_Poisson": 0},
        geom["time_rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="Sup",
    )

    # wrap constraints together
    constraint = {
        sup_constraint.name: sup_constraint,
        pde_constraint.name: pde_constraint,
    }

    # set training hyper-parameters
    EPOCHS = 10000
    EVAL_FREQ = 1000
    # set optimizer
    optimizer = ppsci.optimizer.Adam(0.001)(model_list)

    # set validator
    valida_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": DATASET_PATH_VALID,
            "input_keys": ("t", "x", "y"),
            "label_keys": ("u", "v", "p", "phil"),
        },
        "batch_size": 2419,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    mse_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="bubble_mse",
    )
    validator = {
        mse_validator.name: mse_validator,
    }

    visu_mat = geom["time_rect_eval"].sample_interior(
        NPOINT_PDE * NTIME_PDE, evenly=True
    )

    datafile = "maxmin.mat"
    data = scipy.io.loadmat(datafile)
    u_max = data["u_max"][0][0]
    u_min = data["u_min"][0][0]
    v_max = data["v_max"][0][0]
    v_min = data["v_min"][0][0]
    p_max = data["p_max"][0][0]
    p_min = data["p_min"][0][0]
    phil_max = data["phil_max"][0][0]
    phil_min = data["phil_min"][0][0]
    visualizer = {
        "visulzie_u_v_p": ppsci.visualize.VisualizerVtu(
            visu_mat,
            {
                "u": lambda d: d["u"] * (u_max - u_min) + u_min,
                "v": lambda d: d["v"] * (v_max - v_min) + v_min,
                "p": lambda d: d["p"] * (p_max - p_min) + p_min,
                "phil": lambda d: d["phil"],
            },
            num_timestamps=NTIME_PDE,
            prefix="result_u_v_p",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=EVAL_FREQ,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()
