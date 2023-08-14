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
    OUTPUT_DIR = "./output_bubble" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # load Data
    data = scipy.io.loadmat("bubble.mat")
    # Normalize data
    p_max = data["p"].max(axis=0)
    p_min = data["p"].min(axis=0)
    p_norm = (data["p"] - p_min) / (p_max - p_min)
    u_max = data["u"].max(axis=0)
    u_min = data["u"].min(axis=0)
    u_norm = (data["u"] - u_min) / (u_max - u_min)
    v_max = data["v"].max(axis=0)
    v_min = data["v"].min(axis=0)
    v_norm = (data["v"] - v_min) / (v_max - v_min)

    u_star = u_norm  # N x T
    v_star = v_norm  # N x T
    P_star = p_norm  # N x T
    Phil_star = data["phil"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X"]  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = u_star  # U_star[:,0,:] # N x T
    VV = v_star  # U_star[:,1,:] # N x T
    PP = P_star  # N x T
    Phil = Phil_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    phil = Phil.flatten()[:, None]  # NT x 1

    idx = np.random.choice(N * T, int(N * T * 0.75), replace=False)
    # train data
    train_data = dict()
    train_data["x"] = x[idx, :]
    train_data["y"] = y[idx, :]
    train_data["t"] = t[idx, :]
    train_data["u"] = u[idx, :]
    train_data["v"] = v[idx, :]
    train_data["p"] = p[idx, :]
    train_data["phil"] = phil[idx, :]
    scipy.io.savemat("bubble_train.mat", train_data)
    # valid Data
    test_data = dict()
    test_data["x"] = x
    test_data["y"] = y
    test_data["t"] = t
    test_data["u"] = u
    test_data["v"] = v
    test_data["p"] = p
    test_data["phil"] = phil
    scipy.io.savemat("bubble_valid.mat", test_data)

    DATASET_PATH = "bubble_train.mat"
    DATASET_PATH_VALID = "bubble_valid.mat"

    # set model
    model_psi = ppsci.arch.MLP(("t", "x", "y"), ("psi",), 9, 30, "tanh")
    model_p = ppsci.arch.MLP(("t", "x", "y"), ("p",), 9, 30, "tanh")
    model_phil = ppsci.arch.MLP(("t", "x", "y"), ("phil",), 9, 30, "tanh")

    def transform_out(in_, out):
        psi_y = out["psi"]
        y = in_["y"]
        x = in_["x"]
        u = jacobian(psi_y, y)
        v = -jacobian(psi_y, x)
        return {"u": u, "v": v}

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
            "input_keys": ("t", "x", "y"),
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
