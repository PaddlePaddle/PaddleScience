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

"""
Reference: https://github.com/hanfengzhai/BubbleNet
Bubble data files download link: https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat
"""

import numpy as np
import scipy.io

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger

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
    # normalize data
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
    p_star = p_norm  # N x T
    phil_star = data["phil"]  # N x T
    t_star = data["t"]  # T x 1
    x_star = data["X"]  # N x 2

    N = x_star.shape[0]
    T = t_star.shape[0]

    # rearrange data
    xx = np.tile(x_star[:, 0:1], (1, T))  # N x T
    yy = np.tile(x_star[:, 1:2], (1, T))  # N x T
    tt = np.tile(t_star, (1, N)).T  # N x T

    x = xx.flatten()[:, None].astype("float32")  # NT x 1
    y = yy.flatten()[:, None].astype("float32")  # NT x 1
    t = tt.flatten()[:, None].astype("float32")  # NT x 1

    u = u_star.flatten()[:, None].astype("float32")  # NT x 1
    v = v_star.flatten()[:, None].astype("float32")  # NT x 1
    p = p_star.flatten()[:, None].astype("float32")  # NT x 1
    phil = phil_star.flatten()[:, None].astype("float32")  # NT x 1

    idx = np.random.choice(N * T, int(N * T * 0.75), replace=False)
    # train data
    train_input = {"x": x[idx, :], "y": y[idx, :], "t": t[idx, :]}
    train_label = {"u": u[idx, :], "v": v[idx, :], "p": p[idx, :], "phil": phil[idx, :]}

    # eval data
    test_input = {"x": x, "y": y, "t": t}
    test_label = {"u": u, "v": v, "p": p, "phil": phil}

    # set model
    model_psi = ppsci.arch.MLP(("t", "x", "y"), ("psi",), 9, 30, "tanh")
    model_p = ppsci.arch.MLP(("t", "x", "y"), ("p",), 9, 30, "tanh")
    model_phil = ppsci.arch.MLP(("t", "x", "y"), ("phil",), 9, 30, "tanh")

    # transform
    def transform_out(in_, out):
        psi_y = out["psi"]
        y = in_["y"]
        x = in_["x"]
        u = jacobian(psi_y, y)
        v = -jacobian(psi_y, x)
        return {"u": u, "v": v}

    # register transform
    model_psi.register_output_transform(transform_out)
    model_list = ppsci.arch.ModelList((model_psi, model_p, model_phil))

    # set time-geometry
    # set timestamps(including initial t0)
    timestamps = np.linspace(0, 126, 127, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.PointCloud(
            train_input,
            ("t", "x", "y"),
        ),
        "time_rect_visu": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(1, 126, timestamps=timestamps),
            ppsci.geometry.Rectangle((0, 0), (15, 5)),
        ),
    }

    # set dataloader config
    ITERS_PER_EPOCH = 1
    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": train_input,
            "label": train_label,
            "weight": {},
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
            "batch_size": 228595,
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
    EPOCHS = 10000 if not args.epochs else args.epochs
    EVAL_FREQ = 1000
    # set optimizer
    optimizer = ppsci.optimizer.Adam(0.001)((model_list,))

    # set validator
    valida_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": test_input,
            "label": test_label,
            "weight": {},
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
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        geom=geom,
        validator=validator,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()

    # visualize prediction after finished training
    visu_mat = geom["time_rect_visu"].sample_interior(
        NPOINT_PDE * NTIME_PDE, evenly=True
    )

    pred_norm = solver.predict(visu_mat, no_grad=False)
    # inverse normalization
    p_pred = pred_norm["p"].reshape([NTIME_PDE, NPOINT_PDE]).T
    u_pred = pred_norm["u"].reshape([NTIME_PDE, NPOINT_PDE]).T
    v_pred = pred_norm["v"].reshape([NTIME_PDE, NPOINT_PDE]).T
    pred = dict()
    pred["p"] = (p_pred * (p_max - p_min) + p_min).T.reshape([-1, 1]).numpy()
    pred["u"] = (u_pred * (u_max - u_min) + u_min).T.reshape([-1, 1]).numpy()
    pred["v"] = (v_pred * (v_max - v_min) + v_min).T.reshape([-1, 1]).numpy()
    pred["phil"] = pred_norm["phil"].numpy()
    ppsci.visualize.save_vtu_from_dict(
        "{OUTPUT_DIR}/visual/result.vtu",
        {
            "t": visu_mat["t"],
            "x": visu_mat["x"],
            "y": visu_mat["y"],
            "u": pred["u"],
            "v": pred["v"],
            "p": pred["p"],
            "phil": pred["phil"],
        },
        ("t", "x", "y"),
        ("u", "v", "p", "phil"),
        NTIME_PDE,
    )
