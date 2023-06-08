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
This module is heavily adapted from https://github.com/lululxvi/hpinn
"""

import os

import functions as f
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker

import ppsci

"""All plotting functions."""

# define constants
font = {"weight": "normal", "size": 10}
input_name = ("x", "y")
field_name = [
    "Fig7_E",
    "Fig7_eps",
    "Fig_6C_lambda_re_1",
    "Fig_6C_lambda_im_1",
    "Fig_6C_lambda_re_4",
    "Fig_6C_lambda_im_4",
    "Fig_6C_lambda_re_9",
    "Fig_6C_lambda_im_9",
]

# define constants which will be assigned later
FIGNAME = ""  # type: str
OUTPUT_DIR = ""  # type: str
DATASET_PATH = ""  # type: str
DATASET_PATH_VALID = ""  # type: str
input_valid = None  # type: np.ndarray
output_valid = None  # type: np.ndarray
input_train = None  # type: np.ndarray


def set_params(figname, output_dir, dataset_path, dataset_path_valid):
    global FIGNAME, OUTPUT_DIR, DATASET_PATH, DATASET_PATH_VALID
    FIGNAME = figname
    OUTPUT_DIR = output_dir + "figure/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DATASET_PATH = dataset_path
    DATASET_PATH_VALID = dataset_path_valid


def prepare_data(solver, expr_dict):
    global input_valid, output_valid, input_train
    # train data
    train_dict = ppsci.utils.reader.load_mat_file(DATASET_PATH, ("x", "y", "bound"))

    bound = int(train_dict["bound"])
    x_train = train_dict["x"][bound:]
    y_train = train_dict["y"][bound:]
    input_train = np.stack((x_train, y_train), axis=-1).reshape(-1, 2)

    # valid data
    N = ((f.l_BOX[1] - f.l_BOX[0]) / 0.05).astype(int)

    valid_dict = ppsci.utils.reader.load_mat_file(
        DATASET_PATH_VALID, ("x_val", "y_val", "bound")
    )
    in_dict_val = {"x": valid_dict["x_val"], "y": valid_dict["y_val"]}
    f.init_lambda(in_dict_val, int(valid_dict["bound"]))

    pred_dict_val = solver.predict(
        in_dict_val,
        expr_dict,
        batch_size=np.shape(valid_dict["x_val"])[0],
        no_grad=False,
    )

    input_valid = np.stack((valid_dict["x_val"], valid_dict["y_val"]), axis=-1).reshape(
        N[0], N[1], 2
    )
    output_valid = np.array(
        [
            pred_dict_val["e_real"].numpy(),
            pred_dict_val["e_imaginary"].numpy(),
            pred_dict_val["epsilon"].numpy(),
        ]
    ).T.reshape(N[0], N[1], 3)


def plot_field_horo(coord_visual, field_visual, coord_lambda, field_lambda, title=None):
    fmin, fmax = np.array([0, 1.0]), np.array([0.6, 12])
    cmin, cmax = coord_visual.min(axis=(0, 1)), coord_visual.max(axis=(0, 1))
    emin, emax = np.array([-3, -1]), np.array([3, 0])
    x_pos = coord_visual[:, :, 0]
    y_pos = coord_visual[:, :, 1]

    for fi in range(len(field_name)):
        if fi == 0:
            # Fig7_E
            plt.figure(101, figsize=(8, 6))
            plt.clf()
            plt.rcParams["font.size"] = 20
            f_true = field_visual[..., fi]
            plt.pcolormesh(
                x_pos,
                y_pos,
                f_true,
                cmap="rainbow",
                shading="gouraud",
                antialiased=True,
                snap=True,
            )
            cb = plt.colorbar()
            plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
        elif fi == 1:
            # Fig7_eps
            plt.figure(201, figsize=(8, 1.5))
            plt.clf()
            plt.rcParams["font.size"] = 20
            f_true = field_visual[..., fi]
            plt.pcolormesh(
                x_pos,
                y_pos,
                f_true,
                cmap="rainbow",
                shading="gouraud",
                antialiased=True,
                snap=True,
            )
            cb = plt.colorbar()
            plt.axis((emin[0], emax[0], emin[1], emax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
        else:
            # Fig_6C_lambda_
            plt.figure(fi * 100 + 101, figsize=(8, 6))
            plt.clf()
            plt.rcParams["font.size"] = 20
            f_true = field_lambda[..., fi - 2]
            plt.scatter(
                coord_lambda[..., 0],
                coord_lambda[..., 1],
                c=f_true,
                cmap="rainbow",
                alpha=0.6,
            )
            cb = plt.colorbar()
            plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))

        # colorbar settings
        cb.ax.tick_params(labelsize=20)
        tick_locator = ticker.MaxNLocator(
            nbins=5
        )  # the number of scale values ​​on the colorbar
        cb.locator = tick_locator
        cb.update_ticks()

        plt.xlabel(f"${str(input_name[0])}$", fontdict=font)
        plt.ylabel(f"${str(input_name[1])}$", fontdict=font)
        plt.yticks(size=10)
        plt.xticks(size=10)
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{FIGNAME}_{str(field_name[fi])}.jpg",
            )
        )


def plot_6a(log_loss):
    # plot Fig.6 A of paper
    plt.figure(300, figsize=(8, 6))
    smooth_step = 100  # how many steps of loss are squeezed to one point, num_points is epoch/smooth_step
    if log_loss.shape[0] % smooth_step is not 0:
        vis_loss_ = log_loss[: -(log_loss.shape[0] % smooth_step), :].reshape(
            -1, smooth_step, log_loss.shape[1]
        )
    else:
        vis_loss_ = log_loss.reshape(-1, smooth_step, log_loss.shape[1])

    vis_loss = vis_loss_.mean(axis=1).reshape(-1, 3)
    vis_loss_total = vis_loss[:, :].sum(axis=1)
    vis_loss[:, 1] = vis_loss[:, 2]
    vis_loss[:, 2] = vis_loss_total
    for i in range(vis_loss.shape[1]):
        plt.semilogy(np.arange(vis_loss.shape[0]) * smooth_step, vis_loss[:, i])
    plt.legend(
        ["PDE loss", "Objective loss", "Total loss"],
        loc="lower left",
        prop=font,
    )
    plt.xlabel("Iteration ", fontdict=font)
    plt.ylabel("Loss ", fontdict=font)
    plt.grid()
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{FIGNAME}_Fig6_A.jpg"))


def plot_6b(log_loss_obj):
    # plot Fig.6 B of paper
    plt.figure(400, figsize=(10, 6))
    plt.clf()
    plt.plot(np.arange(len(log_loss_obj)), log_loss_obj, "bo-")
    plt.xlabel("k", fontdict=font)
    plt.ylabel("Objective", fontdict=font)
    plt.grid()
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{FIGNAME}_Fig6_B.jpg"))


def plot_6c7c(log_lambda):
    # plot Fig.6 Cs and Fig.7.Cs of paper
    global input_valid, output_valid, input_train

    field_lambda = np.concatenate(
        [log_lambda[1], log_lambda[4], log_lambda[9]], axis=0
    ).T
    v_visual = output_valid[..., 0] ** 2 + output_valid[..., 1] ** 2
    field_visual = np.stack((v_visual, output_valid[..., -1]), axis=-1)
    plot_field_horo(input_valid, field_visual, input_train, field_lambda)


def plot_6d(log_lambda):
    # plot Fig.6 D of paper
    # lambda/mu
    mu_ = 2 ** np.arange(1, 11)
    log_lambda = np.array(log_lambda) / mu_[:, None, None]
    # randomly pick 3 lambda points to represent all points of each k
    ind = np.random.randint(low=0, high=np.shape(log_lambda)[-1], size=3)
    la_mu_ind = log_lambda[:, :, ind]
    marker = ["ro-", "bo:", "r*-", "b*:", "rp-", "bp:"]
    plt.figure(500, figsize=(7, 5))
    plt.clf()
    for i in range(6):
        plt.plot(
            np.arange(0, 10),
            la_mu_ind[:, int(i % 2), int(i / 2)],
            marker[i],
            linewidth=2,
        )
    plt.legend(
        ["Re, 1", "Im, 1", "Re, 2", "Im, 2", "Re, 3", "Im, 3"],
        loc="upper right",
        prop=font,
    )
    plt.grid()
    plt.xlabel("k", fontdict=font)
    plt.ylabel(r"$ \lambda^k / \mu^k_F$", fontdict=font)
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{FIGNAME}_Fig6_D_lambda.jpg"))


def plot_6ef(log_lambda):
    # plot Fig.6 E and Fig.6.F of paper
    # lambda/mu
    mu_ = 2 ** np.arange(1, 11)
    log_lambda = np.array(log_lambda) / mu_[:, None, None]
    # pick k=1,4,6,9
    iter_ind = [1, 4, 6, 9]
    plt.figure(600, figsize=(5, 5))
    plt.clf()
    for i in iter_ind:
        sns.kdeplot(log_lambda[i, 0, :], label="k = " + str(i), cut=0, linewidth=2)
    plt.legend(prop=font)
    plt.grid()
    plt.xlim([-0.1, 0.1])
    plt.xlabel(r"$ \lambda^k_{Re} / \mu^k_F$", fontdict=font)
    plt.ylabel("Frequency", fontdict=font)
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{FIGNAME}_Fig6_E.jpg"))

    plt.figure(700, figsize=(5, 5))
    plt.clf()
    for i in iter_ind:
        sns.kdeplot(log_lambda[i, 1, :], label="k = " + str(i), cut=0, linewidth=2)
    plt.legend(prop=font)
    plt.grid()
    plt.xlim([-0.1, 0.1])
    plt.xlabel(r"$ \lambda^k_{Im} / \mu^k_F$", fontdict=font)
    plt.ylabel("Frequency", fontdict=font)
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{FIGNAME}_Fig6_F.jpg"))
