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
high-order finite difference solver for 2d Burgers equation
spatial diff: 4th order laplacian
temporal diff: O(dt^5) due to RK4
"""

import os

import functions
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import ppsci

ppsci.utils.misc.set_random_seed(5)


def apply_laplacian(mat, dx=1.0):
    # dx is inversely proportional to N
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix

    For more information see
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -5 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (4 / 3, (-1, 0)),
        (4 / 3, (0, -1)),
        (4 / 3, (0, 1)),
        (4 / 3, (1, 0)),
        (-1 / 12, (-2, 0)),
        (-1 / 12, (0, -2)),
        (-1 / 12, (0, 2)),
        (-1 / 12, (2, 0)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx**2


def apply_dx(mat, dx=1.0):
    """central diff for dx"""

    # np.roll, axis=0 -> row
    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (2, 0)),
        (-8.0 / 12, (1, 0)),
        (8.0 / 12, (-1, 0)),
        (-1.0 / 12, (-2, 0)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx


def apply_dy(mat, dy=1.0):
    """central diff for dy"""

    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (0, 2)),
        (-8.0 / 12, (0, 1)),
        (8.0 / 12, (0, -1)),
        (-1.0 / 12, (0, -2)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dy


def get_temporal_diff(U, V, R, dx):
    # u and v in (h, w)
    laplace_u = apply_laplacian(U, dx)
    laplace_v = apply_laplacian(V, dx)

    u_x = apply_dx(U, dx)
    v_x = apply_dx(V, dx)

    u_y = apply_dy(U, dx)
    v_y = apply_dy(V, dx)

    # governing equation
    u_t = (1.0 / R) * laplace_u - U * u_x - V * u_y
    v_t = (1.0 / R) * laplace_v - U * v_x - V * v_y

    return u_t, v_t


def update_rk4(U0, V0, R=100, dt=0.05, dx=1.0):
    """Update with Runge-kutta-4 method
    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update

    u_t, v_t = get_temporal_diff(U0, V0, R, dx)

    K1_u = u_t
    K1_v = v_t

    ############# Stage 2 ##############
    U1 = U0 + K1_u * dt / 2.0
    V1 = V0 + K1_v * dt / 2.0

    u_t, v_t = get_temporal_diff(U1, V1, R, dx)

    K2_u = u_t
    K2_v = v_t

    ############# Stage 3 ##############
    U2 = U0 + K2_u * dt / 2.0
    V2 = V0 + K2_v * dt / 2.0

    u_t, v_t = get_temporal_diff(U2, V2, R, dx)

    K3_u = u_t
    K3_v = v_t

    ############# Stage 4 ##############
    U3 = U0 + K3_u * dt
    V3 = V0 + K3_v * dt

    u_t, v_t = get_temporal_diff(U3, V3, R, dx)

    K4_u = u_t
    K4_v = v_t

    # Final solution
    U = U0 + dt * (K1_u + 2 * K2_u + 2 * K3_u + K4_u) / 6.0
    V = V0 + dt * (K1_v + 2 * K2_v + 2 * K3_v + K4_v) / 6.0

    return U, V


def postProcess(output, reso, xmin, xmax, ymin, ymax, num, fig_save_dir):
    """num: Number of time step"""
    x = np.linspace(0, reso, reso + 1)
    y = np.linspace(0, reso, reso + 1)
    x_star, y_star = np.meshgrid(x, y)
    x_star, y_star = x_star[:-1, :-1], y_star[:-1, :-1]

    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(
        x_star,
        y_star,
        c=u_pred,
        alpha=0.95,
        edgecolors="none",
        cmap="RdYlBu",
        marker="s",
        s=3,
    )
    ax[0].axis("square")
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    cf.cmap.set_under("black")
    cf.cmap.set_over("whitesmoke")
    ax[0].set_title("u-FDM")
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(
        x_star,
        y_star,
        c=v_pred,
        alpha=0.95,
        edgecolors="none",
        cmap="RdYlBu",
        marker="s",
        s=3,
    )
    ax[1].axis("square")
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    cf.cmap.set_under("black")
    cf.cmap.set_over("whitesmoke")
    ax[1].set_title("v-FDM")
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    plt.savefig(fig_save_dir + "/uv_[i=%d].png" % (num))
    plt.close("all")


if __name__ == "__main__":
    # grid size
    M, N = 128, 128
    n_simu_steps = 30000
    dt = 0.0001  # maximum 0.003
    dx = 1.0 / M
    R = 200.0

    # get initial condition from random field
    GRF = functions.GaussianRF(2, M, alpha=2, tau=5)
    U, V = GRF.sample(2)  # U and V have shape of [128, 128]
    U = U.cpu().numpy()
    V = V.cpu().numpy()

    U_record = U.copy()[None, ...]
    V_record = V.copy()[None, ...]

    for step in range(n_simu_steps):

        U, V = update_rk4(U, V, R, dt, dx)  # [h, w]

        if (step + 1) % 20 == 0:
            print(step)
            U_record = np.concatenate((U_record, U[None, ...]), axis=0)  # [t,h,w]
            V_record = np.concatenate((V_record, V[None, ...]), axis=0)

    UV = np.concatenate((U_record[None, ...], V_record[None, ...]), axis=0)  # (c,t,h,w)
    UV = np.transpose(UV, [1, 0, 2, 3])  # (t,c,h,w)

    fig_save_dir = "./output/figures/2dBurgers/"
    os.makedirs(fig_save_dir, exist_ok=True)
    for i in range(0, 30):
        postProcess(UV, M, 0, M, 0, M, 50 * i, fig_save_dir)

    # save data
    data_save_dir = "./output/"
    os.makedirs(data_save_dir, exist_ok=True)
    scipy.io.savemat(
        os.path.join(data_save_dir, "burgers_1501x2x128x128.mat"), {"uv": UV}
    )

# [umin, umax] = [-0.7, 0.7]
# [vmin, vmax] = [-1.0, 1.0]
