"""high-order FD solver for FN equation"""
# spatial diff: 4th order laplacian
# temporal diff: O(dt^5) due to RK4

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

np.random.seed(66)


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


def update_rk4(U0, V0, DU=1.0, DV=100.0, alpha=0.01, beta=0.25, dt=0.05, dx=1.0):
    """Update with Runge-kutta-4 method
    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update
    laplace_u = apply_laplacian(U0, dx)
    laplace_v = apply_laplacian(V0, dx)

    u_t = DU * laplace_u + U0 - U0**3 - V0 + alpha
    v_t = DV * laplace_v + (U0 - V0) * beta

    K1_u = u_t
    K1_v = v_t

    ############# Stage 2 ##############
    U1 = U0 + K1_u * dt / 2.0
    V1 = V0 + K1_v * dt / 2.0

    laplace_u = apply_laplacian(U1, dx)
    laplace_v = apply_laplacian(V1, dx)

    u_t = DU * laplace_u + U1 - U1**3 - V1 + alpha
    v_t = DV * laplace_v + (U1 - V1) * beta

    K2_u = u_t
    K2_v = v_t

    ############# Stage 3 ##############
    U2 = U0 + K2_u * dt / 2.0
    V2 = V0 + K2_v * dt / 2.0

    laplace_u = apply_laplacian(U2, dx)
    laplace_v = apply_laplacian(V2, dx)

    u_t = DU * laplace_u + U2 - U2**3 - V2 + alpha
    v_t = DV * laplace_v + (U2 - V2) * beta

    K3_u = u_t
    K3_v = v_t

    ############# Stage 4 ##############
    U3 = U0 + K3_u * dt
    V3 = V0 + K3_v * dt

    laplace_u = apply_laplacian(U3, dx)
    laplace_v = apply_laplacian(V3, dx)

    u_t = DU * laplace_u + U3 - U3**3 - V3 + alpha
    v_t = DV * laplace_v + (U3 - V3) * beta

    K4_u = u_t
    K4_v = v_t

    # Final solution
    U = U0 + dt * (K1_u + 2 * K2_u + 2 * K3_u + K4_u) / 6.0
    V = V0 + dt * (K1_v + 2 * K2_v + 2 * K3_v + K4_v) / 6.0

    return U, V


def get_initial_A_and_B(M, N):
    """get the initial chemical concentrations"""

    A = np.random.normal(scale=0.05, size=(M, N))
    B = np.random.normal(scale=0.05, size=(M, N))

    return A, B


def postProcess(output, xmin, xmax, ymin, ymax, num, fig_save_dir):
    """num: Number of time step"""
    x = np.linspace(0, 128, 129)
    x = x[:-1]
    x_star, y_star = np.meshgrid(x, x)

    u = output[num, 0, :, :]
    v = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(
        x_star,
        y_star,
        c=u,
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
        c=v,
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

    # plt.draw()
    plt.savefig(fig_save_dir + "/uv_[i=%d].png" % (num))
    plt.close("all")


if __name__ == "__main__":
    # Diffusion coefficients
    DA = 1
    DB = 100

    # reaction coeff
    alpha = 0.01
    beta = 0.25

    # grid size
    M = 128
    N = 128
    delta_t = 0.0020  # 0.002 for RK4 = 0.001 for Euler
    dx = 1

    # initialize the chemical concentrations, random_incluence=0
    A, B = get_initial_A_and_B(M, N)
    A_record = A.copy()[None, ...]
    B_record = B.copy()[None, ...]

    N_simulation_steps = 10000
    for step in range(N_simulation_steps):

        # RK4
        A, B = update_rk4(
            A, B, DU=1.0, DV=100.0, alpha=0.01, beta=0.25, dt=delta_t, dx=1.0
        )

        # Save every 0.02s
        if (step + 1) % 10 == 0:
            print(step)
            A_record = np.concatenate((A_record, A[None, ...]), axis=0)
            B_record = np.concatenate((B_record, B[None, ...]), axis=0)

    UV = np.concatenate((A_record[None, ...], B_record[None, ...]), axis=0)
    UV = np.transpose(UV, [1, 0, 2, 3])

    fig_save_dir = "./output/figures/2dFN/"
    for i in range(0, 81):
        postProcess(UV, 0, 128, 0, 128, num=10 * i, fig_save_dir=fig_save_dir)

    # save data
    data_save_dir = "./output/data/2dFN/"
    scipy.io.savemat(data_save_dir + "FN_1001x2x128x128.mat", {"uv": UV})
