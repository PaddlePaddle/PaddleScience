import argparse
import os

import fdm
import matplotlib.pyplot as plt
import numpy as np

import ppsci
from ppsci.utils import logger


def main():
    parser = argparse.ArgumentParser(description="Solve a 2D heat equation by PINN")
    parser.add_argument("--epoch", default=1000, type=int, help="max epochs")
    parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
    parser.add_argument(
        "--output_dir",
        default="./output_heat2d",
        type=str,
        help="output folder",
    )

    args = parser.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(2)
    # set training hyper-parameters
    EPOCHS = args.epoch
    ITERS_PER_EPOCH = 1

    # set output directory
    OUTPUT_DIR = args.output_dir
    logger.init_logger("ppsci", os.path.join(OUTPUT_DIR, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(("x", "y"), ("u",), 9, 20, "tanh")

    # set equation
    equation = {"heat": ppsci.equation.Laplace(dim=2)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-1.0, -1.0), (1.0, 1.0))}

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
    }

    NPOINT_PDE = 99**2
    NPOINT_TOP = 25
    NPOINT_BOTTOM = 25
    NPOINT_LEFT = 25
    NPOINT_RIGHT = 25

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["heat"].equations,
        {"laplace": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
        ppsci.loss.MSELoss("mean"),
        weight_dict={
            "laplace": 1,
        },
        evenly=True,
        name="EQ",
    )

    bc_top = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_TOP},
        ppsci.loss.MSELoss("mean"),
        weight_dict={
            "u": 0.25,
        },
        criteria=lambda x, y: np.isclose(y, 1),
        name="BC_top",
    )
    bc_bottom = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 50 / 75},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_BOTTOM},
        ppsci.loss.MSELoss("mean"),
        weight_dict={
            "u": 0.25,
        },
        criteria=lambda x, y: np.isclose(y, -1),
        name="BC_bottom",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 1},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_LEFT},
        ppsci.loss.MSELoss("mean"),
        weight_dict={
            "u": 0.25,
        },
        criteria=lambda x, y: np.isclose(x, -1),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_RIGHT},
        ppsci.loss.MSELoss("mean"),
        weight_dict={
            "u": 0.25,
        },
        criteria=lambda x, y: np.isclose(x, 1),
        name="BC_right",
    )

    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_top.name: bc_top,
        bc_bottom.name: bc_bottom,
        bc_left.name: bc_left,
        bc_right.name: bc_right,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=args.lr)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        equation=equation,
        geom=geom,
    )
    # train model
    solver.train()

    # begin eval
    n = 100
    input_data = geom["rect"].sample_interior(n**2, evenly=True)
    pinn_output = solver.predict(input_data, return_numpy=True)["u"].reshape(n, n)
    fdm_output = fdm.solve(n, 1).T
    mes_loss = np.mean(np.square(pinn_output - (fdm_output / 75.0)))
    logger.info(f"The norm MSE loss between the FDM and PINN is {mes_loss}")

    x = input_data["x"].reshape(n, n)
    y = input_data["y"].reshape(n, n)

    plt.subplot(2, 1, 1)
    plt.pcolormesh(x, y, pinn_output * 75.0, cmap="magma")
    plt.colorbar()
    plt.title("PINN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(2, 1, 2)
    plt.pcolormesh(x, y, fdm_output, cmap="magma")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("FDM")
    plt.tight_layout()
    plt.axis("square")
    plt.savefig(os.path.join(OUTPUT_DIR, "pinn_fdm_comparison.png"))
    plt.close()

    frames_val = np.array([-0.75, -0.5, -0.25, 0.0, +0.25, +0.5, +0.75])
    frames = [*map(int, (frames_val + 1) / 2 * (n - 1))]
    height = 3
    plt.figure("", figsize=(len(frames) * height, 2 * height))

    for i, var_index in enumerate(frames):
        plt.subplot(2, len(frames), i + 1)
        plt.title(f"y = {frames_val[i]:.2f}")
        plt.plot(
            x[:, var_index],
            pinn_output[:, var_index] * 75.0,
            "r--",
            lw=4.0,
            label="pinn",
        )
        plt.plot(x[:, var_index], fdm_output[:, var_index], "b", lw=2.0, label="FDM")
        plt.ylim(0.0, 100.0)
        plt.xlim(-1.0, +1.0)
        plt.xlabel("x")
        plt.ylabel("T")
        plt.tight_layout()
        plt.legend()

    for i, var_index in enumerate(frames):
        plt.subplot(2, len(frames), len(frames) + i + 1)
        plt.title(f"x = {frames_val[i]:.2f}")
        plt.plot(
            y[var_index, :],
            pinn_output[var_index, :] * 75.0,
            "r--",
            lw=4.0,
            label="pinn",
        )
        plt.plot(y[var_index, :], fdm_output[var_index, :], "b", lw=2.0, label="FDM")
        plt.ylim(0.0, 100.0)
        plt.xlim(-1.0, +1.0)
        plt.xlabel("y")
        plt.ylabel("T")
        plt.tight_layout()
        plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "profiles.png"))


if __name__ == "__main__":
    main()
