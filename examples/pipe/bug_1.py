import os

import numpy as np
import paddle

import ppsci
from ppsci.autodiff import jacobian


class My_PDE(ppsci.equation.pde.base.PDE):
    """Class for navier-stokes equation. [nu] as self-variable

    Args:
        rho (float): Density.
        dim (int): Dimension of equation.
        time (bool): Whether the euqation is time-dependent.
    """

    def __init__(self):
        super().__init__()

        def eq_1(out):
            y = out["y"]
            v = out["v"]
            dvdy_paddle = jacobian(v, y)
            return dvdy_paddle

        self.add_equation("eq_1", eq_1)


if __name__ == "__main__":
    ppsci.utils.misc.set_random_seed(42)
    from paddle.fluid import core

    core.set_prim_eager_enabled(True)
    os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/pipe")
    ppsci.utils.logger.init_logger("ppsci", f"./output_pipe/train.log", "info")

    input = np.load("./data/input/input_x_y_nu.npz")
    x = paddle.to_tensor(input["x"], dtype="float32", stop_gradient=False)
    y = paddle.to_tensor(input["y"], dtype="float32", stop_gradient=False)
    nu = paddle.to_tensor(input["nu"], dtype="float32", stop_gradient=False)

    v_dict = np.load("./data/compare_4/v_epoch_1.npz")
    y_dict = np.load("./data/compare_4/y_epoch_1.npz")
    dvdy_dict = np.load("./data/compare_4/dvdy_epoch_1.npz")

    initial_bias_v = np.load("./data/init_net_params/initial_bias_v.npz")
    initial_weight_v = np.load("./data/init_net_params/initial_weight_v.npz")
    model = ppsci.arch.MLP(
        ["x", "x", "y", "nu"],
        ["v"],
        0,
        50,
        "identity",
        False,
        False,
        initial_weight_v,
        initial_bias_v,
    )

    eq = My_PDE()
    interior_geom = ppsci.geometry.PointCloud(
        coord_dict={"x": x, "y": y}, extra_data={"nu": nu}, data_key=["x", "y", "nu"]
    )
    pde_constraint = ppsci.constraint.InteriorConstraint(
        eq.equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom=interior_geom,
        dataloader_cfg={
            "dataset": "NamedArrayDataset",
            "num_workers": 1,
            "batch_size": 128,
            "iters_per_epoch": 10,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean"),
        evenly=True,
        weight_dict={"u": 1, "v": 1, "p": 1},
        name="EQ",
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        {pde_constraint.name: pde_constraint},
        ppsci.optimizer.Adam(5e-3)([model]),
        equation=eq,
    )

    # train model
    solver.train()
