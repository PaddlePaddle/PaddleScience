import math
import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.fluid import core

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian


def ThreeD_mesh(x_1d, x_2d, tmp_1d, sigma, mu):
    tmp_3d = np.expand_dims(np.tile(tmp_1d, len(x_2d)), 1).astype("float")
    x = []
    for x0 in x_2d:
        tmpx = np.tile(x0, len(tmp_1d))
        x.append(tmpx)
    x = np.reshape(x, (len(tmp_3d), 1))
    return x, tmp_3d


class NavierStokes(ppsci.equation.pde.base.PDE):
    """Class for navier-stokes equation.

    Args:
        rho (float): Density.
        dim (int): Dimension of equation.
        time (bool): Whether the euqation is time-dependent.
    """

    def __init__(self, rho: float, dim: int, time: bool):
        super().__init__()
        self.rho = rho
        self.dim = dim
        self.time = time

        def continuity_compute_func(out):
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            continuity = jacobian(u, x) + jacobian(v, y)

            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                continuity += jacobian(w, z)
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def momentum_x_compute_func(out):
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_x = (
                u * jacobian(u, x)
                + v * jacobian(u, y)
                - nu / rho * hessian(u, x)
                - nu / rho * hessian(u, y)
                + 1 / rho * jacobian(p, x)
            )
            if self.time:
                t = out["t"]
                momentum_x += jacobian(u, t)
            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                momentum_x += w * jacobian(u, z)
                momentum_x -= nu / rho * hessian(u, z)
            return momentum_x

        self.add_equation("momentum_x", momentum_x_compute_func)

        def momentum_y_compute_func(out):
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_y = (
                u * jacobian(v, x)
                + v * jacobian(v, y)
                - nu / rho * hessian(v, x)
                - nu / rho * hessian(v, y)
                + 1 / rho * jacobian(p, y)
            )
            if self.time:
                t = out["t"]
                momentum_y += jacobian(v, t)
            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                momentum_y += w * jacobian(v, z)
                momentum_y -= nu / rho * hessian(v, z)
            return momentum_y

        self.add_equation("momentum_y", momentum_y_compute_func)

        if self.dim == 3:

            def momentum_z_compute_func(out):
                x, y = out["x"], out["y"]
                u, v, w, p = out["u"], out["v"], out["w"], out["p"]
                momentum_z = (
                    u * jacobian(w, x)
                    + v * jacobian(w, y)
                    + w * jacobian(w, z)
                    - nu / rho * hessian(w, x)
                    - nu / rho * hessian(w, y)
                    - nu / rho * hessian(w, z)
                    + 1 / rho * jacobian(p, z)
                )
                if self.time:
                    t = out["t"]
                    momentum_z += jacobian(w, t)
                return momentum_z

            self.add_equation("momentum_z", momentum_z_compute_func)


os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm")
output_dir = "./output_0504"
ppsci.utils.misc.set_random_seed(42)

# initialize logger
ppsci.utils.logger.init_logger("ppsci", f"{output_dir}/train.log", "info")
core.set_prim_eager_enabled(True)

EPOCHS = 500

L = 1
X_IN = 0
X_OUT = X_IN + L
P_OUT = 0  # pressure at the outlet of pipe
P_IN = 0.1  # pressure at the inlet of pipe
rInlet = 0.05

nPt = 100
unique_x = np.linspace(X_IN, X_OUT, nPt)
mu = 0.5 * (X_OUT - X_IN)

N_y = 20
x_2d = np.tile(unique_x, N_y)
x_2d = np.reshape(x_2d, (len(x_2d), 1))

nu = 1e-3

sigma = 0.1

scaleStart = -0.02
scaleEnd = 0
Ng = 50
scale_1d = np.linspace(scaleStart, scaleEnd, Ng, endpoint=True)
x, scale = ThreeD_mesh(unique_x, x_2d, scale_1d, sigma, mu)

# axisymetric boundary
R = (
    scale
    * 1
    / math.sqrt(2 * np.pi * sigma**2)
    * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
)

# Generate stenosis
yUp = (rInlet - R) * np.ones_like(x)
yDown = (-rInlet + R) * np.ones_like(x)

idx = np.where(scale == scaleStart)
plt.figure()
plt.scatter(x[idx], yUp[idx])
plt.scatter(x[idx], yDown[idx])
plt.axis("equal")
plt.show()
plt.savefig("idealized_stenotid_vessel", bbox_inches="tight")

y = np.zeros([len(x), 1])
for x0 in unique_x:
    index = np.where(x[:, 0] == x0)[0]
    Rsec = max(yUp[index])
    # print('shape of index',index.shape)
    tmpy = np.linspace(-Rsec, Rsec, len(index)).reshape(len(index), -1)
    # print('shape of tmpy',tmpy.shape)
    y[index] = tmpy

print("shape of x", x.shape)
print("shape of y", y.shape)
print("shape of sacle", scale.shape)

g = 9.8
RHO = 1

BATCH_SIZE = 50
LEARNING_RATE = 1e-3
LAYER_NUMBER = 4 - 1

path = "Cases/"

h_nD = 30
HIDDEN_SIZE = 20

index = [i for i in range(x.shape[0])]
res = list(zip(x, y, scale))
np.random.shuffle(res)
x, y, scale = zip(*res)
x = np.array(x).astype(float)
y = np.array(y).astype(float)
scale = np.array(scale).astype(float)

interior_geom = ppsci.geometry.PointCloud(
    coord_dict={"x": x, "y": y},
    extra_data={"scale": scale},
    data_key=["x", "y", "scale"],
)

model_2 = ppsci.arch.MLP(
    ["x", "y", "scale"],
    ["u"],
    LAYER_NUMBER,
    HIDDEN_SIZE,
    "swish",
    False,
    False,
    np.load(f"data/net2_params/weight_epoch_0.npz"),
    np.load(f"data/net2_params/bias_epoch_0.npz"),
)

model_3 = ppsci.arch.MLP(
    ["x", "y", "scale"],
    ["v"],
    LAYER_NUMBER,
    HIDDEN_SIZE,
    "swish",
    False,
    False,
    np.load(f"data/net3_params/weight_epoch_0.npz"),
    np.load(f"data/net3_params/bias_epoch_0.npz"),
)

model_4 = ppsci.arch.MLP(
    ["x", "y", "scale"],
    ["p"],
    LAYER_NUMBER,
    HIDDEN_SIZE,
    "swish",
    False,
    False,
    np.load(f"data/net4_params/weight_epoch_0.npz"),
    np.load(f"data/net4_params/bias_epoch_0.npz"),
)

h = None


class Output_transform:
    def __init__(self) -> None:
        pass

    def __call__(self, out, input):
        new_out = {}
        x, y, scale = input["x"], input["y"], input["scale"]
        # axisymetric boundary
        if next(iter(out.keys())) == "u":
            R = (
                scale
                * 1
                / np.sqrt(2 * np.pi * sigma**2)
                * paddle.exp(-((x - mu) ** 2) / (2 * sigma**2))
            )
            self.h = rInlet - R
            u = out["u"]
            # The no-slip condition of velocity on the wall
            new_out["u"] = u * (self.h**2 - y**2)
        elif next(iter(out.keys())) == "v":
            v = out["v"]
            # The no-slip condition of velocity on the wall
            new_out["v"] = (self.h**2 - y**2) * v
        elif next(iter(out.keys())) == "p":
            p = out["p"]
            # The pressure inlet [p_in = 0.1] and outlet [p_out = 0]
            new_out["p"] = (
                (X_IN - x) * 0
                + (P_IN - P_OUT) * (X_OUT - x) / L
                + 0 * y
                + (X_IN - x) * (X_OUT - x) * p
            )
        else:
            raise NotImplementedError(f"{out.keys()} are outputs to be implemented")

        return new_out


shared_transform = Output_transform()
model_2.register_output_transform(shared_transform)
model_3.register_output_transform(shared_transform)
model_4.register_output_transform(shared_transform)
model = ppsci.arch.ModelList([model_2, model_3, model_4])

optimizer2 = ppsci.optimizer.Adam(
    LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=10**-15
)([model])

equation = {"NavierStokes": NavierStokes(RHO, 2, False)}

pde_constraint = ppsci.constraint.InteriorConstraint(
    equation["NavierStokes"].equations,
    {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
    geom=interior_geom,
    dataloader_cfg={
        "dataset": "NamedArrayDataset",
        "num_workers": 1,
        "batch_size": BATCH_SIZE,
        "iters_per_epoch": int(x.shape[0] / BATCH_SIZE),
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
    output_dir,
    optimizer2,
    epochs=EPOCHS,
    iters_per_epoch=int(x.shape[0] / BATCH_SIZE),
    eval_during_train=False,
    save_freq=10,
    log_freq=1,
    equation=equation,
    # checkpoint_path="/workspace/wangguan/PaddleScience_Surrogate/examples/pipe/output_pipe/checkpoints/epoch_3000",
)
solver.train()
