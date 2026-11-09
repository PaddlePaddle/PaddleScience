"""spinn separable PINN, 3D Helmholtz (examples/spinn/helmholtz3d.py)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "spinn"
NOTE = (
    "one fixed set of nc=64 axis points (example resamples every 100 iters); "
    "plain Adam lr 1e-3 without ExponentialDecay"
)
A1, A2, A3 = 4, 4, 3
NC = 64
BATCH_SIZE = NC**3 + 6 * NC**2  # grid points of PDE + 6 boundary faces
KEYS = ("x", "y", "z")


def build_model():
    return ppsci.arch.SPINN(
        KEYS, ("u",), r=32, num_layers=4, hidden_size=64, activation="tanh"
    )


def build_batch():
    xc, yc, zc = (C.uniform([NC, 1], -1.0, 1.0, requires_grad=True) for _ in range(3))
    xm, ym, zm = np.meshgrid(xc.numpy(), yc.numpy(), zc.numpy(), indexing="ij")
    u_gt = (
        np.sin(A1 * np.pi * xm) * np.sin(A2 * np.pi * ym) * np.sin(A3 * np.pi * zm)
    )[..., None]
    uc = -((A1 * np.pi) ** 2 + (A2 * np.pi) ** 2 + (A3 * np.pi) ** 2) * u_gt + u_gt
    inputs = {"x": xc, "y": yc, "z": zc}
    one, neg = paddle.to_tensor([[1.0]]), paddle.to_tensor([[-1.0]])
    faces = (
        (one, yc, zc),
        (neg, yc, zc),
        (xc, one, zc),
        (xc, neg, zc),
        (xc, yc, one),
        (xc, yc, neg),
    )
    for i, face in enumerate(faces):
        for k, v in zip(KEYS, face):
            inputs[f"bc{i}_{k}"] = v.detach().clone()
    return inputs, {"helmholtz": paddle.to_tensor(uc.astype("float32"))}


EQUATIONS = {}


def prepare(model):
    EQUATIONS.update(ppsci.equation.Helmholtz(3, 1.0, model).equations)


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    xin = C.take(inputs, KEYS)
    res = EQUATIONS["helmholtz"]({**xin, **model(xin)})
    loss = C.mse(res, labels["helmholtz"])
    for i in range(6):
        u = model(C.take(inputs, KEYS, f"bc{i}_"))["u"]
        loss += C.mse(u, paddle.zeros_like(u))
    return loss
