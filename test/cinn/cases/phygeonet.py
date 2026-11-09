"""phygeonet USCNN, steady heat equation on a curvilinear mesh (examples/phygeonet/heat_equation.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import paddle

import ppsci

_spec = importlib.util.spec_from_file_location(
    "phygeonet_utils",
    Path(__file__).resolve().parents[3] / "examples" / "phygeonet" / "utils.py",
)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)  # dfdx / dfdy finite-difference operators

EXAMPLE = "phygeonet"
NOTE = (
    "synthetic smooth curvilinear mesh [1,2,19,84] (sinusoidally bent channel) with its "
    "metric terms jinv/dxdxi/dydxi/dxdeta/dydeta from np.gradient, instead of heat_equation.npz"
)
BATCH_SIZE = 1
NY, NX, PAD, H = 19, 84, 1, 0.01
METRICS = ("jinvs", "dxdxis", "dydxis", "dxdetas", "dydetas")


def build_model():
    return ppsci.arch.USCNN(
        input_keys=["coords"],
        output_keys=["output_v"],
        hidden_size=[16, 32, 16],
        h=H,
        ny=NY,
        nx=NX,
        nvar_in=2,
        nvar_out=1,
        pad_singleside=PAD,
    )


def build_batch():
    eta, xi = np.meshgrid(np.arange(NY) * H, np.arange(NX) * H, indexing="ij")
    x = xi
    y = eta * (1.0 + 0.5 * np.sin(np.pi * xi / xi.max())) + 0.05 * np.sin(
        2 * np.pi * xi
    )
    dxdeta, dxdxi = np.gradient(x, H, H)
    dydeta, dydxi = np.gradient(y, H, H)
    jinv = 1.0 / (dxdxi * dydeta - dxdeta * dydxi)
    fields = dict(
        jinvs=jinv, dxdxis=dxdxi, dydxis=dydxi, dxdetas=dxdeta, dydetas=dydeta
    )
    inputs = {"coords": paddle.to_tensor(np.stack([x, y])[None].astype("float32"))}
    for k in METRICS:
        inputs[k] = paddle.to_tensor(fields[k][None, None].astype("float32"))
    return inputs, {}


def forward(model, inputs):
    return model({"coords": inputs["coords"]})


def compute_loss(model, inputs, labels):
    p = PAD
    v = model(inputs)["output_v"]
    jinv, dxdxi, dydxi = inputs["jinvs"], inputs["dxdxis"], inputs["dydxis"]
    dxdeta, dydeta = inputs["dxdetas"], inputs["dydetas"]
    v[:, 0, -p:, p:-p] = 0
    v[:, 0, :p, p:-p] = 1
    v[:, 0, p:-p, -p:] = 1
    v[:, 0, p:-p, 0:p] = 1
    v[:, 0, 0, 0] = 0.5 * (v[:, 0, 0, 1] + v[:, 0, 1, 0])
    v[:, 0, 0, -1] = 0.5 * (v[:, 0, 0, -2] + v[:, 0, 1, -1])
    dvdx = utils.dfdx(v, dydeta, dydxi, jinv)
    d2vdx2 = utils.dfdx(dvdx, dydeta, dydxi, jinv)
    dvdy = utils.dfdy(v, dxdxi, dxdeta, jinv)
    d2vdy2 = utils.dfdy(dvdy, dxdxi, dxdeta, jinv)
    return paddle.mean((d2vdy2 + d2vdx2) ** 2)
