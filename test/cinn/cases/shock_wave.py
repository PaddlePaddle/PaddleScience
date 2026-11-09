"""shock_wave Euler2D around a cylinder, Ma=2.0 (examples/shock_wave, shock_wave_Ma2.0.yaml)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci
from ppsci.autodiff import jacobian

EXAMPLE = "shock_wave"
NOTE = (
    "uniform points (cylinder removed) instead of LHS; Adam lr 1e-3 instead of LBFGS lr 0.1; "
    "viscosity factor `relu` fixed at its epoch-50 value (0.45) instead of tracking solver.global_step"
)
MA, LT, LX, LY = 2.0, 0.4, 1.5, 2.0
RX, RY, RD = 1.0, 1.0, 0.25
N_INTERIOR, N_BOUNDARY = 100000, 10000
BATCH_SIZE = N_INTERIOR + 3 * N_BOUNDARY  # before removing points inside the cylinder
RHO1, P1, GAMMA, V1 = 2.112, 3.001, 1.4, 0.0
U1 = float(np.sqrt(GAMMA * P1 / RHO1) * MA)
RELU = max(0.0, 50 / 100 - 0.05)
KEYS = ("t", "x", "y")
OUT_KEYS = ("u", "v", "p", "rho")


def build_model():
    return ppsci.arch.MLP(KEYS, OUT_KEYS, 9, 90, "tanh")


def _box(n, hi, prefix, drop_circle):
    pts = {k: C.uniform([n, 1], 0.0, h) for k, h in zip(KEYS, hi)}
    if drop_circle:
        x, y = pts["x"].numpy(), pts["y"].numpy()
        keep = ~((x[:, 0] - RX) ** 2 + (y[:, 0] - RY) ** 2 < RD**2)
        pts = {k: paddle.to_tensor(v.numpy()[keep]) for k, v in pts.items()}
    for v in pts.values():
        v.stop_gradient = False
    return {prefix + k: v for k, v in pts.items()}


def build_batch():
    inputs = _box(N_INTERIOR, (LT, LX, LY), "", True)
    inputs.update(_box(N_BOUNDARY, (LT, 0.0, LY), "bcL_", False))
    inputs.update(_box(N_BOUNDARY, (0.0, LX, LY), "ic_", True))
    # cylinder wall, as generate_bc_down_circle_points
    theta = 2 * np.pi * C.randn([N_BOUNDARY, 1])
    cos, sin = paddle.cos(np.pi / 2 + theta), paddle.sin(np.pi / 2 + theta)
    inputs.update(
        {
            "bcI_t": C.randn([N_BOUNDARY, 1], True) * LT,
            "bcI_x": RX + cos * RD,
            "bcI_y": RY + sin * RD,
            "bcI_sin": sin,
            "bcI_cos": cos,
        }
    )
    for k in ("bcI_t", "bcI_x", "bcI_y"):
        inputs[k] = inputs[k].detach()
        inputs[k].stop_gradient = False
    return inputs, {}


def _lam(out):
    delta_u = jacobian(out["u"], out["x"]) + jacobian(out["v"], out["y"])
    return 0.1 * (paddle.abs(delta_u) - delta_u) * RELU + 1


def euler2d(out):
    t, x, y = out["t"], out["x"], out["y"]
    u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
    lam = _lam(out)
    continuity = jacobian(rho, t) + jacobian(rho * u, x) + jacobian(rho * v, y)
    x_mom = (
        jacobian(rho * u, t) + jacobian(rho * u**2 + p, x) + jacobian(rho * u * v, y)
    )
    y_mom = (
        jacobian(rho * v, t) + jacobian(rho * u * v, x) + jacobian(rho * v**2 + p, y)
    )
    e1 = (rho * 0.5 * (u**2 + v**2) + 3.5 * p) * u
    e2 = (rho * 0.5 * (u**2 + v**2) + 3.5 * p) * v
    e = rho * 0.5 * (u**2 + v**2) + p / 0.4
    energy = jacobian(e, t) + jacobian(e1, x) + jacobian(e2, y)
    return [r / lam for r in (continuity, x_mom, y_mom, energy)]


def bc_eq(out):
    x, y, u, v, p, rho = (out[k] for k in ("x", "y", "u", "v", "p", "rho"))
    sin, cos = out["sin"], out["cos"]
    lam = _lam(out)
    item1 = u * cos + v * sin
    item2 = jacobian(p, x) * cos + jacobian(p, y) * sin
    item3 = jacobian(rho, x) * cos + jacobian(rho, y) * sin
    return [r / lam for r in (item1, item2, item3)]


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    xin = C.take(inputs, KEYS)
    loss = sum(C.sq_mean(r) for r in euler2d({**xin, **model(xin)}))
    ppsci.autodiff.clear()
    ic = model(C.take(inputs, KEYS, "ic_"))
    for k, val in zip(OUT_KEYS, (U1, 0.0, P1, RHO1)):
        loss += 10 * C.sq_mean(ic[k] - val)
    bcl = model(C.take(inputs, KEYS, "bcL_"))
    for k, val in zip(OUT_KEYS, (U1, V1, P1, RHO1)):
        loss += 10 * C.sq_mean(bcl[k] - val)
    bci_in = C.take(inputs, KEYS + ("sin", "cos"), "bcI_")
    out = {**bci_in, **model(C.take(bci_in, KEYS))}
    loss += 10 * sum(C.sq_mean(r) for r in bc_eq(out))
    ppsci.autodiff.clear()
    return loss
