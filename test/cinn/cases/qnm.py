"""qnm Teukolsky quasinormal modes with DeepONet, float64 (examples/qnm/main.py, main.yaml)."""
import os
import sys

import numpy as np
import paddle
from cases import _common as C

import ppsci
from ppsci.autodiff import hessian as hes
from ppsci.autodiff import jacobian as jac

_EX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "qnm")
sys.path.insert(0, os.path.abspath(_EX_DIR))
from model import DeepONet  # noqa: E402

DTYPE = "float64"
paddle.set_default_dtype(DTYPE)  # as examples/qnm/main.py

EXAMPLE = "qnm"
NOTE = (
    "50 spins a in [0, 0.49] with eigenvalues near (w, A)=(0.747-0.178i, 4) and random "
    "f/g labels instead of data/*.npy + demo.csv; `a` stays a tensor (example calls "
    "a.numpy() and np.sqrt, which cannot be traced); Adam instead of SOAP, no lr decay"
)
M, S = 0, -2
A_NUM, SV_BS, PDE_BS = 50, 128, 128
BATCH_SIZE = A_NUM * (SV_BS + PDE_BS)


def F_terms(a, w, A, s, m, x):
    r_plus = (1 + paddle.sqrt(1 - 4 * a**2)) / 2
    F0 = (
        -(a**4) * x**2 * w**2
        - 2 * a**3 * m * x**2 * w
        + a**2
        * (
            -A * x**2
            + x**2
            * (
                4 * (r_plus + 1) * w**2
                + 2j * (r_plus + 2) * w
                + 2j * s * (w + 1j)
                - 2
            )
            + x * w**2
            - w**2
        )
        + 2 * a * m * (r_plus * x**2 * (2 * w + 1j) - x * (w + 1j) - w)
        + A * (x - 1)
        - 1j
        * r_plus
        * (2 * w + 1j)
        * (x**2 * (s - 2j * w + 1) - 2 * (s + 1) * x + 2j * w)
        + (s + 1) * (x - 2j * w)
    )
    F1 = (
        2 * a**4 * x**4 * (x - 1j * w)
        - 2j * a**3 * m * x**4
        + a**2
        * x**2
        * (
            2 * r_plus * x**2 * (-1 + 2j * w)
            - (s + 3) * x**2
            + 2 * x * (s + 1j * w + 2)
            - 4j * w
        )
        + 2j * a * m * (x - 1) * x**2
        + (x - 1)
        * (
            2 * r_plus * x**2 * (1 - 2j * w)
            + (s + 1) * x**2
            - 2 * (s + 1) * x
            + 2j * w
        )
    )
    F2 = a**4 * x**6 - 2 * a**2 * (x - 1) * x**4 + (x - 1) ** 2 * x**2
    return F0, F1, F2


def G_terms(a, w, A, s, m, u):
    ams, aps = np.abs(m - s), np.abs(m + s)
    G0 = (
        4 * a**2 * (u**2 - 1) * w**2
        - 4 * a * (u**2 - 1) * w * ((u - 1) * ams + (u + 1) * aps + 2 * (s + 1) * u)
        + 4 * (A * (u**2 - 1) + m**2 + 2 * m * s * u + s * ((s + 1) * u**2 - 1))
        - 2 * (u**2 - 1) * aps
        - 2 * (u**2 - 1) * ams * (aps + 1)
        - (u - 1) ** 2 * ams**2
        - (u + 1) ** 2 * aps**2
    )
    G1 = -8 * a * (u**2 - 1) ** 2 * w - 4 * (u**2 - 1) * (
        (u - 1) * ams + (u + 1) * aps + 2 * u
    )
    G2 = -4 * (u**2 - 1) ** 2
    return G0, G1, G2


def qnm_residuals(out):
    a = out["a"]
    w = paddle.as_complex(paddle.stack([out["w_r"], out["w_i"]], axis=-1))
    A = paddle.as_complex(paddle.stack([out["A_r"], out["A_i"]], axis=-1))
    x, u = out["x"], out["u"]
    f_r, f_i, g_r, g_i = out["f_r"], out["f_i"], out["g_r"], out["g_i"]
    F0, F1, F2 = F_terms(a, w, A, S, M, x)
    G0, G1, G2 = G_terms(a, w, A, S, M, u)
    F0_r, F0_i, F1_r, F1_i = F0.real(), F0.imag(), F1.real(), F1.imag()
    G0_r, G0_i, G1_r, G1_i = G0.real(), G0.imag(), G1.real(), G1.imag()
    qnm1 = (
        F2 * hes(f_r, x)
        + F1_r * jac(f_r, x)
        - F1_i * jac(f_i, x)
        + F0_r * f_r
        - F0_i * f_i
    )
    qnm2 = (
        F2 * hes(f_i, x)
        + F1_r * jac(f_i, x)
        + F1_i * jac(f_r, x)
        + F0_r * f_i
        + F0_i * f_r
    )
    qnm3 = (
        G2 * hes(g_r, u)
        + G1_r * jac(g_r, u)
        - G1_i * jac(g_i, u)
        + G0_r * g_r
        - G0_i * g_i
    )
    qnm4 = (
        G2 * hes(g_i, u)
        + G1_r * jac(g_i, u)
        + G1_i * jac(g_r, u)
        + G0_r * g_i
        + G0_i * g_r
    )
    return qnm1, qnm2, qnm3, qnm4


def build_model():
    return DeepONet(
        trunk_input_keys=("x", "u"),
        branch_a_input_keys=("a",),
        output_keys=("f_r", "f_i", "g_r", "g_i"),
        trunk_num_layers=3,
        trunk_hidden_size=256,
        trunk_activation="tanh",
        branch_input_dim=1,
        branch_num_layers=2,
        branch_hidden_size=128,
        branch_activation="tanh",
        hidden_out_size=128,
    )


def _rand(shape, low, high, requires_grad=False):
    t = paddle.uniform(shape, dtype=DTYPE, min=low, max=high)
    t.stop_gradient = not requires_grad
    return t


def _per_a(values, bs):
    return paddle.to_tensor(np.repeat(values, bs).reshape(-1, 1).astype(DTYPE))


def build_batch():
    a_list = np.linspace(0.0, 0.49, A_NUM)
    lin_x = np.linspace(0.0, 1.0, SV_BS)
    lin_u = np.linspace(-1.0, 1.0, SV_BS)
    inputs = {  # supervised batch: same linspace grid for every a
        "sv_x": paddle.to_tensor(np.tile(lin_x, A_NUM).reshape(-1, 1).astype(DTYPE)),
        "sv_u": paddle.to_tensor(np.tile(lin_u, A_NUM).reshape(-1, 1).astype(DTYPE)),
        "sv_a": _per_a(a_list, SV_BS),
        "x": _rand([A_NUM * PDE_BS, 1], 0.0, 1.0, True),
        "u": _rand([A_NUM * PDE_BS, 1], -1.0, 1.0, True),
        "a": _per_a(a_list, PDE_BS),
        "w_r": _per_a(0.74734 + 0.1 * a_list, PDE_BS),
        "w_i": _per_a(-0.17793 + 0.01 * a_list, PDE_BS),
        "A_r": _per_a(4.0 - 0.5 * a_list, PDE_BS),
        "A_i": _per_a(0.02 * a_list, PDE_BS),
    }
    n = A_NUM * SV_BS
    labels = {
        k: _rand([n, 1], -1.0, 1.0)
        for k in ("fr_label", "fi_label", "gr_label", "gi_label")
    }
    return inputs, labels


def forward(model, inputs):
    return model(C.take(inputs, ("x", "u", "a")))


SV_WEIGHTS = {
    "fr_label": 2.0,
    "fi_label": 1.0,
    "gr_label": 1.0,
    "gi_label": 1.0,
}  # "fi_lable" typo in example


def compute_loss(model, inputs, labels):
    sv = model(C.take(inputs, ("x", "u", "a"), "sv_"))
    loss = 0.0
    for key, out_key in zip(SV_WEIGHTS, ("f_r", "f_i", "g_r", "g_i")):
        loss += SV_WEIGHTS[key] * C.mse(sv[out_key], labels[key])
    pde_in = C.take(inputs, ("x", "u", "a", "w_r", "w_i", "A_r", "A_i"))
    out = {**pde_in, **model(C.take(pde_in, ("x", "u", "a")))}
    # MAELoss weights use keys ode1..4 that never match QNM1..4, so all are 1
    for r in qnm_residuals(out):
        loss += paddle.mean(paddle.abs(r))
    ppsci.autodiff.clear()
    return loss
