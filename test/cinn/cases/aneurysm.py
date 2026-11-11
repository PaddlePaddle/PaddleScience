"""aneurysm 3D steady flow in an aneurysm (examples/aneurysm/aneurysm.py, conf/aneurysm.yaml)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "aneurysm"
NOTE = (
    "points sampled uniformly in a box (inlet: cube around INLET_CENTER) instead of from the STL "
    "meshes; per-point 'area' set to a guessed measure / batch and random unit normals for the "
    "integral constraints; constant lr instead of ExponentialDecay"
)
NU, SCALE, RHO = 0.025, 0.4, 1.0
INLET_NORMAL = (0.8526, -0.428, 0.299)
INLET_CENTER = (-4.24298030045776, 4.082857101816247, -4.637790193399717)
INLET_VEL = 1.5
INLET_RADIUS = float(np.sqrt(21.1284 * SCALE**2 / np.pi))
N_INLET, N_OUTLET, N_NOSLIP, N_PDE, N_IGC = 1100, 650, 5200, 6000, 310
BATCH_SIZE = N_INLET + N_OUTLET + N_NOSLIP + N_PDE + 2 * N_IGC
KEYS = ("x", "y", "z")
VEL = ("u", "v", "w")
NORMALS = ("normal_x", "normal_y", "normal_z")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u", "v", "w", "p"),
        num_layers=6,
        hidden_size=512,
        activation="silu",
        weight_norm=True,
    )


def _area(n, measure, shape=None):
    return paddle.full(shape or [n, 1], measure / n, dtype="float32")


def build_batch():
    inputs = C.points(KEYS, N_PDE, -8.0, 8.0, prefix="pde_")
    inputs["pde_area"] = _area(N_PDE, 60.0)
    inlet = {
        f"inlet_{k}": C.uniform([N_INLET, 1], c - INLET_RADIUS, c + INLET_RADIUS)
        for k, c in zip(KEYS, INLET_CENTER)
    }
    inputs.update(inlet, inlet_area=_area(N_INLET, 21.1284 * SCALE**2))
    inputs.update(C.points(KEYS, N_OUTLET, -8.0, 8.0, prefix="outlet_"))
    inputs["outlet_area"] = _area(N_OUTLET, 3.4)
    inputs.update(C.points(KEYS, N_NOSLIP, -8.0, 8.0, prefix="noslip_"))
    inputs["noslip_area"] = _area(N_NOSLIP, 100.0)
    for name in ("igc_outlet", "igc_integral"):
        shape = [
            1,
            N_IGC,
            1,
        ]  # [batch_size, integral_batch_size, 1] as IntegralConstraint
        inputs.update({f"{name}_{k}": C.uniform(shape, -8.0, 8.0) for k in KEYS})
        normal = C.randn([1, N_IGC, 3])
        normal = normal / paddle.linalg.norm(normal, axis=-1, keepdim=True)
        for i, k in enumerate(NORMALS):
            inputs[f"{name}_{k}"] = normal[..., i : i + 1]
        inputs[f"{name}_area"] = _area(N_IGC, 3.4, shape)

    dist = paddle.sqrt(
        sum((inlet[f"inlet_{k}"] - c) ** 2 for k, c in zip(KEYS, INLET_CENTER))
    )
    parabola = INLET_VEL * paddle.clip(1 - (dist / INLET_RADIUS) ** 2, min=0.0)
    labels = {f"inlet_{k}": n * parabola for k, n in zip(VEL, INLET_NORMAL)}
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    ns = ppsci.equation.NavierStokes(NU * SCALE, RHO, 3, False)
    EQUATIONS.update(C.lambdify(ns.equations, model))
    ndv = ppsci.equation.NormalDotVec(VEL)
    EQUATIONS.update(C.lambdify(ndv.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS, "pde_"))


def _sum_sq(x, area, w=1.0):
    # ppsci.loss.MSELoss("sum") with the geometry "area" weighting
    return paddle.sum(paddle.square(x) * w * area)


def compute_loss(model, inputs, labels):
    pde_in = C.take(inputs, KEYS, "pde_")
    data = {**pde_in, **model(pde_in)}
    loss = 0.0
    for name in ("continuity", "momentum_x", "momentum_y", "momentum_z"):
        loss = loss + _sum_sq(EQUATIONS[name](data), inputs["pde_area"])
    C.autodiff.clear()

    out = model(C.take(inputs, KEYS, "inlet_"))
    for k in VEL:
        loss = loss + _sum_sq(out[k] - labels[f"inlet_{k}"], inputs["inlet_area"])
    out = model(C.take(inputs, KEYS, "outlet_"))
    loss = loss + _sum_sq(out["p"], inputs["outlet_area"])
    out = model(C.take(inputs, KEYS, "noslip_"))
    for k in VEL:
        loss = loss + _sum_sq(out[k], inputs["noslip_area"])

    for name, target in (("igc_outlet", 2.54), ("igc_integral", -2.54)):
        igc_in = C.take(inputs, KEYS, f"{name}_")
        data = {**C.take(inputs, NORMALS, f"{name}_"), **igc_in, **model(igc_in)}
        flux = (EQUATIONS["normal_dot_vec"](data) * inputs[f"{name}_area"]).sum(axis=1)
        loss = loss + 0.1 * paddle.sum(
            paddle.square(flux - target)
        )  # IntegralLoss("sum")
    return loss
