"""heart LV forward elasticity, Hooke 3D (examples/heart/forward.py, conf/forward.yaml)."""
import importlib.util
from pathlib import Path

import paddle
from cases import _common as C

import ppsci

EXAMPLE = "heart"
NOTE = (
    "points in [-1, 1]^3 with random unit normals instead of STL meshes (mesh 'area' "
    "factor dropped); DATA constraint uses 1000 random points/labels instead of forward.csv"
)
E, NU, P = 9.0, 0.45, 1.064
N_BASE, N_ENDO, N_EPI, N_INT, N_DATA = 256, 2048, 32, 8000, 1000
BATCH_SIZE = N_BASE + N_ENDO + N_EPI + N_INT + N_DATA
KEYS = ("x", "y", "z")
NORMALS = ("normal_x", "normal_y", "normal_z")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u", "v", "w"),
        num_layers=10,
        hidden_size=20,
        activation="silu",
        weight_norm=True,
    )


def _normals(n, prefix):
    nrm = paddle.nn.functional.normalize(paddle.randn([n, 3]), axis=1)
    return {prefix + k: nrm[:, i : i + 1] for i, k in enumerate(NORMALS)}


def build_batch():
    inputs = C.points(KEYS, N_INT, -1.0, 1.0)
    inputs.update(C.points(KEYS, N_BASE, -1.0, 1.0, prefix="base_"))
    for side, n in (("endo", N_ENDO), ("epi", N_EPI)):
        inputs.update(C.points(KEYS, n, -1.0, 1.0, prefix=f"{side}_"))
        inputs.update(_normals(n, f"{side}_"))
    inputs.update({f"data_{k}": C.uniform([N_DATA, 1], -1.0, 1.0) for k in KEYS})
    labels = {k: 0.1 * C.randn([N_DATA, 1]) for k in ("u", "v", "w")}
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    path = Path(__file__).resolve().parents[3] / "examples" / "heart" / "equation.py"
    spec = importlib.util.spec_from_file_location("heart_equation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hooke = mod.Hooke(E=E, nu=NU, P=P, dim=3)
    # lambdify per constraint, as the Solver does with each constraint's output_expr
    EQUATIONS["interior"] = C.lambdify(
        hooke.equations, model, ("hooke_x", "hooke_y", "hooke_z")
    )
    EQUATIONS["endo"] = C.lambdify(hooke.equations, model, ("traction",))
    EQUATIONS["epi"] = C.lambdify(hooke.equations, model, ("traction",))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def _res_loss(model, inputs, funcs, targets, weights):
    """weighted MSE(expr, target) over one constraint, like BoundaryConstraint."""
    data = {**inputs, **model(inputs)}
    loss = 0.0
    for name, target in targets.items():
        loss = loss + weights.get(name, 1.0) * C.sq_mean(funcs[name](data) - target)
    ppsci.autodiff.clear()
    return loss


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(
        model,
        C.take(inputs, KEYS),
        EQUATIONS["interior"],
        {"hooke_x": 0.2, "hooke_y": 0.2, "hooke_z": 0.2},
    )
    out = model(C.take(inputs, KEYS, "base_"))
    loss += sum(0.2 * C.sq_mean(out[k]) for k in ("u", "v", "w"))
    # weight_dict of BC_ENDO names traction_{x,y,z}, so "traction" keeps weight 1
    loss += _res_loss(
        model,
        C.take(inputs, KEYS + NORMALS, "endo_"),
        EQUATIONS["endo"],
        {"traction": -P},
        {},
    )
    loss += _res_loss(
        model,
        C.take(inputs, KEYS + NORMALS, "epi_"),
        EQUATIONS["epi"],
        {"traction": 0.0},
        {"traction": 0.2},
    )
    out = model(C.take(inputs, KEYS, "data_"))
    loss += sum(paddle.sum(paddle.square(out[k] - labels[k])) for k in labels)
    return loss
