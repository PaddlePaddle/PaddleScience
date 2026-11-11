"""fpde 2D fractional Poisson on the unit disk (examples/fpde/fractional_poisson_2d.py)."""
import numpy as np
import paddle
from cases import _common as C
from paddle import sparse
from scipy import special

import ppsci

EXAMPLE = "fpde"
NOTE = (
    "100 interior points drawn with a fixed numpy seed (r<0.99) instead of Hammersley; "
    "integration background points and sparse int_mat built by FractionalPoisson.get_x; "
    "residual reimplemented locally (same math) with tensor int_mat since ppsci's version "
    "passes numpy to sparse_coo_tensor, which fails under to_static; BATCH_SIZE counts the 100 collocation points + 1 BC point"
)
ALPHA = 1.8
NPOINT_INTERIOR = 100
BATCH_SIZE = NPOINT_INTERIOR + 1
KEYS = ("x", "y")
_STATE = {}


def _equation():
    """FractionalPoisson with its quadrature built on deterministic points, so the
    dynamic and CINN workers get the same int_mat without passing it around."""
    if "eq" not in _STATE:
        rng = np.random.RandomState(0)
        r = np.sqrt(rng.uniform(0, 0.99**2, NPOINT_INTERIOR))
        th = rng.uniform(0, 2 * np.pi, NPOINT_INTERIOR)
        pts = np.stack([r * np.cos(th), r * np.sin(th)], 1).astype("float32")
        eq = ppsci.equation.FractionalPoisson(
            ALPHA, ppsci.geometry.Disk((0, 0), 1), [8, 100]
        )
        _STATE["x"] = eq.get_x(pts)
        _STATE["eq"] = eq
    return _STATE["eq"]


def build_model():
    model = ppsci.arch.MLP(KEYS, ("u",), 4, 20, "tanh")

    def output_transform(in_, out):
        return {"u": (1 - (in_["x"] ** 2 + in_["y"] ** 2)) * out["u"]}

    model.register_output_transform(output_transform)
    return model


def build_batch():
    _equation()
    inputs = {k: paddle.to_tensor(v) for k, v in _STATE["x"].items()}
    inputs["bc_x"] = paddle.to_tensor([[-1.0]])
    inputs["bc_y"] = paddle.to_tensor([[0.0]])
    return inputs, {"bc_u": paddle.zeros([1, 1])}  # |1-(x^2+y^2)|^(1+a/2) at (-1, 0)


def prepare(model):
    # outside to_static: numpy/scipy quadrature setup; int_mat indices/values become
    # tensors because sparse_coo_tensor under to_static rejects numpy inputs
    eq = _equation()
    indices, values, shape = eq.int_mat
    _STATE["idx"] = paddle.to_tensor(np.array(indices, dtype="int64").T)
    _STATE["val"] = paddle.to_tensor(values)
    _STATE["shape"] = list(shape)
    a = ALPHA
    _STATE["c_lhs"] = float(
        special.gamma((1 - a) / 2) * special.gamma((2 + a) / 2) / (2 * np.pi**1.5)
    )
    _STATE["c_rhs"] = float(
        2**a * special.gamma(2 + a / 2) * special.gamma(1 + a / 2)
    )


def fpde_residual(out):
    """FractionalPoisson.equations["fpde"] with tensor int_mat inputs."""
    int_mat = sparse.sparse_coo_tensor(
        _STATE["idx"], _STATE["val"], _STATE["shape"], stop_gradient=False
    )
    lhs = sparse.matmul(int_mat, out["u"])[:, 0] * _STATE["c_lhs"]
    x = paddle.concat((out["x"], out["y"]), axis=1)[: _STATE["shape"][0]]
    rhs = _STATE["c_rhs"] * (1 - (1 + ALPHA / 2) * paddle.sum(x**2, axis=1))
    return lhs - rhs


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    data = C.take(inputs, KEYS)
    res = fpde_residual({**data, **model(data)})
    loss = C.sq_mean(res)
    loss += C.mse(model(C.take(inputs, KEYS, "bc_"))["u"], labels["bc_u"])
    return loss
