"""fsi vortex-induced vibration with learnable k1/k2 (examples/fsi/viv.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "fsi"
NOTE = (
    "random t_f∈[0.0625,9.9375], eta/f labels in ±0.17/±0.34 instead of the .mat file; "
    "constant lr 1e-3 instead of Step schedule"
)
BATCH_SIZE = 100


class VIVNet(paddle.nn.Layer):
    """MLP plus the Vibration equation's learnable k1/k2 so they are trained and saved."""

    def __init__(self):
        super().__init__()
        self.mlp = ppsci.arch.MLP(("t_f",), ("eta",), 5, 50, "tanh")
        self.equation = ppsci.equation.Vibration(2, -4, 0)
        self.k1, self.k2 = self.equation.k1, self.equation.k2

    def forward(self, inputs):
        return self.mlp(inputs)


def build_model():
    return VIVNet()


def build_batch():
    inputs = {"t_f": C.uniform([BATCH_SIZE, 1], 0.0625, 9.9375, requires_grad=True)}
    labels = {
        "eta": C.uniform([BATCH_SIZE, 1], -0.175, 0.175),
        "f": C.uniform([BATCH_SIZE, 1], -0.34, 0.34),
    }
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    eq = model.equation
    f = ppsci.lambdify(
        eq.equations["f"], model.mlp, eq.learnable_parameters, fuse_derivative=True
    )
    EQUATIONS["f"] = f


def compute_loss(model, inputs, labels):
    out = model(inputs)
    f = EQUATIONS["f"]({**inputs, **out})
    C.autodiff.clear()
    return 100.0 * C.mse(out["eta"], labels["eta"]) + C.mse(f, labels["f"])
