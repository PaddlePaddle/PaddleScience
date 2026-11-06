"""deephpms Burgers, stage 2: PDE-discovery net (examples/deephpms/burgers.py, burgers.yaml)."""
import paddle
import paddle.nn.functional as F
from cases import _common as C

import ppsci
from ppsci.autodiff import hessian, jacobian

EXAMPLE = "deephpms"
NOTE = (
    "stage 2 only (idn net frozen via per-param lr=0, pde net trained on f_pde vs du_t); random t∈[0,10], "
    "x∈[-8,8] instead of burgers_sine.mat (N_train=10000 as in DeepHPM); infer runs idn on "
    "(t,x) and pde net's MLP body on random features because the derivative transform "
    "needs grad, which infer's no_grad forbids"
)
BATCH_SIZE = 10000
T_LB, T_UB, X_LB, X_UB = 0.0, 10.0, -8.0, 10.0  # X_UB = T_UB as in burgers.py


class DeepHPM(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.idn = ppsci.arch.MLP(("t", "x"), ("u_idn",), 4, 50, "sin")
        self.pde = ppsci.arch.MLP(("u_x", "du_x", "du_xx"), ("f_pde",), 2, 100, "sin")

        def transform_u(_in):
            t = 2.0 * (_in["t"] - T_LB) * (T_UB - T_LB) ** -1 - 1.0
            x = 2.0 * (_in["x"] - X_LB) * (X_UB - X_LB) ** -1 - 1.0
            return {"t": t, "x": x}

        def transform_f_idn(_in):
            x = _in["x"]
            u = self.idn({"t": _in["t"], "x": x})["u_idn"]
            return {"u_x": u, "du_x": jacobian(u, x), "du_xx": hessian(u, x)}

        self.idn.register_input_transform(transform_u)
        self.pde.register_input_transform(transform_f_idn)
        # stage 2 only steps optimizer_pde; idn still gets grads, so freeze it via lr=0
        for p in self.idn.parameters():
            p.optimize_attr["learning_rate"] = 0.0

    def forward(self, inputs):
        return {**self.idn(inputs), **self.pde(inputs)}


def build_model():
    return DeepHPM()


def build_batch():
    inputs = {
        "t": C.uniform([BATCH_SIZE, 1], 0.0, 10.0, requires_grad=True),
        "x": C.uniform([BATCH_SIZE, 1], -8.0, 8.0, requires_grad=True),
        "feat": C.randn([BATCH_SIZE, 3]),
    }
    return inputs, {}


def forward(model, inputs):
    u = model.idn(C.take(inputs, ("t", "x")))
    return {**u, "f_pde": model.pde.forward_tensor(inputs["feat"])}


def compute_loss(model, inputs, labels):
    out = model(C.take(inputs, ("t", "x")))
    du_t = jacobian(out["u_idn"], inputs["t"])
    loss = F.mse_loss(out["f_pde"], du_t, "sum")
    C.autodiff.clear()
    return loss
