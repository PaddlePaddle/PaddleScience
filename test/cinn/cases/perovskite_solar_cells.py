"""perovskite solar cell Jsc regressor (examples/perovskite_solar_cells/psc_nn.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "perovskite_solar_cells"
NOTE = (
    "hidden sizes from conf model.hidden_size instead of the optuna search; "
    "Adam with constant lr instead of the warmup+ExponentialDecay schedule"
)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def build_model():
    return ppsci.arch.MLP(
        input_keys=("input",),
        output_keys=("target",),
        num_layers=None,
        hidden_size=[128, 96, 64, 32],
        activation="relu",
        input_dim=2808,
        output_dim=1,
    )


def build_batch():
    # Jsc labels are positive (mA/cm^2), the loss weights samples by label
    return {"input": C.randn([BATCH_SIZE, 2808])}, {
        "target": C.uniform([BATCH_SIZE, 1], 5.0, 25.0)
    }


def compute_loss(model, inputs, labels):
    pred, true = model(inputs)["target"], labels["target"]
    weights = true / (paddle.sum(true) + 1e-6)
    return paddle.sum((true - pred) ** 2 * weights) / true.shape[0]
