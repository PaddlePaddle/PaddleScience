"""iops air-pollution classifier (examples/iops/iops.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "iops"
BATCH_SIZE = 32


def build_model():
    return ppsci.arch.MLP(
        input_keys=("input",),
        output_keys=("output",),
        input_dim=5,
        output_dim=6,
        num_layers=3,
        hidden_size=64,
        activation="ReLU",
    )


def build_batch():
    return {"input": C.randn([BATCH_SIZE, 5])}, {"output": C.randint([BATCH_SIZE], 6)}


def compute_loss(model, inputs, labels):
    out = model(inputs)
    return paddle.nn.functional.cross_entropy(out["output"], labels["output"])
