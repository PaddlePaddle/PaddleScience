"""DeepONet antiderivative operator (examples/operator_learning/deeponet.py)."""
from cases import _common as C

import ppsci

EXAMPLE = "operator_learning"
NOTE = "IterableNPZDataset full batch: u [10000, 100], y [10000, 1], G [10000, 1] synthetic"
BATCH_SIZE = 10000


def build_model():
    return ppsci.arch.DeepONet(
        u_key="u",
        y_key="y",
        G_key="G",
        num_loc=100,
        num_features=40,
        branch_num_layers=1,
        trunk_num_layers=1,
        branch_hidden_size=40,
        trunk_hidden_size=40,
        branch_activation="relu",
        trunk_activation="relu",
        use_bias=True,
    )


def build_batch():
    inputs = {"u": C.randn([BATCH_SIZE, 100]), "y": C.uniform([BATCH_SIZE, 1])}
    return inputs, {"G": C.randn([BATCH_SIZE, 1])}


def compute_loss(model, inputs, labels):
    return C.mse(model(inputs)["G"], labels["G"])
