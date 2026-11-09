"""TADF oscillator strength f regressor (examples/tadf/TADF_f/f_model.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "tadf"
NOTE = (
    "TADF_f sub-model; PCA(0.99) dim of Morgan fps taken as 587 (= first hidden "
    "size), one [N, 1] input key per PCA component as in the example; full train "
    "fold guessed as 512 rows (IterableNamedArrayDataset ignores batch_size 8); "
    "weight_decay via per-param L2Decay, Adam beta2 0.999 instead of 0.99"
)
IN_DIM = 587
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
KEYS = tuple(f"key_{i}" for i in range(IN_DIM))


def build_model():
    model = ppsci.arch.TADF(
        input_keys=KEYS,
        output_keys=("u",),
        hidden_size=[587, 256],
        num_layers=None,
        activation="relu",
        dropout=0.5,
    )
    for p in model.parameters():
        p.regularizer = paddle.regularizer.L2Decay(1e-5)
    return model


def build_batch():
    x = C.randn([BATCH_SIZE, IN_DIM])
    inputs = {k: x[:, i : i + 1].clone() for i, k in enumerate(KEYS)}
    return inputs, {"u": C.uniform([BATCH_SIZE, 1])}


def compute_loss(model, inputs, labels):
    return C.mse(model(inputs)["u"], labels["u"])
