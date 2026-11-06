"""Suzuki-Miyaura yield predictor (examples/smc_reac/smc_reac.py)."""
from cases import _common as C

import ppsci

EXAMPLE = "smc_reac"
NOTE = (
    "IterableNamedArrayDataset feeds the whole train split per step (batch_size 8 "
    "is ignored); split size guessed as 1024 rows; inputs are 5 x 2048-bit Morgan "
    "fingerprints (binary) generated directly instead of via rdkit"
)
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4


def build_model():
    return ppsci.arch.SuzukiMiyauraModel(
        input_dim=2048,
        output_dim=1,
        hidden_dim=512,
        hidden_dim2=1024,
        hidden_dim3=2048,
        hidden_dim4=1024,
    )


def build_batch():
    v = (C.uniform([BATCH_SIZE, 5 * 2048]) < 0.03).astype("float32")
    return {"v": v}, {"u": C.uniform([BATCH_SIZE, 1], 0.0, 100.0)}


def compute_loss(model, inputs, labels):
    return C.mse(model(inputs)["u"], labels["u"])
