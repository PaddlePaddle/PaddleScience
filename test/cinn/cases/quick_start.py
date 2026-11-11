"""quick_start case1: fit u=sin(x) on [-pi, pi] (examples/quick_start/case1.py)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "quick_start"
NOTE = "x sampled with paddle.uniform instead of Interval geometry"
BATCH_SIZE = 32
LEARNING_RATE = 2e-3


def build_model():
    return ppsci.arch.MLP(("x",), ("u",), 3, 64)


def build_batch():
    x = C.uniform([BATCH_SIZE, 1], -np.pi, np.pi)
    return {"x": x}, {"u": paddle.sin(x)}


def compute_loss(model, inputs, labels):
    return C.data_loss(model(inputs), labels)
