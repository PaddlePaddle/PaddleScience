"""preformer Preformer on ERA5 (examples/preformer/conf/preformer.yaml)."""
from cases import _common as C

import ppsci

EXAMPLE = "preformer"
NOTE = "synthetic ERA5 input/label of shape (T=6, C, 192, 256); drop_path is nn.Identity in the arch"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
SHAPE_IN = (6, 12, 192, 256)
NUM_CLASSES = 1


def build_model():
    return ppsci.arch.Preformer(
        input_keys=("input",), output_keys=("output",), shape_in=SHAPE_IN
    )


def build_batch():
    T, C_IN, H, W = SHAPE_IN
    return (
        {"input": C.randn([BATCH_SIZE, T, C_IN, H, W])},
        {"output": C.randn([BATCH_SIZE, T, NUM_CLASSES, H, W])},
    )


def compute_loss(model, inputs, labels):
    return C.data_loss(model(inputs), labels)
