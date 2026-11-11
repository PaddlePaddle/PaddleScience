"""deepcfd UNetEx flow-field surrogate (examples/deepcfd/deepcfd.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "deepcfd"
NOTE = (
    "synthetic [B,3,172,79] input/label; CHANNELS_WEIGHTS = per-channel label RMS of "
    "the batch, not the full dataset; harness Adam has no weight_decay=0.005"
)
BATCH_SIZE = 64
SHAPE = [BATCH_SIZE, 3, 172, 79]


def build_model():
    return ppsci.arch.UNetEx(
        input_key="input",
        output_key="output",
        in_channel=3,
        out_channel=3,
        kernel_size=5,
        filters=[8, 16, 32, 32],
        weight_norm=False,
        batch_norm=False,
    )


def build_batch():
    return {"input": C.randn(SHAPE)}, {"output": C.randn(SHAPE)}


def compute_loss(model, inputs, labels):
    y = labels["output"]
    weights = paddle.sqrt(paddle.mean(y**2, axis=[0, 2, 3])).reshape([1, -1, 1, 1])
    out = model(inputs)["output"]
    loss_u = (out[:, 0:1] - y[:, 0:1]) ** 2
    loss_v = (out[:, 1:2] - y[:, 1:2]) ** 2
    loss_p = (out[:, 2:3] - y[:, 2:3]).abs()
    return ((loss_u + loss_v + loss_p) / weights).sum()
