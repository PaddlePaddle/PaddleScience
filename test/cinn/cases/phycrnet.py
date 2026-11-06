"""phycrnet PhyCRNet, 2D Burgers' equation (examples/phycrnet/main.py, burgers_equations.yaml)."""
import paddle
from cases import _common as C

import ppsci
from ppsci.arch import phycrnet

EXAMPLE = "phycrnet"
NOTE = (
    "time unroll reduced from TIME_BATCH_SIZE=1000 (1001 recurrent steps) to 20 so the "
    "full static graph stays compilable; synthetic IC uv[0:1]=randn [1,2,128,128]; "
    "constant lr instead of Step(50, 0.99) scheduler"
)
BATCH_SIZE = 1
LEARNING_RATE = 6.0e-4
DT, DX = 0.002, 1.0 / 128
TIME_BATCH_SIZE = 20  # example: 1000
STEPS = TIME_BATCH_SIZE + 1
LOSS_FUNC = []


def build_model():
    return ppsci.arch.PhyCRNet(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=DT,
        num_layers=[3, 1],
        upscale_factor=8,
        step=STEPS,
        effective_step=list(range(STEPS)),
    )


def build_batch():
    # num_convlstm=1: initial_state = [(h0, c0)], h0/c0 ~ randn(1,128,16,16)
    h0, c0 = C.randn([1, 128, 16, 16]), C.randn([1, 128, 16, 16])
    state = paddle.stack([paddle.stack([h0, c0])])  # [1, 2, 1, 128, 16, 16]
    return {"initial_state": state, "input": 0.1 * C.randn([1, 2, 128, 128])}, {}


def prepare(model):
    LOSS_FUNC.append(phycrnet.loss_generator(DT, DX))


def forward(model, inputs):
    return model(inputs)["outputs"]


def _phy_loss(output, loss_func):  # examples/phycrnet/functions.py::compute_loss
    output = paddle.concat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), axis=3)
    output = paddle.concat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), axis=2)
    f_u, f_v = loss_func.get_phy_Loss(output)
    return C.mse(f_u, paddle.zeros_like(f_u)) + C.mse(f_v, paddle.zeros_like(f_v))


def compute_loss(model, inputs, labels):
    # transform_out with num_time_batch = 1
    out = model(inputs)
    output = paddle.concat(tuple(out["outputs"]), axis=0)
    output = paddle.concat((inputs["input"], output), axis=0)
    return _phy_loss(output, LOSS_FUNC[0])
