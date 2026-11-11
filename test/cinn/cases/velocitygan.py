"""velocityGAN full-waveform inversion, flatvel-a (examples/velocityGAN/velocityGAN.py)."""
import paddle
import paddle.nn.functional as F
from cases import _common as C

import ppsci

EXAMPLE = "velocityGAN"
NOTE = (
    "generator + discriminator in one Layer, loss = loss_g + loss_d (incl. gradient "
    "penalty via paddle.grad create_graph=True), one Adam instead of separate "
    "AdamW(wd=1e-4) for G/D with 3 D iters per G iter; GP alpha comes from the batch; "
    "data/labels uniform in [-1, 1] (min-max normalized). Known issue: Generator crops "
    "with F.pad(x, [-5]*4), whose GPU backward is wrong (mis-strided, partly uninitialized) in "
    "dynamic and static mode alike, so train losses of the two runs drift apart"
)
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LAMBDA_G1V, LAMBDA_G2V, LAMBDA_ADV, LAMBDA_GP = 100.0, 0.0, 1.0, 10.0


class GAN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.gen = ppsci.arch.VelocityGenerator(
            ("data",), ("fake_image",), 32, 64, 128, 256, 512, sample_spatial=1.0
        )
        self.dis = ppsci.arch.VelocityDiscriminator(
            ("image",), ("score",), 32, 64, 128, 256
        )


def build_model():
    return GAN()


def build_batch():
    inputs = {
        "data": C.uniform([BATCH_SIZE, 5, 1000, 70], -1.0, 1.0),
        "alpha": C.uniform([BATCH_SIZE, 1, 1, 1]),
    }
    return inputs, {"real_image": C.uniform([BATCH_SIZE, 1, 70, 70], -1.0, 1.0)}


def forward(model, inputs):
    out = model.gen({"data": inputs["data"]})
    return {**out, **model.dis({"image": out["fake_image"]})}


def loss_gen(dis, pred, label):  # functions.GenFuncs.loss_func_gen
    loss = LAMBDA_G1V * F.l1_loss(pred, label) + LAMBDA_G2V * F.mse_loss(pred, label)
    return loss + LAMBDA_ADV * -paddle.mean(dis({"image": pred})["score"])


def loss_dis(dis, pred, label, alpha):  # functions.DisFuncs.loss_func_dis
    pred = pred.detach()
    interp = alpha * label + (1 - alpha) * pred
    interp.stop_gradient = False
    score = dis({"image": interp})["score"]
    g = paddle.grad(score, interp, create_graph=True, retain_graph=True)[0]
    g = g.reshape([g.shape[0], -1])
    gp = paddle.mean((paddle.norm(g, p=2, axis=1) - 1) ** 2)
    loss_real = paddle.mean(dis({"image": label})["score"])
    loss_fake = paddle.mean(dis({"image": pred})["score"])
    return -loss_real + loss_fake + gp * LAMBDA_GP


def compute_loss(model, inputs, labels):
    label = labels["real_image"]
    # the example runs a separate generator forward in each Solver
    loss = loss_dis(
        model.dis, model.gen(take_data(inputs))["fake_image"], label, inputs["alpha"]
    )
    return loss + loss_gen(model.dis, model.gen(take_data(inputs))["fake_image"], label)


def take_data(inputs):
    return {"data": inputs["data"]}
