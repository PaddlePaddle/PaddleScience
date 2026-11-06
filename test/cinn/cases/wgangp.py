"""wgangp conditional WGAN-GP on CIFAR-10 (examples/wgangp/wgangp_cifar10.py)."""
import sys
from pathlib import Path

import paddle
import paddle.nn.functional as F
from cases import _common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "wgangp"))
import wgangp_cifar10_model as M  # noqa: E402

EXAMPLE = "wgangp"
NOTE = (
    "generator + discriminator in one Layer, loss = loss_d (WGAN + gradient penalty via "
    "paddle.grad create_graph=True + ACGAN CE) + loss_g, each with its own generator "
    "forward as in the 2 Solvers; one Adam (default betas instead of 0/0.9, no linear lr "
    "decay) instead of G/D Adams with 5 D iters per G iter, so loss_d grads also reach G "
    "(the example builds them too but discards them); noise and GP alpha come from the "
    "batch; running-stat updates in ConditionalBatchNorm/LayerNorm use detach()+assign "
    "instead of .numpy() (not traceable); real data uniform in [-1, 1]"
)
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
ACGAN_SCALE_G, ACGAN_SCALE = 0.1, 1.0


def _cbn_forward(self, x, labels):  # ConditionalBatchNorm.forward without .numpy()
    if self.training:
        mean = paddle.mean(x, axis=[0, 2, 3], keepdim=True)
        var = paddle.sum(paddle.square(x - mean), [0, 2, 3], keepdim=True) / (
            x.shape[0] * x.shape[2] * x.shape[3] - 1
        )
        m = self.momentum
        paddle.assign((1 - m) * self.mean + m * mean.squeeze().detach(), self.mean)
        paddle.assign((1 - m) * self.var + m * var.squeeze().detach(), self.var)
    else:
        mean = self.mean.reshape([1, -1, 1, 1])
        var = self.var.reshape([1, -1, 1, 1])
    scale = paddle.gather(self.scale, labels, axis=0).unsqueeze(-1).unsqueeze(-1)
    offset = paddle.gather(self.offset, labels, axis=0).unsqueeze(-1).unsqueeze(-1)
    return (x - mean) / paddle.sqrt(var + self.eps) * scale + offset


def _ln_forward(self, x):  # LayerNorm.forward without .numpy()
    if self.training:
        mean = paddle.mean(x, axis=[1, 2, 3], keepdim=True)
        var = paddle.sum(paddle.square(x - mean), [1, 2, 3], keepdim=True) / (
            x.shape[1] * x.shape[2] * x.shape[3] - 1
        )
        m = self.momentum
        paddle.assign((1 - m) * self.mean + m * mean.mean().detach(), self.mean)
        paddle.assign((1 - m) * self.var + m * var.mean().detach(), self.var)
    else:
        mean, var = self.mean, self.var
    return (x - mean) / paddle.sqrt(var + self.eps) * self.scale + self.offset


M.ConditionalBatchNorm.forward = _cbn_forward
M.LayerNorm.forward = _ln_forward


class GAN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.gen = M.WganGpCifar10Generator(
            ("labels",), ("fake_data",), dim=128, output_dim=3072, label_num=10
        )
        self.dis = M.WganGpCifar10Discriminator(
            ("data", "labels"), ("disc_fake", "disc_acgan"), dim=128, label_num=10
        )

    def generate(
        self, labels, noise
    ):  # WganGpCifar10Generator.forward with given noise
        return self.gen.generator(labels, noise)


def build_model():
    return GAN()


def build_batch():
    inputs = {
        "labels": paddle.randint(0, 10, [BATCH_SIZE], dtype="int32"),
        "noise_d": C.randn([BATCH_SIZE, 128]),
        "noise_g": C.randn([BATCH_SIZE, 128]),
        "alpha": C.uniform([BATCH_SIZE, 1]),
    }
    return inputs, {"real_data": C.uniform([BATCH_SIZE, 3072], -1.0, 1.0)}


def forward(model, inputs):
    fake = model.generate(inputs["labels"], inputs["noise_g"])
    return {"fake_data": fake, **model.dis({"data": fake, "labels": inputs["labels"]})}


def loss_gen(dis, fake, labels):  # Cifar10GenFuncs.loss
    out = dis({"data": fake, "labels": labels})
    cost = -paddle.mean(out["disc_fake"])
    return cost + ACGAN_SCALE_G * F.cross_entropy(out["disc_acgan"], labels)


def loss_dis(dis, fake, real, labels, alpha):  # Cifar10DisFuncs.loss
    disc_fake = dis({"data": fake, "labels": labels})["disc_fake"]
    out = dis({"data": real, "labels": labels})
    interp = real + alpha * (fake - real)
    g = paddle.grad(
        dis({"data": interp, "labels": labels})["disc_fake"],
        interp,
        create_graph=True,
        retain_graph=False,
    )[0]
    slopes = paddle.sqrt(paddle.sum(paddle.square(g), axis=1))
    gp = 10 * paddle.mean((slopes - 1.0) ** 2)
    cost = paddle.mean(disc_fake) - paddle.mean(out["disc_fake"]) + gp
    return cost + ACGAN_SCALE * F.cross_entropy(out["disc_acgan"], labels).sum()


def compute_loss(model, inputs, labels):
    y = inputs["labels"]
    fake_d = model.generate(y, inputs["noise_d"])
    loss = loss_dis(model.dis, fake_d, labels["real_data"], y, inputs["alpha"])
    return loss + loss_gen(model.dis, model.generate(y, inputs["noise_g"]), y)
