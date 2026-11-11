"""RegAE variational autoencoder (examples/RegAE/RegAE.py)."""
import paddle
from cases import _common as C
from paddle.nn import functional as F

import ppsci

EXAMPLE = "RegAE"
NOTE = (
    "reparameterization noise is an input tensor 'eps' instead of paddle.randn "
    "inside AutoEncoder.forward_tensor; p_train synthetic in [-1, 1]"
)
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
INPUT_DIM, LATENT_DIM, HIDDEN_DIM = 10000, 100, 100


def build_model():
    return ppsci.arch.AutoEncoder(
        input_keys=("p_train",),
        output_keys=("mu", "log_sigma", "decoder_z"),
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    )


def build_batch():
    x = C.uniform([BATCH_SIZE, INPUT_DIM], -1.0, 1.0)
    return {"p_train": x, "eps": C.randn([BATCH_SIZE, LATENT_DIM])}, {"p_train": x}


def forward(model, inputs):
    # AutoEncoder.forward with the sampled noise supplied from the batch
    mu, log_sigma = model.encoder(inputs["p_train"])
    z = mu + inputs["eps"] * paddle.exp(log_sigma)
    return {"mu": mu, "log_sigma": log_sigma, "decoder_z": model.decoder(z)}


def compute_loss(model, inputs, labels):
    out = forward(model, inputs)
    mu, log_sigma = out["mu"], out["log_sigma"]
    base = paddle.exp(2.0 * log_sigma) + paddle.pow(mu, 2) - 1.0 - 2.0 * log_sigma
    kl = 0.5 * paddle.sum(base) / mu.shape[0]
    return F.mse_loss(out["decoder_z"], labels["p_train"]) + kl
