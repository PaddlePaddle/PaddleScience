"""kmcast SR3 diffusion UNet for wind downscaling (examples/kmcast/conf/kmcast.yaml)."""
import sys
from pathlib import Path

import numpy as np
import paddle
from cases import _common as C

KMCAST_DIR = Path(__file__).resolve().parents[3] / "examples" / "kmcast"
sys.path.insert(0, str(KMCAST_DIR))
from model.networks import init_weights  # noqa: E402
from model.sr3_modules import diffusion  # noqa: E402
from model.sr3_modules import unet  # noqa: E402

EXAMPLE = "kmcast"
NOTE = (
    "network built directly as in model/networks.define_G (sr3 UNet + GaussianDiffusion, "
    "orthogonal init), bypassing model/model.py's own to_static(denoise_fn) wrapper; "
    "synthetic LR [B,2,192,256] in [0,1] / HR [B,1,192,256] (dataset crops 192x256, "
    "conf image_H=196 is only used by sampling); GaussianDiffusion.p_losses draws t, "
    "sqrt_alpha_cumprod and noise with np.random/paddle.randn inside, so build_batch "
    "draws them the same way (linear schedule, 2000 steps) and compute_loss reproduces "
    "p_losses + DDPM.optimize_parameters (L1 sum / b*c*h*w) from denoise_fn; inference "
    "is one denoise_fn forward, no sampling loop; dropout 0.2 disabled by the harness; "
    "infer rel diff ~2.3e-3 also appears with plain to_static (no CINN, no prim) and "
    "drops to ~4e-6 with NVIDIA_TF32_OVERRIDE=0, i.e. TF32 kernel choice differs between "
    "dynamic and static on A100, not a CINN error (FLAGS_allow_tf32_cudnn/cublas=0 "
    "do not remove it)"
)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
H, W = 192, 256
SCHEDULE = {
    "schedule": "linear",
    "n_timestep": 2000,
    "linear_start": 1e-6,
    "linear_end": 1e-2,
}


def build_model():
    denoise = unet.UNet(
        in_channel=3,
        out_channel=1,
        norm_groups=32,
        inner_channel=32,
        channel_mults=[1, 2, 4, 8, 8],
        res_blocks=2,
        dropout=0.2,
    )
    net = diffusion.GaussianDiffusion(
        denoise,
        image_H=196,
        image_W=256,
        channels=1,
        loss_type="l1",
        conditional=True,
        schedule_opt=SCHEDULE,
    )
    init_weights(net, init_type="orthogonal")
    return net


def build_batch():
    # same sampling as GaussianDiffusion.p_losses, but done once outside the graph
    betas = diffusion.make_beta_schedule(**SCHEDULE)
    sqrt_prev = np.sqrt(np.append(1.0, np.cumprod(1.0 - betas)))
    t = np.random.randint(1, SCHEDULE["n_timestep"] + 1)
    gamma = np.random.uniform(sqrt_prev[t - 1], sqrt_prev[t], size=BATCH_SIZE)
    inputs = {
        "LR": C.uniform([BATCH_SIZE, 2, H, W]),
        "HR": C.uniform([BATCH_SIZE, 1, H, W]),
        "noise": C.randn([BATCH_SIZE, 1, H, W]),
        "sqrt_alpha_cumprod": paddle.to_tensor(
            gamma.reshape([BATCH_SIZE, 1]), dtype="float32"
        ),
    }
    return inputs, {}


def forward(model, inputs):
    gamma = inputs["sqrt_alpha_cumprod"]
    x_noisy = model.q_sample(
        x_start=inputs["HR"],
        continuous_sqrt_alpha_cumprod=gamma.reshape([-1, 1, 1, 1]),
        noise=inputs["noise"],
    )
    return model.denoise_fn(paddle.concat(x=[inputs["LR"], x_noisy], axis=1), gamma)


def compute_loss(model, inputs, labels):
    x_recon = forward(model, inputs)
    b, c, h, w = inputs["HR"].shape
    return paddle.abs(inputs["noise"] - x_recon).sum() / int(b * c * h * w)
