import math
from inspect import isfunction

import numpy as np
import paddle
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas


def make_beta_schedule(
    schedule, n_timestep, linear_start=0.0001, linear_end=0.02, cosine_s=0.008
):
    if schedule == "quad":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            paddle.arange(dtype="float64", end=n_timestep + 1) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = paddle.cos(x=alphas).pow(y=2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clip(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(paddle.nn.Layer):
    def __init__(
        self,
        denoise_fn,
        image_H,
        image_W,
        channels=3,
        loss_type="l1",
        conditional=True,
        schedule_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.H, self.W = image_H, image_W
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == "l1":
            self.loss_func = paddle.nn.L1Loss(reduction="sum").to(device)
        elif self.loss_type == "l2":
            self.loss_func = paddle.nn.MSELoss(reduction="sum").to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt["linear_start"],
            linear_end=schedule_opt["linear_end"],
        )
        betas = (
            betas.detach().cpu().numpy() if isinstance(betas, paddle.Tensor) else betas
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))
        (timesteps,) = tuple(betas.shape)
        self.num_timesteps = int(timesteps)
        self.register_buffer(name="betas", tensor=paddle.to_tensor(betas))
        self.register_buffer(
            name="alphas_cumprod", tensor=paddle.to_tensor(alphas_cumprod)
        )
        self.register_buffer(
            name="alphas_cumprod_prev", tensor=paddle.to_tensor(alphas_cumprod_prev)
        )
        self.register_buffer(
            name="sqrt_alphas_cumprod", tensor=paddle.to_tensor(np.sqrt(alphas_cumprod))
        )
        self.register_buffer(
            name="sqrt_one_minus_alphas_cumprod",
            tensor=paddle.to_tensor(np.sqrt(1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            name="log_one_minus_alphas_cumprod",
            tensor=paddle.to_tensor(np.log(1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            name="sqrt_recip_alphas_cumprod",
            tensor=paddle.to_tensor(np.sqrt(1.0 / alphas_cumprod)),
        )
        self.register_buffer(
            name="sqrt_recipm1_alphas_cumprod",
            tensor=paddle.to_tensor(np.sqrt(1.0 / alphas_cumprod - 1)),
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            name="posterior_variance", tensor=paddle.to_tensor(posterior_variance)
        )
        self.register_buffer(
            name="posterior_log_variance_clipped",
            tensor=paddle.to_tensor(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            name="posterior_mean_coef1",
            tensor=paddle.to_tensor(
                betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            ),
        )
        self.register_buffer(
            name="posterior_mean_coef2",
            tensor=paddle.to_tensor(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = tuple(x.shape)[0]
        noise_level = (
            paddle.to_tensor(
                data=[self.sqrt_alphas_cumprod_prev[t + 1]], dtype="float32"
            )
            .tile(repeat_times=[batch_size, 1])
            .to("gpu:0")
        )
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x,
                t=t,
                noise=self.denoise_fn(
                    paddle.concat(
                        x=[condition_x.astype("float32"), x.astype("float32")], axis=1
                    ),
                    noise_level,
                ),
            )
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level)
            )
        if clip_denoised:
            x_recon.clip_(min=-1.0, max=1.0)
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_log_variance

    @paddle.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        noise = (
            paddle.randn(shape=x.shape, dtype=x.dtype)
            if t > 0
            else paddle.zeros_like(x=x)
        )
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @paddle.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        sample_inter = 1 | self.num_timesteps // 10
        if not self.conditional:
            shape = x_in
            img = paddle.randn(shape=shape)
            ret_img = img
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = paddle.concat(x=[ret_img, img], axis=0)
        else:
            shape = tuple(x_in["HR"].shape)
            img = paddle.randn(shape=shape)
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                img = self.p_sample(img, i, condition_x=x_in["LR"])
        return img

    @paddle.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_H, image_W = self.H, self.W
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_H, image_W), continous)

    @paddle.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(
            noise, lambda: paddle.randn(shape=x_start.shape, dtype=x_start.dtype)
        )
        return (
            continuous_sqrt_alpha_cumprod * x_start
            + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in["HR"]
        [b, c, h, w] = tuple(x_start.shape)
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = paddle.to_tensor(
            data=np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b,
            ),
            dtype="float32",
        ).to("gpu:0")
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.reshape([b, -1])
        noise = default(
            noise, lambda: paddle.randn(shape=x_start.shape, dtype=x_start.dtype)
        )
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.reshape(
                [-1, 1, 1, 1]
            ),
            noise=noise,
        )
        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                paddle.concat(x=[x_in["LR"], x_noisy], axis=1),
                continuous_sqrt_alpha_cumprod,
            )
        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
