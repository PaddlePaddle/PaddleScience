"""
Diffusion process implementation for TurbDiff model in PaddleScience.
"""

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def linear_beta_schedule(timesteps):
    """
    Linear schedule, proposed in original DDPM paper.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return paddle.linspace(beta_start, beta_end, timesteps, dtype=paddle.float32)


def log_linear_beta_schedule(timesteps):
    """
    A version of the linear beta schedule that works for arbitrary timesteps.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = paddle.linspace(beta_start, beta_end, timesteps, dtype=paddle.float32)
    
    # Map step indices from [0, T-1] to [-1, 1]
    t = paddle.linspace(-1, 1, timesteps, dtype=paddle.float32)
    
    # Define the mapping function
    a = 3
    # Make the log-slope more shallow for small indices
    slope = (1 + t) ** a
    # Normalize to maintain the same beta sum
    slope = slope / paddle.sum(slope) * timesteps
    
    # Map to the corresponding beta value
    mapped_indices = slope.cumsum(0) - 1
    mapped_indices = paddle.clip(mapped_indices, 0, timesteps - 1).astype(paddle.int64)
    
    return paddle.gather(betas, mapped_indices)


def log_snr_linear_beta_schedule(timesteps, snr_1=1e3, snr_T=1e-5):
    """
    A beta schedule that decays the log-SNR linearly.
    """
    log_snr_1 = math.log(snr_1)
    log_snr_T = math.log(snr_T)
    
    # Linear schedule in log-SNR space
    log_snr = paddle.linspace(log_snr_1, log_snr_T, timesteps, dtype=paddle.float32)
    
    # Convert log-SNR to alpha_cumprod using the formula
    # log(snr) = log(alpha_cumprod / (1 - alpha_cumprod))
    alphas_cumprod = paddle.exp(log_snr) / (1 + paddle.exp(log_snr))
    
    # Get betas from alphas_cumprod
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], [1, 0], value=1.0)
    alphas = alphas_cumprod / alphas_cumprod_prev
    betas = 1 - alphas
    
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = paddle.linspace(0, timesteps, steps, dtype=paddle.float32) / timesteps
    alphas_cumprod = paddle.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return paddle.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Sigmoid schedule proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    """
    steps = timesteps + 1
    t = paddle.linspace(0, timesteps, steps, dtype=paddle.float32) / timesteps
    v_start = paddle.sigmoid(paddle.to_tensor(start / tau, dtype=paddle.float32))
    v_end = paddle.sigmoid(paddle.to_tensor(end / tau, dtype=paddle.float32))
    alphas_cumprod = paddle.sigmoid((t * (end - start) + start) / tau)
    alphas_cumprod = (alphas_cumprod - v_start) / (v_end - v_start)
    alphas_cumprod = paddle.clip(alphas_cumprod, clamp_min, 1.0)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )
    return kl


def normal_log_lk(x, mean, log_var):
    """Log-likelihood of x under the given normal distribution."""
    log_2pi = math.log(2 * math.pi)
    return -0.5 * (log_2pi + log_var + (x - mean) ** 2 * paddle.exp(-log_var))


class GaussianDiffusion(nn.Layer):
    """
    Gaussian diffusion model for 3D turbulence.
    """
    
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        loss_type="l2",
        beta_schedule="sigmoid",
        clip_denoised=False,
        noise_bcs=False,
        learned_variances=False,
        elbo_weight=None,
        detach_elbo_mean=True,
    ):
        super().__init__()
        
        self.model = model
        self.clip_denoised = clip_denoised
        self.noise_bcs = noise_bcs
        self.learned_variances = learned_variances
        self.elbo_weight = elbo_weight
        self.detach_elbo_mean = detach_elbo_mean
        
        # Set up beta schedule
        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "log-linear":
            beta_schedule_fn = log_linear_beta_schedule
        elif beta_schedule == "log-snr-linear":
            beta_schedule_fn = log_snr_linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")
        
        betas = beta_schedule_fn(timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = paddle.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], [1, 0], value=1.0)
        
        self.num_timesteps = timesteps
        self.loss_type = loss_type
        
        # Register buffers for diffusion process
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", paddle.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", paddle.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", 1.0 / paddle.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", paddle.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("log_betas", paddle.log(betas))
        
        # Posterior log var - numerically stable version
        posterior_log_var = (
            self.log_betas
            + paddle.log1p(-alphas_cumprod_prev)
            - paddle.log1p(-alphas_cumprod)
        )
        
        # Adjust the first timestep to avoid -inf
        posterior_log_var[0] = self.log_betas[0] * (
            posterior_log_var[1] / self.log_betas[1]
        )
        self.register_buffer("posterior_log_var", posterior_log_var)
        
        self.register_buffer(
            "posterior_mean_coef1",
            betas * paddle.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * paddle.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from noise."""
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from x_0."""
        return (
            (self.sqrt_recip_alphas_cumprod[t] * x_t - x0)
            / self.sqrt_recipm1_alphas_cumprod[t]
        )
    
    def q_posterior(self, x_start, x_t, t):
        """Compute the posterior mean and log-variance."""
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_log_var = self.posterior_log_var[t]
        return posterior_mean, posterior_log_var
    
    def model_predictions(self, x_t, t, C, cell_idx, clip_x_start=False):
        """Get model predictions for x_0 and noise."""
        model_output = self.model(x_t, t, C)
        
        if self.learned_variances:
            # Split the output into prediction and variance
            model_output, model_log_var = paddle.split(model_output, 2, axis=1)
            
            # Apply boundary conditions
            if cell_idx is not None:
                bc_mask = paddle.zeros_like(cell_idx, dtype=paddle.float32)
                bc_mask = paddle.where(cell_idx > 0, paddle.ones_like(bc_mask), bc_mask)
                bc_mask = bc_mask.unsqueeze(1).tile([1, model_output.shape[1], 1, 1, 1])
                model_log_var = paddle.where(bc_mask > 0, model_log_var, paddle.ones_like(model_log_var) * -20)
        else:
            model_log_var = self.posterior_log_var[t]
        
        # Get x_0 prediction
        x_start = self.predict_start_from_noise(x_t, t, model_output)
        
        if clip_x_start and self.clip_denoised:
            x_start = paddle.clip(x_start, -1.0, 1.0)
        
        # Get the mean for q(x_{t-1} | x_t, x_0)
        model_mean, _ = self.q_posterior(x_start, x_t, t)
        
        return model_output, x_start, model_mean, model_log_var
    
    def p_sample(self, x_t, t, C, cell_idx):
        """Sample from p(x_{t-1} | x_t)."""
        noise, x_start, model_mean, model_log_var = self.model_predictions(
            x_t, t, C, cell_idx, clip_x_start=True
        )
        
        # No noise when t == 0
        nonzero_mask = paddle.cast(t > 0, dtype=paddle.float32)
        nonzero_mask = nonzero_mask.reshape([-1, 1, 1, 1, 1])
        
        # Sample from the predicted distribution
        noise = paddle.randn(x_t.shape, dtype=paddle.float32)
        
        # Apply boundary conditions to noise
        if cell_idx is not None:
            # If cell_idx > 0, it's a boundary cell, so we don't add noise
            bc_mask = paddle.zeros_like(cell_idx, dtype=paddle.float32)
            bc_mask = paddle.where(cell_idx > 0, paddle.ones_like(bc_mask), bc_mask)
            bc_mask = bc_mask.unsqueeze(1).tile([1, noise.shape[1], 1, 1, 1])
            
            if not self.noise_bcs:
                noise = paddle.where(bc_mask > 0, paddle.zeros_like(noise), noise)
        
        sample = model_mean + nonzero_mask * paddle.exp(0.5 * model_log_var) * noise
        
        return sample
    
    def p_sample_loop(self, x_bcs, C, cell_idx, pbar=False, start_from=None):
        """Run the reverse diffusion process to generate samples."""
        # Start from pure noise
        b = x_bcs.shape[0]
        sample = paddle.randn(x_bcs.shape, dtype=paddle.float32)
        
        # Apply boundary conditions from the start
        if cell_idx is not None:
            bc_mask = paddle.zeros_like(cell_idx, dtype=paddle.float32)
            bc_mask = paddle.where(cell_idx > 0, paddle.ones_like(bc_mask), bc_mask)
            bc_mask = bc_mask.unsqueeze(1).tile([1, sample.shape[1], 1, 1, 1])
            sample = paddle.where(bc_mask > 0, x_bcs, sample)
        
        # Choose starting timestep
        timesteps = self.num_timesteps
        if start_from is not None:
            timesteps = min(timesteps, start_from)
        
        # Iterate through all timesteps (or from start_from)
        time_range = list(reversed(range(0, timesteps)))
        if pbar:
            # If pbar is True, we would use tqdm here in PyTorch, but for simplicity
            # we'll just use the range directly in this implementation
            pass
        
        for i in time_range:
            t = paddle.full([b], i, dtype=paddle.int64)
            sample = self.p_sample(sample, t, C, cell_idx)
            
            # Apply boundary conditions at each step
            if cell_idx is not None:
                bc_mask = paddle.zeros_like(cell_idx, dtype=paddle.float32)
                bc_mask = paddle.where(cell_idx > 0, paddle.ones_like(bc_mask), bc_mask)
                bc_mask = bc_mask.unsqueeze(1).tile([1, sample.shape[1], 1, 1, 1])
                sample = paddle.where(bc_mask > 0, x_bcs, sample)
        
        return sample
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = paddle.randn(x_start.shape, dtype=paddle.float32)
        
        return (
            self.sqrt_alphas_cumprod[t].reshape([-1, 1, 1, 1, 1]) * x_start
            + self.sqrt_one_minus_alphas_cumprod[t].reshape([-1, 1, 1, 1, 1]) * noise
        )
    
    def loss_fn(self):
        """Get the appropriate loss function."""
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")
    
    def p_losses(self, x_start, t, C, cell_idx, cell_mask=None):
        """Compute training losses."""
        # Generate random noise
        noise = paddle.randn(x_start.shape, dtype=paddle.float32)
        
        # Apply boundary conditions to noise
        if cell_idx is not None and not self.noise_bcs:
            bc_mask = paddle.zeros_like(cell_idx, dtype=paddle.float32)
            bc_mask = paddle.where(cell_idx > 0, paddle.ones_like(bc_mask), bc_mask)
            bc_mask = bc_mask.unsqueeze(1).tile([1, noise.shape[1], 1, 1, 1])
            noise = paddle.where(bc_mask > 0, paddle.zeros_like(noise), noise)
        
        # Get noisy samples
        x_t = self.q_sample(x_start, t, noise)
        
        # Get model predictions
        model_output, x_0_pred, model_mean, model_log_var = self.model_predictions(
            x_t, t, C, cell_idx
        )
        
        # Basic loss term
        loss_fn = self.loss_fn()
        
        if self.learned_variances:
            # If learning variances, predict noise
            target = noise
        else:
            # Otherwise, directly predict x_0
            target = x_start
            model_output = x_0_pred
        
        # Compute the basic loss
        loss = loss_fn(model_output, target, reduction="none")
        
        # Apply cell mask if provided
        if cell_mask is not None:
            cell_mask = cell_mask.unsqueeze(1).tile([1, loss.shape[1], 1, 1, 1])
            loss = loss * cell_mask
        
        # Reduce over non-batch dimensions
        loss = paddle.mean(loss, axis=[1, 2, 3, 4])
        
        # Add ELBO term if specified
        if self.elbo_weight is not None:
            # Compute KL divergence for learned variances
            # Between the model posterior and the true posterior
            true_mean, true_log_var = self.q_posterior(x_start, x_t, t)
            if self.detach_elbo_mean:
                true_mean = true_mean.detach()
            kl = normal_kl(model_mean, model_log_var, true_mean, true_log_var)
            kl = paddle.mean(kl, axis=[1, 2, 3, 4])
            
            # Compute log likelihood of x_start under the posterior
            decoder_nll = -normal_log_lk(x_start, model_mean, model_log_var)
            decoder_nll = paddle.mean(decoder_nll, axis=[1, 2, 3, 4])
            
            # At t=0, use the decoder NLL,
            # otherwise use the KL divergence
            mask_t0 = (t == 0).astype(paddle.float32)
            mask_not_t0 = 1 - mask_t0
            
            elbo_loss = mask_t0 * decoder_nll + mask_not_t0 * kl
            loss = loss + self.elbo_weight * elbo_loss
        
        return loss
    
    def forward(self, x, C, cell_idx=None, cell_mask=None):
        """Forward pass for training."""
        b = x.shape[0]
        t = paddle.randint(0, self.num_timesteps, [b], dtype=paddle.int64)
        loss = self.p_losses(x, t, C, cell_idx, cell_mask)
        return paddle.mean(loss), t
