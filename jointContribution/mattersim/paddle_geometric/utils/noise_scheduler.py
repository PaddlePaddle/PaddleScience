import math
from typing import Literal, Optional

import paddle
from paddle import Tensor


def get_smld_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    num_scales: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tensor:
    r"""Generates a set of noise values on a logarithmic scale for "Score
    Matching with Langevin Dynamics" from the `"Generative Modeling by
    Estimating Gradients of the Data Distribution"
    <https://arxiv.org/abs/1907.05600>`_ paper.

    This function returns a vector of sigma values that define the schedule of
    noise levels used during Score Matching with Langevin Dynamics.
    The sigma values are determined on a logarithmic scale from
    :obj:`sigma_max` to :obj:`sigma_min`, inclusive.

    Args:
        sigma_min (float): The minimum value of sigma, corresponding to the
            lowest noise level.
        sigma_max (float): The maximum value of sigma, corresponding to the
            highest noise level.
        num_scales (int): The number of sigma values to generate, defining the
            granularity of the noise schedule.
        dtype (paddle.dtype, optional): The output data type.
            (default: :obj:`None`)
        device (str, optional): The output device.
            (default: :obj:`None`)
    """
    log_sigma_max = math.log(sigma_max)
    log_sigma_min = math.log(sigma_min)
    return paddle.exp(paddle.linspace(
        log_sigma_max,
        log_sigma_min,
        num_scales,
        dtype=dtype,
    ))


def get_diffusion_beta_schedule(
    schedule_type: Literal['linear', 'quadratic', 'constant', 'sigmoid'],
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tensor:
    r"""Generates a schedule of beta values according to the specified strategy
    for the diffusion process from the `"Denoising Diffusion Probabilistic
    Models" <https://arxiv.org/abs/2006.11239>`_ paper.

    Beta values are used to scale the noise added during the diffusion process
    in generative models. This function creates an array of beta values
    according to a pre-defined schedule, which can be either :obj:`"linear"`,
    :obj:`"quadratic"`, :obj:`"constant"`, or :obj:`"sigmoid"`.

    Args:
        schedule_type (str): The type of schedule to use for beta values.
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        num_diffusion_timesteps (int): The number of timesteps for the
            diffusion process.
        dtype (paddle.dtype, optional): The output data type.
            (default: :obj:`None`)
        device (str, optional): The output device.
            (default: :obj:`None`)
    """
    if schedule_type == 'linear':
        return paddle.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=dtype,
        )

    if schedule_type == 'quadratic':
        return paddle.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_diffusion_timesteps,
            dtype=dtype,
        )**2

    if schedule_type == 'constant':
        return paddle.full(
            shape=[num_diffusion_timesteps],
            fill_value=beta_end,
            dtype=dtype,
        )

    if schedule_type == 'sigmoid':
        return paddle.linspace(
            -6,
            6,
            num_diffusion_timesteps,
            dtype=dtype,
        ).sigmoid() * (beta_end - beta_start) + beta_start

    raise ValueError(f"Found invalid 'schedule_type' (got '{schedule_type}')")
