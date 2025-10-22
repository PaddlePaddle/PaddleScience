from typing import Callable

import paddle
from deepali.core import functional as U
from deepali.core.kernels import gaussian1d
from deepali.spatial import is_linear_transform

from .engine import RegistrationEngine
from .engine import RegistrationResult

RegistrationEvalHook = Callable[
    [RegistrationEngine, int, int, RegistrationResult], None
]
RegistrationStepHook = Callable[[RegistrationEngine, int, int, float], None]


def noop(reg: RegistrationEngine, *args, **kwargs) -> None:
    """Dummy no-op loss evaluation hook."""
    ...


def normalize_linear_grad(reg: RegistrationEngine, *args, **kwargs) -> None:
    """Loss evaluation hook for normalization of linear transformation gradient after backward pass."""
    denom = None
    for param in reg.model.parameters():
        if not param.stop_gradient and param.grad is not None:
            max_abs_grad = paddle.max(paddle.abs(param.grad))
            if denom is None or denom < max_abs_grad:
                denom = max_abs_grad
    if denom is None:
        return
    for param in reg.model.parameters():
        if not param.stop_gradient and param.grad is not None:
            param.grad /= denom


def normalize_nonrigid_grad(reg: RegistrationEngine, *args, **kwargs) -> None:
    """Loss evaluation hook for normalization of non-rigid transformation gradient after backward pass."""
    for param in reg.model.parameters():
        if not param.stop_gradient and param.grad is not None:
            paddle.assign(
                paddle.nn.functional.normalize(x=param.grad, p=2, axis=1),
                output=param.grad,
            )


def normalize_grad_hook(transform) -> RegistrationEvalHook:
    """Loss evaluation hook for normalization of transformation gradient after backward pass."""
    if is_linear_transform(transform):
        return normalize_linear_grad
    return normalize_nonrigid_grad


def _smooth_nonrigid_grad(reg: RegistrationEngine, sigma: float = 1) -> None:
    """Loss evaluation hook for Gaussian smoothing of non-rigid transformation gradient after backward pass."""
    if sigma <= 0:
        return
    kernel = gaussian1d(sigma)
    for param in reg.model.parameters():
        if not param.stop_gradient and param.grad is not None:
            param.grad.copy_(U.conv(param.grad, kernel))


def smooth_grad_hook(transform, sigma: float) -> RegistrationEvalHook:
    """Loss evaluation hook for Gaussian smoothing of non-rigid gradient after backward pass."""
    if is_linear_transform(transform):
        return noop

    def fn(reg: RegistrationEngine, *args, **kwargs):
        return _smooth_nonrigid_grad(reg, sigma=sigma)

    return fn
