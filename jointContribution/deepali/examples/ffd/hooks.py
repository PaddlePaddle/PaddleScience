r"""Hooks for iterative optimization-based registration engine."""
from typing import Callable

import paddle
from deepali.core import functional as U
from deepali.core.kernels import gaussian1d
from deepali.spatial import is_linear_transform
from deepali.utils import paddle_aux  # noqa

from .engine import RegistrationEngine
from .engine import RegistrationResult

RegistrationEvalHook = Callable[[RegistrationEngine, int, int, RegistrationResult], None]
RegistrationStepHook = Callable[[RegistrationEngine, int, int, float], None]


def noop(reg: RegistrationEngine, *args, **kwargs) -> None:
    r"""Dummy no-op loss evaluation hook."""
    ...


def normalize_linear_grad(reg: RegistrationEngine, *args, **kwargs) -> None:
    r"""Loss evaluation hook for normalization of linear transformation gradient after backward pass."""
    denom = None
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            max_abs_grad = p.grad.abs().max()
            if denom is None or denom < max_abs_grad:
                denom = max_abs_grad
    if denom is None:
        return
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            p.grad /= denom


def normalize_nonrigid_grad(reg: RegistrationEngine, *args, **kwargs) -> None:
    r"""Loss evaluation hook for normalization of non-rigid transformation gradient after backward pass."""
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            paddle.assign(paddle.nn.functional.normalize(x=p.grad, p=2, axis=1), output=p.grad)


def normalize_grad_hook(transform) -> RegistrationEvalHook:
    r"""Loss evaluation hook for normalization of transformation gradient after backward pass."""
    if is_linear_transform(transform):
        return normalize_linear_grad
    return normalize_nonrigid_grad


def _smooth_nonrigid_grad(reg: RegistrationEngine, sigma: float = 1) -> None:
    r"""Loss evaluation hook for Gaussian smoothing of non-rigid transformation gradient after backward pass."""
    if sigma <= 0:
        return
    kernel = gaussian1d(sigma)
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            p.grad.copy_(U.conv(p.grad, kernel))


def smooth_grad_hook(transform, sigma: float) -> RegistrationEvalHook:
    r"""Loss evaluation hook for Gaussian smoothing of non-rigid gradient after backward pass."""
    if is_linear_transform(transform):
        return noop

    def fn(reg: RegistrationEngine, *args, **kwargs):
        return _smooth_nonrigid_grad(reg, sigma=sigma)

    return fn
