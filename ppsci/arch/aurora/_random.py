"""Explicit, independent random streams for ensemble forecasts."""

import paddle
from paddle.base import core
from paddle.tensor.random import gaussian


def make_generator(seed: int):
    """Create an independent seed stream without reseeding Paddle globally.

    Paddle's public Generator constructor currently returns a device's shared
    generator. A standalone core generator provides independent per-member
    streams; each draw seeds a native Paddle Gaussian kernel on the target device.
    """
    generator = core.Generator()
    generator.manual_seed(seed)
    return generator


def randn(shape, device, dtype, generator=None):
    if generator is None:
        return paddle.randn(shape, device=device, dtype=dtype)
    seed = int(generator.random()) % (2**31 - 2) + 1
    return gaussian(shape, seed=seed, device=device, dtype=dtype)
