from typing import Optional
from typing import Union

import paddle
from pkg_resources import parse_version

paddle.Generator = Union[
    paddle.framework.core.default_cuda_generator,
    paddle.framework.core.default_cpu_generator,
]


def multinomial(
    input: paddle.Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[paddle.Generator] = None,
    out: Optional = None,
) -> paddle.int64:
    """Sample from a multinomial probability distribution.

    Args:
        input: Input vector of shape ``(N,)`` or matrix ``(M, N)``.
        num_samples: Number of random samples to draw.
        replacement: Whether to sample with or without replacement.
        generator: Random number generator to use.
        out: Pre-allocated output tensor.

    Returns:
        Indices of random samples. When ``input`` is a vector, a vector of ``num_samples`` indices
        is returned. Otherwise, a matrix of shape ``(M, num_samples)`` is returned. When ``out``
        is given, the returned tensor is a reference to ``out``.

    """
    if input.ndim == 0 or input.ndim > 2:
        raise ValueError("multinomial() 'input' must be vector or matrix")
    num_candidates = input.shape[-1]
    if not replacement and num_candidates < num_samples:
        raise ValueError(
            "multinomial() 'num_samples' cannot be greater than number of categories"
        )
    if num_candidates > 2**24:
        impl = _multinomial
    else:
        impl = paddle.multinomial
        input = input.astype(dtype="float32")
    return impl(
        input, num_samples, replacement=replacement, generator=generator, out=out
    )


def _multinomial(
    input: paddle.Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[paddle.Generator] = None,
    out: Optional = None,
) -> paddle.int64:
    """Sample from a multinomial probability distribution.

    This function can be used for inputs of any size and is unlike ``paddle.multinomial`` not limited
    to 2**24 categories at the expense of a less efficient implementation.

    Args:
        input: Input vector of shape ``(N,)`` or matrix ``(M, N)``.
        num_samples: Number of random samples to draw.
        replacement: Whether to sample with or without replacement.
        generator: Random number generator to use.
        out: Pre-allocated output tensor.

    Returns:
        Indices of random samples. When ``input`` is a vector, a vector of ``num_samples`` indices
        is returned. Otherwise, a matrix of shape ``(M, num_samples)`` is returned. When ``out``
        is given, the returned tensor is a reference to ``out``.

    """
    if input.ndim == 0 or input.ndim > 2:
        raise ValueError("_multinomial() 'input' must be vector or matrix")
    num_candidates = input.shape[-1]
    out_shape = tuple(input.shape)[:-1] + (num_samples,)
    if out is not None:
        if not isinstance(out, paddle.Tensor):
            raise TypeError("_multinomial() 'out' must be paddle.Tensor")
        if out.dtype != "int64":
            raise TypeError("_multinomial() 'out' must be int64 tensor")
        if tuple(out.shape) != out_shape:
            raise ValueError(f"_multinomial() 'out' must have shape {out_shape}")
    if replacement:
        cdf = input.astype("float64").cumsum(axis=-1)
        cdf = cdf.divide_(y=paddle.to_tensor(cdf[(...), -1:].clone()))
        val = paddle.rand(shape=out_shape, dtype=cdf.dtype)
        out = paddle.assign(
            paddle.searchsorted(sorted_sequence=cdf, values=val), output=out
        ).clip_(min=0, max=num_candidates - 1)
    else:
        if num_samples > num_candidates:
            raise ValueError(
                "_multinomial() 'num_samples' cannot be greater than number of categories"
            )
        logit = input.log()
        value = paddle.rand(
            shape=tuple(input.shape)[:-1] + (num_candidates,), dtype=logit.dtype
        )
        value = value.log_().neg_().log_().neg_().add_(y=paddle.to_tensor(logit))
        if parse_version(paddle.__version__) < parse_version("1.12"):
            _, index = paddle.topk(k=num_samples, sorted=False, x=value, axis=-1)
            out = index if out is None else paddle.assign(index, output=out)
        else:
            if out is None:
                out = paddle.empty(shape=out_shape, dtype="int64")
            _ = paddle.empty(shape=out_shape, dtype=value.dtype)
            out1, out2 = paddle.topk(k=num_samples, sorted=False, x=value, axis=-1)
            _, out = paddle.assign(out1, (_, out)[0]), paddle.assign(out2, (_, out)[1])
    return out
