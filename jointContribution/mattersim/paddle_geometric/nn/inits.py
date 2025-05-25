import math
from typing import Any

import paddle
from paddle import Tensor


def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        paddle.nn.initializer.Uniform(-bound, bound)(value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)


def kaiming_uniform(value: Any, fan: int, a: float):
    if isinstance(value, Tensor):
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        paddle.nn.initializer.Uniform(-bound, bound)(value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            kaiming_uniform(v, fan, a)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            kaiming_uniform(v, fan, a)


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.shape[-2] + value.shape[-1]))
        paddle.nn.initializer.Uniform(-stdv, stdv)(value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        init_orthogonal = paddle.nn.initializer.Orthogonal()
        init_orthogonal(tensor)
        scale /= ((tensor.shape[-2] + tensor.shape[-1]) * tensor.var())
        tensor *= paddle.sqrt(scale)


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        paddle.nn.initializer.Constant(fill_value)(value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def ones(tensor: Any):
    constant(tensor, 1.)


def normal(value: Any, mean: float, std: float):
    if isinstance(value, Tensor):
        paddle.nn.initializer.Normal(mean, std)(value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            normal(v, mean, std)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            normal(v, mean, std)


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)
