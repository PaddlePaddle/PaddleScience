from typing import Any

import math

import paddle
from paddle import Tensor
from paddle.nn.initializer import Orthogonal

def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)

def kaiming_uniform(value: Any, fan: int, a: float):
    if isinstance(value, Tensor):
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            kaiming_uniform(v, fan, a)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            kaiming_uniform(v, fan, a)

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.shape[-2] + value.shape[-1]))
        value = paddle.uniform(value.shape, value.dtype, -stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        tensor = paddle.create_parameter(tensor.shape, attr=Orthogonal())
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value = paddle.full(value.shape, fill_value, value.dtype)
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
        value.data.normal_(mean, std)
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
