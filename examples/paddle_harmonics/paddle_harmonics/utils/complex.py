import paddle
import numpy as np


# Temporary solution to the problem that 'paddle.norm' and 'paddle.sqrt' 
# operators in paddle do not support complex number operations

def norm_complex(x: paddle.Tensor, p=None, axis=None, keepdim=False, name=None):
    if x.dtype == paddle.complex64 or x.dtype == paddle.complex128:
        return paddle.norm(x.abs(), p, axis, keepdim, name)
    else:
        return paddle.norm(x, p, axis)


def sqrt_complex(x: paddle.Tensor, name=None):
    if x.dtype == paddle.complex64 or x.dtype == paddle.complex128:
        # TODO: May stop the gradient
        return paddle.to_tensor(np.sqrt(x.numpy()))
    else:
        return paddle.sqrt(x, name)
