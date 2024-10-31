import numpy as np
import torch
import torch.nn.functional as F
from ..factorized_tensors import TensorizedTensor
from ..factorized_tensors.tensorized_matrices import CPTensorized, TuckerTensorized, BlockTT
from .factorized_linear import linear_blocktt, linear_cp, linear_tucker

import tensorly as tl
tl.set_backend('pytorch')

# Author: Jean Kossaifi
# License: BSD 3 clause


def factorized_linear(x, weight, bias=None, in_features=None, implementation="factorized"):
    """Linear layer with a dense input x and factorized weight
    """
    assert implementation in {"factorized", "reconstructed"}, f"Expect implementation from [factorized, reconstructed], but got {implementation}"
    
    if in_features is None:
        in_features = np.prod(x.shape[-1])

    if not torch.is_tensor(weight):
        # Weights are in the form (out_features, in_features) 
        # PyTorch's linear returns dot(x, weight.T)!
        if isinstance(weight, TensorizedTensor):
            if implementation == "factorized":
                x_shape = x.shape[:-1] + weight.tensorized_shape[1]
                out_shape = x.shape[:-1] + (-1, )
                if isinstance(weight, CPTensorized):
                    x = linear_cp(x.reshape(x_shape), weight).reshape(out_shape)
                    if bias is not None:
                        x = x + bias
                    return x
                elif isinstance(weight, TuckerTensorized):
                    x = linear_tucker(x.reshape(x_shape), weight).reshape(out_shape)
                    if bias is not None:
                        x = x + bias
                    return x
                elif isinstance(weight, BlockTT):
                    x = linear_blocktt(x.reshape(x_shape), weight).reshape(out_shape)
                    if bias is not None:
                        x = x + bias
                    return x
            # if no efficient implementation available or force to use reconstructed mode: use reconstruction
            weight = weight.to_matrix()
        else:
            weight = weight.to_tensor()
        
    return F.linear(x, torch.reshape(weight, (-1, in_features)), bias=bias)
