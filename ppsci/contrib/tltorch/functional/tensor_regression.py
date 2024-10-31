from ..factorized_tensors import FactorizedTensor, TuckerTensor

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg

# Author: Jean Kossaifi
# License: BSD 3 clause


def trl(x, weight, bias=None, **kwargs):
    """Tensor Regression Layer

    Parameters
    ----------
    x : torch.tensor
        batch of inputs
    weight : FactorizedTensor
        factorized weights of the TRL
    bias : torch.Tensor, optional
        1D tensor, by default None

    Returns
    -------
    result
        input x contracted with regression weights
    """
    if isinstance(weight, TuckerTensor):
        return tucker_trl(x, weight, bias=bias, **kwargs)
    else:
        if bias is None:
            return tenalg.inner(x, weight.to_tensor(), n_modes=tl.ndim(x)-1)
        else:
            return tenalg.inner(x, weight.to_tensor(), n_modes=tl.ndim(x)-1) + bias


def tucker_trl(x, weight, project_input=False, bias=None):
        n_input = tl.ndim(x) - 1
        if project_input:
            x = tenalg.multi_mode_dot(x, weight.factors[:n_input], modes=range(1, n_input+1), transpose=True)
            regression_weights = tenalg.multi_mode_dot(weight.core, weight.factors[n_input:], 
                                                       modes=range(n_input, weight.order))
        else:
            regression_weights = weight.to_tensor()

        if bias is None:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1)
        else:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1) + bias

