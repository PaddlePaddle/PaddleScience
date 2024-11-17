"""Module for initializing tensor decompositions
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import math

import numpy as np
import paddle
import tensorly as tl

tl.set_backend("paddle")


def tensor_init(tensor, std=0.02):
    """Initializes directly the parameters of a factorized tensor so the reconstruction has the specified standard deviation and 0 mean

    Parameters
    ----------
    tensor : torch.Tensor or FactorizedTensor
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor
    """
    from .factorized_tensors import FactorizedTensor

    if isinstance(tensor, FactorizedTensor):
        tensor.normal_(0, std)
    elif paddle.is_tensor(tensor):
        tensor.normal_(0, std)
    else:
        raise ValueError(
            f"Got tensor of class {tensor.__class__.__name__} but expected torch.Tensor or FactorizedWeight."
        )


def cp_init(cp_tensor, std=0.02):
    """Initializes directly the weights and factors of a CP decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    cp_tensor : CPTensor
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor

    Notes
    -----
    We assume the given (weights, factors) form a correct CP decomposition, no checks are done here.
    """
    rank = cp_tensor.rank  # We assume we are given a valid CP
    order = cp_tensor.orders
    std_factors = (std / math.sqrt(rank)) ** (1 / order)

    with paddle.no_grad():
        cp_tensor.weights.fill_(1)
        for factor in cp_tensor.factors:
            factor.normal_(0, std_factors)
    return cp_tensor


def tucker_init(tucker_tensor, std=0.02):
    """Initializes directly the weights and factors of a Tucker decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    tucker_tensor : TuckerTensor
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor

    Notes
    -----
    We assume the given (core, factors) form a correct Tucker decomposition, no checks are done here.
    """
    order = tucker_tensor.order
    rank = tucker_tensor.rank
    r = np.prod([math.sqrt(r) for r in rank])
    std_factors = (std / r) ** (1 / (order + 1))
    with paddle.no_grad():
        tucker_tensor.core.normal_(0, std_factors)
        for factor in tucker_tensor.factors:
            factor.normal_(0, std_factors)
    return tucker_tensor


def tt_init(tt_tensor, std=0.02):
    """Initializes directly the weights and factors of a TT decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    tt_tensor : TTTensor
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor

    Notes
    -----
    We assume the given factors form a correct TT decomposition, no checks are done here.
    """
    order = tt_tensor.order
    r = np.prod(tt_tensor.rank)
    std_factors = (std / r) ** (1 / order)
    with paddle.no_grad():
        for factor in tt_tensor.factors:
            factor.normal_(0, std_factors)
    return tt_tensor


def block_tt_init(block_tt, std=0.02):
    """Initializes directly the weights and factors of a BlockTT decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    block_tt : Matrix in the tensor-train format
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor

    Notes
    -----
    We assume the given factors form a correct Block-TT decomposition, no checks are done here.
    """
    return tt_init(block_tt, std=std)
