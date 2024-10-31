"""Tensor Dropout for TensorModules"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import tensorly as tl
tl.set_backend('pytorch')
import torch
from ..factorized_tensors import TuckerTensor, CPTensor, TTTensor


class TensorDropout():
    """Decomposition Hook for Tensor Dropout on FactorizedTensor

    Parameters
    ----------
    name : FactorizedTensor parameter on which to apply the dropout
    proba : float, probability of dropout
    min_dim : int
        Minimum dimension size for which to apply dropout.
        For instance, if a tensor if of shape (32, 32, 3, 3) and min_dim = 4
        then dropout will *not* be applied to the last two modes.
    """
    _factorizations = dict()

    def __init_subclass__(cls, factorization, **kwargs):
        """When a subclass is created, register it in _factorizations"""
        cls._factorizations[factorization.__name__] = cls

    def __init__(self, proba, min_dim=1, min_values=1, drop_test=False):
        assert 0 <= proba < 1, f'Got prob={proba} but tensor dropout is defined for 0 <= proba < 1.'
        self.proba = proba
        self.min_dim = min_dim
        self.min_values = min_values
        self.drop_test = drop_test

    def __call__(self, module, input, factorized_tensor):
        return self._apply_tensor_dropout(factorized_tensor, training=module.training)

    def _apply_tensor_dropout(self, factorized_tensor, training=True):
        raise NotImplementedError()

    @classmethod
    def apply(cls, module, proba, min_dim=3, min_values=1, drop_test=False):
        cls = cls._factorizations[module.__class__.__name__]
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, cls):
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter")

        dropout = cls(proba, min_dim=min_dim, min_values=min_values, drop_test=drop_test)
        handle = module.register_forward_hook(dropout)
        return handle


class TuckerDropout(TensorDropout, factorization=TuckerTensor):
    def _apply_tensor_dropout(self, tucker_tensor, training=True):
        if (not self.proba) or ((not training) and (not self.drop_test)):
            return tucker_tensor

        core, factors = tucker_tensor.core, tucker_tensor.factors
        tucker_rank = tucker_tensor.rank

        sampled_indices = []
        for rank in tucker_rank:
            idx = tl.arange(rank, device=core.device, dtype=torch.int64)
            if rank > self.min_dim:
                idx = idx[torch.bernoulli(torch.ones(rank, device=core.device)*(1 - self.proba),
                                      out=torch.empty(rank, device=core.device, dtype=torch.bool))]
                if len(idx) == 0:
                    idx = torch.randint(0, rank, size=(self.min_values, ), device=core.device, dtype=torch.int64)

            sampled_indices.append(idx)
        
        if training:
            core = core[torch.meshgrid(*sampled_indices)]*(1/((1 - self.proba)**core.ndim))
        else:
            core = core[torch.meshgrid(*sampled_indices)]

        factors = [factor[:, idx]  for (factor, idx) in zip(factors, sampled_indices)]

        return TuckerTensor(core, factors)

 
class CPDropout(TensorDropout, factorization=CPTensor):
    def _apply_tensor_dropout(self, cp_tensor, training=True):
        if (not self.proba) or ((not training) and (not self.drop_test)):
            return cp_tensor

        rank = cp_tensor.rank
        device = cp_tensor.factors[0].device
        
        if rank > self.min_dim:
            sampled_indices = tl.arange(rank, device=device, dtype=torch.int64)
            sampled_indices = sampled_indices[torch.bernoulli(torch.ones(rank, device=device)*(1 - self.proba),
                                  out=torch.empty(rank, device=device, dtype=torch.bool))]
            if len(sampled_indices) == 0:
                sampled_indices = torch.randint(0, rank, size=(self.min_values, ), device=device, dtype=torch.int64)

            factors = [factor[:, sampled_indices] for factor in cp_tensor.factors]
            if training:
                weights = cp_tensor.weights[sampled_indices]*(1/(1 - self.proba))
            else:
                weights = cp_tensor.weights[sampled_indices]

        return CPTensor(weights, factors)


class TTDropout(TensorDropout, factorization=TTTensor):
    def _apply_tensor_dropout(self, tt_tensor, training=True):
        if (not self.proba) or ((not training) and (not self.drop_test)):
            return tt_tensor

        device = tt_tensor.factors[0].device

        sampled_indices = []
        for i, rank in enumerate(tt_tensor.rank[1:]):
            if rank > self.min_dim:
                idx = tl.arange(rank, device=device, dtype=torch.int64)
                idx = idx[torch.bernoulli(torch.ones(rank, device=device)*(1 - self.proba),
                                      out=torch.empty(rank, device=device, dtype=torch.bool))]
                if len(idx) == 0:
                    idx = torch.randint(0, rank, size=(self.min_values, ), device=device, dtype=torch.int64)
            else:
                idx = tl.arange(rank, device=device, dtype=torch.int64).tolist()

            sampled_indices.append(idx)

        sampled_factors = []
        if training:
            scaling = 1/(1 - self.proba)
        else:
            scaling = 1
        for i, f in enumerate(tt_tensor.factors):
            if i == 0:
                sampled_factors.append(f[..., sampled_indices[i]]*scaling)
            elif i == (tt_tensor.order - 1):
                sampled_factors.append(f[sampled_indices[i-1], ...])
            else:
                sampled_factors.append(f[sampled_indices[i-1], ...][..., sampled_indices[i]]*scaling)

        return TTTensor(sampled_factors)


def tensor_dropout(factorized_tensor, p=0, min_dim=3, min_values=1, drop_test=False):
    """Tensor Dropout

    Parameters
    ----------
    factorized_tensor : FactorizedTensor
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    p : float
        dropout probability
        if 0, no dropout is applied
        if 1, all the components but 1 are dropped in the latent space
    min_dim : int, default is 3
        only apply dropout to modes with dimension larger than `min_dim`
    min_values : int, default is 1
        minimum number of components to select

    Returns
    -------
    FactorizedTensor
        the module to which tensor dropout has been attached

    Examples
    --------
    >>> tensor = FactorizedTensor.new((3, 4, 2), rank=0.5, factorization='CP').normal_()
    >>> tensor = tensor_dropout(tensor, p=0.5)
    >>> remove_tensor_dropout(tensor)
    """
    TensorDropout.apply(factorized_tensor, p, min_dim=min_dim, min_values=min_values, drop_test=drop_test)

    return factorized_tensor


def remove_tensor_dropout(factorized_tensor):
    """Removes the tensor dropout from a TensorModule 

    Parameters
    ----------
    factorized_tensor : tltorch.FactorizedTensor
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    
    Examples
    --------
    >>> tensor = FactorizedTensor.new((3, 4, 2), rank=0.5, factorization='CP').normal_()
    >>> tensor = tensor_dropout(tensor, p=0.5)
    >>> remove_tensor_dropout(tensor)
    """
    for key, hook in factorized_tensor._forward_hooks.items():
        if isinstance(hook, TensorDropout):
            del factorized_tensor._forward_hooks[key]
            return factorized_tensor

    raise ValueError(f'TensorLasso not found in factorized tensor {factorized_tensor}')
