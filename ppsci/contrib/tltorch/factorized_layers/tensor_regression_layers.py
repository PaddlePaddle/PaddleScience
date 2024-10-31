"""Tensor Regression Layers
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import torch
import torch.nn as nn

import tensorly as tl
tl.set_backend('pytorch')
from ..functional.tensor_regression import trl

from ..factorized_tensors import FactorizedTensor

class TRL(nn.Module):
    """Tensor Regression Layers 
        
    Parameters
    ----------
    input_shape : int iterable
        shape of the input, excluding batch size
    output_shape : int iterable
        shape of the output, excluding batch size
    verbose : int, default is 0
        level of verbosity

    References
    ----------
    .. [1] Tensor Regression Networks, Jean Kossaifi, Zachary C. Lipton, Arinbjorn Kolbeinsson, 
        Aran Khanna, Tommaso Furlanello, Anima Anandkumar, JMLR, 2020. 
    """
    def __init__(self, input_shape, output_shape, bias=False, verbose=0, 
                factorization='cp', rank='same', n_layers=1,
                device=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

        if isinstance(input_shape, int):
            self.input_shape = (input_shape, )
        else:
            self.input_shape = tuple(input_shape)
            
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, )
        else:
            self.output_shape = tuple(output_shape)
        
        self.n_input = len(self.input_shape)
        self.n_output = len(self.output_shape)
        self.weight_shape = self.input_shape + self.output_shape
        self.order = len(self.weight_shape)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_shape, device=device, dtype=dtype))
        else:
            self.bias = None

        if n_layers == 1:
            factorization_shape = self.weight_shape
        elif isinstance(n_layers, int):
            factorization_shape = (n_layers, ) + self.weight_shape
        elif isinstance(n_layers, tuple):
            factorization_shape = n_layers + self.weight_shape
        
        if isinstance(factorization, FactorizedTensor):
            self.weight = factorization.to(device).to(dtype)
        else:
            self.weight = FactorizedTensor.new(factorization_shape, rank=rank, factorization=factorization,
                                               device=device, dtype=dtype)
            self.init_from_random()
    
        self.factorization = self.weight.name

    def forward(self, x):
        """Performs a forward pass"""
        return trl(x, self.weight, bias=self.bias)
    
    def init_from_random(self, decompose_full_weight=False):
        """Initialize the module randomly

        Parameters
        ----------
        decompose_full_weight : bool, default is False
            if True, constructs a full weight tensor and decomposes it to initialize the factors
            otherwise, the factors are directly initialized randomlys        
        """
        with torch.no_grad():
            if decompose_full_weight:
                full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
                self.weight.init_from_tensor(full_weight)
            else:
                self.weight.normal_()
            if self.bias is not None:
                self.bias.uniform_(-1, 1)

    def init_from_linear(self, linear, unsqueezed_modes=None, **kwargs):                                                                                                                                                                                                                                                                       
        """Initialise the TRL from the weights of a fully connected layer

        Parameters
        ----------
        linear : torch.nn.Linear
        unsqueezed_modes : int list or None
            For Tucker factorization, this allows to replace pooling layers and instead 
            learn the average pooling for the specified modes ("unsqueezed_modes").
            **for factorization='Tucker' only**
        """
        if unsqueezed_modes is not None:
            if self.factorization != 'Tucker':
                raise ValueError(f'unsqueezed_modes is only supported for factorization="tucker" but factorization is {self.factorization}.')
    
            unsqueezed_modes = sorted(unsqueezed_modes)
            weight_shape = list(self.weight_shape)
            for mode in unsqueezed_modes[::-1]:
                if mode == 0:
                    raise ValueError(f'Cannot learn pooling for mode-0 (channels).')
                if mode > self.n_input:
                    msg = 'Can only learn pooling for the input tensor. '
                    msg += f'The input has only {self.n_input} modes, yet got a unsqueezed_mode for mode {mode}.'
                    raise ValueError(msg)

                weight_shape.pop(mode)
                kwargs['unsqueezed_modes'] = unsqueezed_modes
        else:
            weight_shape = self.weight_shape
        
        with torch.no_grad():
            weight = torch.t(linear.weight).contiguous().view(weight_shape)

            self.weight.init_from_tensor(weight, **kwargs)
            if self.bias is not None:
                self.bias.data = linear.bias.data
