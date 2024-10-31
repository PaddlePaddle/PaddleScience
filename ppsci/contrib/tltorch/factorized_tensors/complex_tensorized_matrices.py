import paddle
import tensorly as tl
tl.set_backend('paddle')
from ..factorized_tensors.tensorized_matrices import TuckerTensorized, DenseTensorized, CPTensorized, BlockTT
from .complex_factorized_tensors import ComplexHandler

# Author: Jean Kossaifi
# License: BSD 3 clause


class ComplexDenseTensorized(ComplexHandler, DenseTensorized, name='ComplexDense'):
    """Complex DenseTensorized Factorization
    """
    _complex_params = ['tensor']

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype=paddle.complex64, **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexTuckerTensorized(ComplexHandler, TuckerTensorized, name='ComplexTucker'):
    """Complex TuckerTensorized Factorization
    """
    _complex_params = ['core', 'factors']

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype=paddle.complex64, **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)

class ComplexBlockTT(ComplexHandler, BlockTT, name='ComplexTT'):
    """Complex BlockTT Factorization
    """
    _complex_params = ['factors']

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype=paddle.complex64, **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)

class ComplexCPTensorized(ComplexHandler, CPTensorized, name='ComplexCP'):
    """Complex Tensorized CP Factorization
    """
    _complex_params = ['weights', 'factors']

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype=paddle.complex64, **kwargs):
        print(">>>>>>>> apslpxw ComplexCPTensorized")
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)
