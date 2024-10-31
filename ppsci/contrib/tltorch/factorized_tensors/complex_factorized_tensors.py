
import paddle
from paddle import nn

import tensorly as tl

from ..factorized_tensors.factorized_tensors import TuckerTensor, CPTensor, TTTensor, DenseTensor
from ..utils.parameter_list import FactorList, ComplexFactorList

tl.set_backend('paddle')

# Author: Jean Kossaifi
# License: BSD 3 clause


class ComplexHandler():
    def __setattr__(self, key, value):
        if isinstance(value, (FactorList)):
            print(f"aa FactorListFactorListFactorListFactorList: {key}")
            value = ComplexFactorList(value)
            super().__setattr__(key, value)
            
        elif isinstance(value, paddle.base.framework.EagerParamBase):
            self.add_parameter(key, value)
        elif paddle.is_tensor(value):
            self.register_buffer(key, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        print(f"key: {key}")
        value = super().__getattr__(key)
        if paddle.is_tensor(value):
            print(">>>>>>>>>>. running view_as_complex!!!")
            print(f"value: {value.shape}")
            value = paddle.as_complex(value)
            print(f"value: {value.shape}")
            print(">>>>>>>>>>. running view_as_complex done!!!")
        return value

    def add_parameter(self, key, value):
        print(f">>>> add key: {key}")
        value = paddle.base.framework.EagerParamBase.from_tensor(paddle.as_real(value))
        super().add_parameter(key, value)

    def register_buffer(self, key, value):
        print(f">>>> register_buffer key: {key}")
        value = paddle.as_real(value)
        super().register_buffer(key, value)


class ComplexDenseTensor(ComplexHandler, DenseTensor, name='ComplexDense'):
    """Complex Dense Factorization
    """
    @classmethod
    def new(cls, shape, rank=None, device=None, dtype=paddle.complex64, **kwargs):
        return super().new(shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexTuckerTensor(ComplexHandler, TuckerTensor, name='ComplexTucker'):
    """Complex Tucker Factorization
    """
    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=paddle.complex64, **kwargs):
        return super().new(shape, rank, fixed_rank_modes=fixed_rank_modes,
                           device=device, dtype=dtype, **kwargs)


class ComplexTTTensor(ComplexHandler, TTTensor, name='ComplexTT'):
    """Complex TT Factorization
    """
    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=paddle.complex64, **kwargs):
        return super().new(shape, rank,
                           device=device, dtype=dtype, **kwargs)


class ComplexCPTensor(ComplexHandler, CPTensor, name='ComplexCP'):
    """Complex CP Factorization
    """
    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=paddle.complex64, **kwargs):
        print(">>>>>>>> apslpxw ComplexCPTensor")
        return super().new(shape, rank,
                           device=device, dtype=dtype, **kwargs)
