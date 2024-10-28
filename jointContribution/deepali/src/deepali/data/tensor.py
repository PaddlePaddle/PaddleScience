r"""Base class of tensor subclasses with additional attributes and methods."""
from __future__ import annotations  # noqa

from typing import Optional
from typing import Type
from typing import TypeVar

import paddle
from deepali.core.tensor import as_tensor
from deepali.core.typing import Array
from deepali.core.typing import Device
from deepali.core.typing import DType
from paddle import Tensor

T = TypeVar("T", bound="DataTensor")


__all__ = ("DataTensor",)


class DataTensor(paddle.Tensor):
    r"""Data tensor base class."""

    def __new__(
        cls: Type[T],
        data: Array,
        *args,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
        **kwargs,
    ) -> T:
        obj = super(DataTensor, cls).__new__(cls)
        data = as_tensor(data, dtype=dtype, device=device)
        if pin_memory:
            data = data.pin_memory()
        obj.data = data.data
        if requires_grad is not None:
            obj.stop_gradient = not requires_grad
        return obj

    def __init__(
        self: T,
        data: Array,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize data tensor.

        Args:
            data: Tensor data.
            dtype: Data type. A copy of the data is only made when the desired
                ``dtype`` is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store the data. A copy of the data is only made when
                the data has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned data tensor.
            pin_memory: If set, returned data tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        ...

    def _make_instance(self: T, data: Optional[paddle.Tensor] = None, **kwargs) -> T:
        r"""Create a new instance while preserving subclass (meta-)data."""
        if data is None:
            data = self
        if type(data) is not Tensor:
            data = paddle.to_tensor(data)
        return type(self)(data, **kwargs)

    def __copy__(self: T) -> T:
        return self._make_instance()

    def __deepcopy__(self: T, memo) -> T:
        if id(self) in memo:
            return memo[id(self)]
        result = self._make_instance(
            self.data.clone(), requires_grad=self.requires_grad, pin_memory=self.is_pinned()
        )
        memo[id(self)] = result
        return result

    def tensor(self: T) -> paddle.Tensor:
        r"""Convert to plain paddle.Tensor."""
        ret = paddle.Tensor()
        ret.data = self.data
        ret.stop_gradient = self.stop_gradient
        return ret
