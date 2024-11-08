from __future__ import annotations

from collections import OrderedDict
from typing import Optional
from typing import Type
from typing import TypeVar

import paddle

from ..core.tensor import as_tensor
from ..core.types import Array
from ..core.types import Device
from ..core.types import DType

T = TypeVar("T", bound="DataTensor")
__all__ = ("DataTensor",)


class DataTensor(paddle.Tensor):
    """Data tensor base class."""

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
        data = as_tensor(data, dtype=dtype, device=device)
        if requires_grad is None:
            requires_grad = not data.stop_gradient
        if pin_memory:
            data = data.pin_memory()

        instance = super().__new__(cls)
        instance = paddle.assign(data, instance)

        if requires_grad:
            instance.stop_gradient = False
        else:
            instance.stop_gradient = True

        return instance

    def __init__(
        self: T,
        data: Array,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        """Initialize data tensor.

        Args:
            data: paddle.Tensor data.
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
        """Create a new instance while preserving subclass (meta-)data."""
        if data is None:
            data = self
        if type(data) is not paddle.Tensor:
            data = data.as_subclass(paddle.Tensor)
        return type(self)(data, **kwargs)

    def __copy__(self: T) -> T:
        return self._make_instance()

    def __deepcopy__(self: T, memo) -> T:
        if id(self) in memo:
            return memo[id(self)]
        result = self._make_instance(
            self.data.clone(),
            requires_grad=self.requires_grad,
            pin_memory=self.is_pinned(),
        )
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        paddle.utils.hooks.warn_if_has_hooks(self)
        args = self.storage(), self.storage_offset(), tuple(self.size()), self.stride()
        if self.is_quantized:
            args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = (
            paddle._utils._rebuild_qtensor
            if self.is_quantized
            else paddle._utils._rebuild_tensor_v2
        )
        return _rebuild_from_type, (f, type(self), args, self.__dict__)

    def tensor(self: T) -> paddle.Tensor:
        """Convert to plain paddle.Tensor."""
        return self.detach()


def _rebuild_from_type(func, type, args, dict):
    """Function used by DataTensor.__reduce_ex__ to support unpickling of subclass type."""
    ret: paddle.Tensor = func(*args)
    if type is not paddle.Tensor:
        ret = ret.as_subclass(type)
        ret.__dict__ = dict
    return ret
