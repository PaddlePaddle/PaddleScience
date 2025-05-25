import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import paddle
import paddle_geometric.typing as pyg_typing
from paddle import Tensor


HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def ptr2index(ptr: Tensor, output_size: Optional[int] = None) -> Tensor:
    index = paddle.arange(ptr.shape[0] - 1, dtype=ptr.dtype)
    return index.repeat_interleave(paddle.diff(ptr), output_size=output_size)


def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:
    if size is None:
        size = int(index.max()) + 1 if index.shape[0] > 0 else 0

    return paddle.incubate.sparse.convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype != paddle.int64)


class CatMetadata(NamedTuple):
    nnz: List[int]
    dim_size: List[Optional[int]]
    is_sorted: List[bool]


def implements(paddle_function: Callable) -> Callable:
    r"""Registers a PaddlePaddle function override."""
    @functools.wraps(paddle_function)
    def decorator(my_function: Callable) -> Callable:
        HANDLED_FUNCTIONS[paddle_function] = my_function
        return my_function

    return decorator


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in pyg_typing.INDEX_DTYPES:
        raise ValueError(f"'Index' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{pyg_typing.INDEX_DTYPES})")


def assert_one_dimensional(tensor: Tensor) -> None:
    if len(tensor.shape) != 1:
        raise ValueError(f"'Index' needs to be one-dimensional "
                         f"(got {len(tensor.shape)} dimensions)")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError("'Index' needs to be contiguous. Please call "
                         "`index.contiguous()` before proceeding.")


def assert_sorted(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: 'Index', *args: Any, **kwargs: Any) -> Any:
        if not self.is_sorted:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Cannot call '{func.__name__}' since '{cls_name}' is not "
                f"sorted. Please call `{cls_name}.sort()` first.")
        return func(self, *args, **kwargs)

    return wrapper


class Index(Tensor):
    r"""A one-dimensional `index` tensor with additional (meta)data attached.

    :class:`Index` is a subclass of :class:`paddle.Tensor` that holds
    indices of shape `[num_indices]`.

    It includes:
    - `dim_size`: The size of the underlying sparse vector size.
    - `is_sorted`: Whether indices are sorted in ascending order.
    """
    _data: Tensor
    _dim_size: Optional[int] = None
    _is_sorted: bool = False
    _indptr: Optional[Tensor] = None
    _cat_metadata: Optional[CatMetadata] = None

    @staticmethod
    def __new__(
        cls: Type,
        data: Any,
        *args: Any,
        dim_size: Optional[int] = None,
        is_sorted: bool = False,
        **kwargs: Any,
    ) -> 'Index':
        if not isinstance(data, Tensor):
            data = paddle.to_tensor(data, *args, **kwargs)

        assert_valid_dtype(data)
        assert_one_dimensional(data)
        assert_contiguous(data)

        out = data  # Tensor subclassing is handled by wrapping logic

        # Attach metadata:
        out._dim_size = dim_size
        out._is_sorted = is_sorted

        return out

    def validate(self) -> 'Index':
        r"""Validates the `Index` representation."""
        assert_valid_dtype(self._data)
        assert_one_dimensional(self._data)
        assert_contiguous(self._data)

        if self.shape[0] > 0 and self._data.min() < 0:
            raise ValueError(f"'Index' contains negative indices")

        if (self.shape[0] > 0 and self.dim_size is not None
                and self._data.max() >= self.dim_size):
            raise ValueError(f"'Index' contains indices larger than dim_size")

        if self.is_sorted and (paddle.diff(self._data) < 0).any():
            raise ValueError(f"'Index' is not sorted")

        return self

    @property
    def dim_size(self) -> Optional[int]:
        return self._dim_size

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    def get_dim_size(self) -> int:
        if self._dim_size is None:
            self._dim_size = int(self._data.max()) + 1 if self.shape[0] > 0 else 0
        return self._dim_size

    def as_tensor(self) -> Tensor:
        return self._data

    def __repr__(self) -> str:
        prefix = f'{self.__class__.__name__}('
        tensor_str = self._data.__str__()
        suffixes = [f'dim_size={self.dim_size}', f'is_sorted={self.is_sorted}']
        return f"{prefix}{tensor_str}, {', '.join(suffixes)})"
