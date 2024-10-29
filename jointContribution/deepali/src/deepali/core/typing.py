r"""Type annotations for paddle functions."""
from dataclasses import Field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import paddle
from paddle.base import core

EllipsisType = type(...)
T = TypeVar("T")
BoolStr = Union[bool, str]
ScalarOrTuple = Union[T, Tuple[T, ...]]
ScalarOrTuple1d = Union[T, Tuple[T]]
ScalarOrTuple2d = Union[T, Tuple[T, T]]
ScalarOrTuple3d = Union[T, Tuple[T, T, T]]
ScalarOrTuple4d = Union[T, Tuple[T, T, T, T]]
ScalarOrTuple5d = Union[T, Tuple[T, T, T, T, T]]
ScalarOrTuple6d = Union[T, Tuple[T, T, T, T, T, T]]

# Cannot use Sequence type annotation when using paddle
ListOrTuple = Union[List[T], Tuple[T, ...]]

Device = paddle.any
DeviceStr = Union[paddle.any, str]
DType = paddle.dtype
DTypeStr = Union[paddle.dtype, str]
Name = Optional[str]  # Optional tensor dimension name
Size = ScalarOrTuple[int]  # Order of spatial dimensions: (X, ...)
Shape = ScalarOrTuple[int]  # Order of spatial dimensions: (..., X)
Scalar = Union[int, float, paddle.Tensor]
Array = Union[Sequence[Scalar], paddle.Tensor]

PathStr = Union[Path, str]
PathUri = Union[Path, str]


class Dataclass(Protocol):
    r"""Type annotation for any dataclass."""

    __dataclass_fields__: Dict[str, Any]


# While a Batch and Sample have the same type, the difference is in their values
# - Sample: Values must be without batch dimension if tensors/arrays, but can also be of other
#           types such as a file Path or SimpleITK.Image, or None if optional.
# - Batch: Values must be tensors with batch dimension (first dimension) or None if optional.
Batch = Union[Dataclass, Dict[str, Any], NamedTuple]
Sample = Union[Dataclass, Dict[str, Any], NamedTuple]

TensorMapOrSequence = Union[Mapping[str, paddle.Tensor], Sequence[paddle.Tensor]]

TensorCollection = Union[
    TensorMapOrSequence, Mapping[str, TensorMapOrSequence], Sequence[TensorMapOrSequence]
]


def is_bool_dtype(dtype: DType) -> bool:
    r"""Checks if ``dtype`` of given NumPy array or PyTorch tensor is boolean type."""
    is_dtype = dtype == core.VarDesc.VarType.BOOL or dtype == core.DataType.BOOL or dtype == "bool"
    return is_dtype


def is_float_dtype(dtype: DType) -> bool:
    r"""Checks if ``dtype`` of given tensor is a floating point type."""
    is_fp_dtype = (
        dtype == core.VarDesc.VarType.FP32
        or dtype == core.VarDesc.VarType.FP64
        or dtype == core.VarDesc.VarType.FP16
        or dtype == core.VarDesc.VarType.BF16
        or dtype == core.DataType.FLOAT32
        or dtype == core.DataType.FLOAT64
        or dtype == core.DataType.FLOAT16
        or dtype == core.DataType.BFLOAT16
        or dtype == "float32"
        or dtype == "float64"
        or dtype == "float16"
        or dtype == "bfloat16"
    )
    return is_fp_dtype


def is_int_dtype(dtype: DType) -> bool:
    r"""Checks if ``dtype`` of given tensor is a signed integer type."""
    is_dtype = (
        dtype == core.VarDesc.VarType.INT8
        or dtype == core.VarDesc.VarType.INT16
        or dtype == core.VarDesc.VarType.INT32
        or dtype == core.VarDesc.VarType.INT64
        or dtype == core.DataType.INT8
        or dtype == core.DataType.INT16
        or dtype == core.DataType.INT32
        or dtype == core.DataType.INT64
        or dtype == "int8"
        or dtype == "int16"
        or dtype == "int32"
        or dtype == "int64"
    )
    return is_dtype


def is_uint_dtype(dtype: DType) -> bool:
    r"""Checks if ``dtype`` of given tensor is an unsigned integer type."""
    is_dtype = (
        dtype == core.VarDesc.VarType.UINT8 or dtype == core.DataType.UINT8 or dtype == "uint8"
    )
    return is_dtype


def is_namedtuple(arg: Any) -> bool:
    r"""Check if given object is a named tuple."""
    return isinstance(arg, tuple) and hasattr(arg, "_fields")


def is_optional_field(field: Field) -> bool:
    r"""Whether given dataclass field type is ``Optional[T] = Union[T, NoneType]``."""
    return is_optional_type_hint(field.type)


def is_optional_type_hint(type_hint: Any) -> bool:
    r"""Whether given type hint is ``Optional[T] = Union[T, NoneType]``."""
    type_origin = getattr(type_hint, "__origin__", None)
    if type_origin is Union:
        return type(None) in type_hint.__args__


def is_path_str(arg: Any) -> bool:
    r"""Whether given object is of type ``pathlib.Path`` or ``str``."""
    return isinstance(arg, (Path, str))


def is_path_str_type_hint(type_hint: Any, required: bool = False) -> bool:
    r"""Check if given type annotation is ``pathlib.Path``, ``PathStr = Union[pathlib.Path, str]``.

    Args:
        type_hint: Type annotation, e.g., ``dataclasses.Field.type``.
        required: Whether path argument is required. If ``False``, ``type(None)`` in the
            type hint is ignore, i.e., also ``Optional[T]`` is considered valid.

    Returns:
        Whether type hint is ``pathlib.Path``, ``Union[pathlib.Path, str]``, or
        ``Union[str, pathlib.Path]``. When ``required=False``, type annotations
        ``Optional[T] = Union[T, None]`` where ``T`` is one of the aforementioned
        path string types also results in a return value of ``True``.

    """
    if type_hint in (Path, "Path", "Optional[Path]", "PathStr", "Optional[PathStr]"):
        return True
    type_origin = getattr(type_hint, "__origin__", None)
    if type_origin is Union:
        type_args = set(type_hint.__args__)
        if not required:
            type_args.discard(type(None))
            type_args.discard("None")
        type_args.discard(str)
        type_args.discard("str")
        if not type_args:
            return False
        for type_arg in type_args:
            if type_arg not in (Path, "Path", "PathStr"):
                return False
        return True
    return False


def is_path_str_field(field: Field, required: bool = False) -> bool:
    r"""Check if given dataclass field type is ``pathlib.Path``, ``PathStr = Union[pathlib.Path, str]``."""
    return is_path_str_type_hint(field.type, required=required)
