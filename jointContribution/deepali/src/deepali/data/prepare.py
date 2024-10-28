r"""Functions for preparing model input containing tensor decorators."""

from collections import abc
from dataclasses import is_dataclass
from typing import Any
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from typing import overload

import paddle
from deepali.core.typing import Batch
from deepali.core.typing import Dataclass
from deepali.core.typing import Device
from deepali.core.typing import is_namedtuple

from .sample import replace_all_sample_field_values
from .sample import sample_field_names
from .sample import sample_field_value

__all__ = ("prepare_batch",)


@overload
def prepare_batch(batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ...


@overload
def prepare_batch(batch: Sequence[Dataclass]) -> Dataclass:
    ...


@overload
def prepare_batch(batch: Sequence[NamedTuple]) -> NamedTuple:
    ...


def prepare_batch(
    batch: Batch,
    device: Optional[Union[Device, str]] = None,
    non_blocking: bool = False,
    memory_format=None,
) -> Batch:
    r"""Move batch data to execution device."""
    names = sample_field_names(batch)
    values = []
    for name in names:
        value = sample_field_value(batch, name)
        value = prepare_item(
            value, device=device, non_blocking=non_blocking, memory_format=memory_format
        )
        values.append(value)
    return replace_all_sample_field_values(batch, values)


def prepare_item(
    value: Any,
    device: Optional[Union[Device, str]] = None,
    non_blocking: bool = False,
    memory_format=None,
) -> Any:
    r"""Move batch item data to execution device."""
    kwargs = dict(device=device, blocking=not non_blocking)
    if isinstance(value, paddle.Tensor):
        value = value.to(**kwargs)
    elif isinstance(value, abc.Mapping) or is_dataclass(value) or is_namedtuple(value):
        value = prepare_batch(value, **kwargs)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = [prepare_item(item, **kwargs) for item in value]
    return value
