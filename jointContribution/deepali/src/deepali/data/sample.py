r"""Functions for dealing with dataset sample collections."""

from collections import OrderedDict
from copy import copy as shallowcopy
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Tuple

from deepali.core.typing import Sample
from deepali.core.typing import is_namedtuple

__all__ = (
    "Sample",
    "sample_field_names",
    "sample_field_value",
    "replace_all_sample_field_values",
)


def sample_field_names(sample: Sample) -> Tuple[str]:
    r"""Get names of fields in data sample."""
    if is_dataclass(sample):
        return tuple(field.name for field in fields(sample))
    if is_namedtuple(sample):
        return sample._fields
    if not isinstance(sample, Mapping):
        raise TypeError("Dataset 'sample' must be dataclass, Mapping, or NamedTuple")
    return tuple(sample.keys())


def sample_field_value(sample: Sample, name: str) -> Any:
    r"""Get sample value of named data field."""
    if isinstance(sample, Mapping):
        return sample[name]
    return getattr(sample, name)


def replace_all_sample_field_values(sample: Sample, values: Sequence[Any]) -> Sample:
    r"""Replace all sample field values."""
    names = sample_field_names(sample)
    if len(names) != len(values):
        raise ValueError("replace_all_values() 'values' must contain an entry for every field")
    if is_dataclass(sample):
        result = shallowcopy(sample)
        for name, value in zip(names, values):
            setattr(result, name, value)
        return result
    if is_namedtuple(sample):
        return sample._replace(**{name: value for name, value in zip(names, values)})
    if isinstance(sample, OrderedDict):
        return OrderedDict([(name, value) for name, value in zip(names, values)])
    return {name: value for name, value in zip(names, values)}
