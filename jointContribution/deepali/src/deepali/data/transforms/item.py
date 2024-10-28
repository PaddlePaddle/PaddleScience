r"""Generic wrappers and mix-ins for data transforms."""
from collections.abc import KeysView
from copy import copy as shallowcopy
from copy import deepcopy
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Union

import paddle
from deepali.core.collections import re_output_key_index
from deepali.core.typing import is_namedtuple

__all__ = ("ItemTransform", "ItemwiseTransform")


class ItemTransform(paddle.nn.Layer):
    r"""Transform only specified item/field of dict, named tuple, tuple, list, or dataclass."""

    def __init__(
        self,
        transform: Callable,
        key: Optional[Union[int, str, KeysView, Iterable[Union[int, str]]]] = None,
        copy: bool = False,
        ignore_meta: bool = True,
        ignore_missing: bool = False,
    ):
        r"""Initialize item transformation.

        Args:
            transform: Item value transformation.
            key: Index, key, or field name of item to transform. If ``None``, empty string, or 'all',
                apply ``transform`` to all items in the input ``data`` object. Can be a nested
                key with dot ('.') character as subkey delimiters, and contain indices to access
                list or tuple items. For example, "a.b.c", "a.b.2", "a.b.c[1]", "a[1][0]",
                "a[0].c", and "a.0.c", are all valid keys as long as the data to transform has
                the appropriate entries and (nested) container types.
            copy: Whether to copy all input items. By default, a shallow copy of the input ``data``
                is made and only the item specified by ``key`` is replaced by its transformed value.
                If ``True``, a deep copy of all input ``data`` items is made.
            ignore_missing: Whether to skip processing of items with value ``None``.
            ignore_meta: Preserve 'meta' dictionary key value as is (cf. ``MetaDataset``).

        """
        super().__init__()
        self.transform = transform
        if not isinstance(key, str) and isinstance(key, (KeysView, Iterable)):
            key = set(key)
        self.key = key
        self.copy = copy
        self.ignore_meta = ignore_meta
        self.ignore_missing = ignore_missing

    def forward(self, data: Any) -> Any:
        r"""Apply transformation.

        Args:
            data: Input value, dict, tuple, list, or dataclass.

        Returns:
            Copy of ``data`` with value of ``self.key`` replaced by its transformed value.
            If ``self.key is None``, all values in the input ``data`` are transformed.
            By default, a shallow copy of ``data`` is made unless ``self.copy == True``.

        """
        if self.key in (None, "", "all", "ALL"):
            return self._apply_all(data)
        if isinstance(self.key, set):
            keys = self.key
        elif not isinstance(self.key, str) and isinstance(self.key, (KeysView, Iterable)):
            keys = set(self.key)
        else:
            keys = [self.key]
        for key in keys:
            if isinstance(key, int):
                data = self._apply_index(data, key)
            elif isinstance(key, str):
                key = re_output_key_index.sub(".\\1", key)
                data = self._apply_key(data, key)
            else:
                raise TypeError(f"{type(self).__name__}() 'key' must be None, int, or str")
        return data

    def _apply_all(self, data: Any) -> Any:
        r"""Transform all leaf items."""
        if is_dataclass(data):
            output = shallowcopy(data)
            for field in fields(data):
                value = getattr(data, field.name)
                if not self.ignore_meta or field.name != "meta":
                    value = self._apply_all(value)
                setattr(output, field.name, value)
        elif isinstance(data, Mapping):
            output = {}
            for k, v in data.items():
                if not self.ignore_meta or k != "meta":
                    v = self._apply_all(v)
                output[k] = v
        elif isinstance(data, tuple):
            output = tuple(self._apply_all(d) for d in data)
            if is_namedtuple(data):
                output = type(data)(*output)
        elif isinstance(data, list):
            output = list(self._apply_all(d) for d in data)
        elif data is None:
            if self.ignore_missing:
                output = None
            else:
                raise ValueError(f"{type(self).__name__}() value is None (ignore_missing=False)")
        else:
            output = self.transform(data)
        return output

    def _apply_index(self, data: Any, index: int) -> Any:
        r"""Transform item at specified index."""
        if not isinstance(data, (list, tuple)):
            raise TypeError(f"{type(self).__name__}() 'data' must be list or tuple when key is int")
        try:
            item = data[index]
        except IndexError:
            raise IndexError(
                f"{type(self).__name__}() 'data' sequence must have item at index {index}"
            )
        item = self.transform(item)
        args = (item if i == index else self._maybe_copy(v) for i, v in enumerate(data))
        if is_namedtuple(data):
            return type(data)(*args)
        return type(data)(args)

    def _apply_key(self, data: Any, key: str, prefix: str = "") -> Any:
        r"""Transform specified item in data map."""
        parts = key.split(".", 1)
        if len(parts) == 1:
            parts = [parts[0], ""]
        key, subkey = parts
        if not key:
            raise KeyError(
                f"{type(self).__name__}() 'key' must not be empty (prefix={prefix!r}, key={key!r}, subkey={subkey!r})"
            )
        # Get item at specified entry
        index = None
        item = None
        if is_dataclass(data) or is_namedtuple(data):
            try:
                item = getattr(data, key)
            except AttributeError:
                if prefix:
                    msg = f"{type(self).__name__}() 'data' entry {prefix!r} has no attribute named {key}"
                else:
                    msg = f"{type(self).__name__}() 'data' has no attribute named {key}"
                raise AttributeError(msg)
        elif isinstance(data, (list, tuple)):
            try:
                index = int(key)
            except (TypeError, ValueError):
                if prefix:
                    msg = f"{type(self).__name__}() 'data' entry {prefix!r} is list or tuple, but key is no index"
                else:
                    msg = f"{type(self).__name__}() 'data' is list or tuple, but key is no index"
                raise AttributeError(msg)
            try:
                item = data[index]
            except IndexError:
                if prefix:
                    msg = f"{type(self).__name__}() 'data' entry {prefix!r} index {index} is out of bounds"
                else:
                    msg = f"{type(self).__name__}() 'data' index {index} is out of bounds"
                raise IndexError(msg)
        elif isinstance(data, Mapping):
            try:
                item = data[key]
            except KeyError:
                try:
                    index = int(key)
                    item = data[index]
                    key = index
                except (IndexError, KeyError, TypeError, ValueError):
                    if prefix:
                        msg = (
                            f"{type(self).__name__}() 'data' dict {prefix!r} must have key {key!r}"
                        )
                    else:
                        msg = f"{type(self).__name__}() 'data' dict must have key {key!r}"
                    raise KeyError(msg)
        else:
            if prefix:
                msg = f"{type(self).__name__}() 'data' entry {prefix!r} must be list, tuple, dict, dataclass, or namedtuple"
            else:
                msg = f"{type(self).__name__}() 'data' must be list, tuple, dict, dataclass, or namedtuple"
            raise TypeError(msg)
        # Transform item
        if subkey:
            item = self._apply_key(item, subkey, prefix=prefix + "." + key if prefix else key)
        else:
            item = self._apply_all(item)
        # Replace item, copy others if requested
        if is_dataclass(data):
            if self.copy:
                args = (
                    item if field.name == key else deepcopy(getattr(data, field.name))
                    for field in fields(data)
                )
                data = type(data)(*args)
            else:
                setattr(data, key, item)
        elif is_namedtuple(data):
            if self.copy:
                args = (item if k == key else deepcopy(v) for k, v in zip(data._fields, data))
                data = type(data)(*args)
            else:
                data = data._replace(**{key: item})
        elif isinstance(data, tuple):
            assert index is not None
            data = tuple(item if i == index else self._maybe_copy(v) for i, v in enumerate(data))
        elif isinstance(data, (list, tuple)):
            assert index is not None
            data = list(item if i == index else self._maybe_copy(v) for i, v in enumerate(data))
        else:
            assert isinstance(data, Mapping)
            data = {k: (item if k == key else self._maybe_copy(v)) for k, v in data.items()}
        return data

    def _maybe_copy(self, data: Any) -> Any:
        if self.copy:
            return deepcopy(data)
        return data

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self.transform!r}, key={self.key!r}, copy={self.copy!r})"


class ItemwiseTransform:
    r"""Mix-in for data preprocessing and augmentation transforms."""

    @classmethod
    def item(
        cls,
        key: Union[int, str, KeysView, Iterable[Union[int, str]]],
        *args,
        ignore_meta: bool = True,
        ignore_missing: bool = False,
        **kwargs,
    ) -> ItemTransform:
        r"""Apply transform to specified item only."""
        return ItemTransform(
            cls(*args, **kwargs), key=key, ignore_meta=ignore_meta, ignore_missing=ignore_missing
        )
