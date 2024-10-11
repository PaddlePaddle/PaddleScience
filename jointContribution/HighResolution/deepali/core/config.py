"""Auxiliary functions and classes for dealing with configuration files."""
from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Type
from typing import TypeVar

import dacite
import yaml

from .path import abspath_template
from .types import PathStr
from .types import is_path_str_field

T = TypeVar("T", bound="DataclassConfig")


class DataclassConfig(object):
    """Base class of configuration data classes."""

    @staticmethod
    def section() -> str:
        """Common key prefix of configuration entries in configuration file."""
        return ""

    @classmethod
    def _from_dict(
        cls: Type[T], arg: Mapping[str, Any], parent: Optional[Path] = None
    ) -> T:
        """Create configuration from dictionary."""
        config = dacite.from_dict(cls, arg)
        return config

    @classmethod
    def from_dict(
        cls: Type[T], arg: Mapping[str, Any], parent: Optional[Path] = None
    ) -> T:
        """Create configuration from dictionary."""
        config = cls._from_dict(arg, parent=parent)
        config._finalize(parent)
        return config

    @classmethod
    def from_path(cls: Type[T], path: PathStr, section: Optional[str] = None) -> T:
        """Load configuration from file."""
        path = Path(path).absolute()
        text = path.read_text()
        if path.suffix == ".json":
            config = json.loads(text)
        elif path.suffix in (".yml", ".yaml"):
            config = yaml.load(text, Loader=yaml.SafeLoader)
        else:
            raise ValueError(
                f"{cls.__name__}.from_path() 'path' has unsupported suffix {path.suffix}"
            )
        if config is None:
            config = {}
        if section is None:
            section = cls.section()
        if section:
            for key in section.split("."):
                config = config.get(key, {})
        return cls.from_dict(config, parent=path.parent)

    @classmethod
    def read(cls: Type[T], path: PathStr, section: Optional[str] = None) -> T:
        """Load configuration from file."""
        return cls.from_path(path, section=section)

    def write(self, path: PathStr) -> None:
        """Write configuration to file."""
        path = Path(path).absolute()
        config = self.asdict()
        if path.suffix == ".json":
            text = json.dumps(config)
        elif path.suffix in (".yml", ".yaml"):
            text = yaml.safe_dump(config)
        else:
            raise ValueError(
                f"{type(self).__name__}.write() 'path' has unsupported suffix {path.suffix}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def asdict(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return asdict(self)

    def _finalize(self: T, parent: Optional[Path] = None) -> None:
        """Finalize parameters after loading these from input file."""
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, DataclassConfig):
                value._finalize(parent)
            elif is_path_str_field(field):
                value = abspath_template(value, parent=parent)
                setattr(self, field.name, value)

    def _join_kwargs_in_sequence(self, attr: str):
        """Merge kwarg dictionaries in 'norm' or 'acti' sequence.

        In case of a sequence instead of a single string or dictionary, ``ConvLayer`` expects
        ``norm`` and ``acti`` parameters to be a sequence of length 2, where the first entry
        is the normalization or activiation layer name, respectively, and the second entry is
        a dictionary of keyword arguments for the respective layer. In a YAML file, it may
        be convenient, however, to specify these arguments on a single line as follows:

        .. code-block:: yaml

            norm: [batch, momentum: 0.1, eps: 0.001]

        This represents a sequence, where the first item is a string and the following items
        are dictionaries with a single key. This ``__post_init__`` function merges these separate
        dictionaries into a single dictionary in order to support above YAML syntax.

        The same functionality may be useful for other configuration entries, not only
        ``normalization()``, ``activiation()``, or ``pooling()`` related parameters.

        Alternatively, one could use a single dictionary in the YAML configuration:

        .. code-block:: yaml

            norm: {name: batch, momentum: 0.1, eps: 0.001}

        Args:
            attr: Name of dataclass attribute to modify in place.

        """
        arg = getattr(self, attr)
        arg = join_kwargs_in_sequence(arg)
        setattr(self, attr, arg)


def join_kwargs_in_sequence(arg):
    """Merge kwarg dictionaries in 'norm' or 'acti' sequence.

    In case of a sequence instead of a single string or dictionary, ``ConvLayer`` expects
    ``norm`` and ``acti`` parameters to be a sequence of length 2, where the first entry
    is the normalization or activiation layer name, respectively, and the second entry is
    a dictionary of keyword arguments for the respective layer. In a YAML file, it may
    be convenient, however, to specify these arguments on a single line as follows:

    .. code-block:: yaml

        norm: [batch, momentum: 0.1, eps: 0.001]

    This represents a sequence, where the first item is a string and the following items
    are dictionaries with a single key. This ``__post_init__`` function merges these separate
    dictionaries into a single dictionary in order to support above YAML syntax.

    The same functionality may be useful for other configuration entries, not only
    ``normalization()``, ``activiation()``, or ``pooling()`` related parameters.

    Alternatively, one could use a single dictionary in the YAML configuration:

    .. code-block:: yaml

        norm: {name: batch, momentum: 0.1, eps: 0.001}

    Args:
        attr: Name of dataclass attribute to modify in place.

    """
    if not isinstance(arg, str) and isinstance(arg, Sequence):
        if not all(
            isinstance(item, (str, dict) if i == 0 else dict)
            for i, item in enumerate(arg)
        ):
            raise TypeError(
                "join_kwargs_in_sequence() 'arg' must be str, dict, or sequence of dicts with the first item being either a str or dict"
            )
        if len(arg) == 1 and isinstance(arg[0], dict):
            arg = arg[0]
        elif len(arg) > 1:
            start = 0 if isinstance(arg[0], dict) else 1
            kwargs = dict(arg[start])
            for i in range(start + 1, len(arg)):
                cur = arg[i]
                assert isinstance(cur, dict)
                for name, value in cur.items():
                    if name in kwargs:
                        raise ValueError(
                            f"join_kwargs_in_sequence() 'arg' has duplicate kwarg {name}"
                        )
                    kwargs[name] = value
            arg = kwargs if start == 0 else (arg[0], kwargs)
    return arg
