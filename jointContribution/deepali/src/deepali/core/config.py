r"""Auxiliary functions and classes for dealing with configuration files."""

from __future__ import annotations  # noqa

import json
from dataclasses import asdict
from dataclasses import fields
from io import StringIO
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Type
from typing import TypeVar

import dacite

# PyYAML does not support YAML 1.2, which is required by DVC for example
# (cf. https://github.com/iterative/dvc/issues/8466#issuecomment-1290757564)
from ruamel.yaml import YAML

from .pathlib import Path
from .pathlib import PathUri
from .pathlib import abspath_template
from .pathlib import is_uri
from .storage import StorageObject
from .typing import is_path_str_field

T = TypeVar("T", bound="DataclassConfig")


class DataclassConfig(object):
    r"""Base class of configuration data classes."""

    @staticmethod
    def section() -> str:
        r"""Common key prefix of configuration entries in configuration file."""
        return ""

    @classmethod
    def _from_dict(cls: Type[T], arg: Mapping[str, Any], parent: Optional[Path] = None) -> T:
        r"""Create configuration from dictionary."""
        config = dacite.from_dict(cls, arg)
        return config

    @classmethod
    def from_dict(cls: Type[T], arg: Mapping[str, Any], parent: Optional[Path] = None) -> T:
        r"""Create configuration from dictionary."""
        config = cls._from_dict(arg, parent=parent)
        config._finalize(parent)
        return config

    @classmethod
    def from_path(cls: Type[T], path: PathUri, section: Optional[str] = None) -> T:
        r"""Load configuration from file."""
        config = read_config_dict(path)
        if section is None:
            section = cls.section()
        if section:
            for key in section.split("."):
                config = config.get(key, {})
        if is_uri(path):
            if path.startswith("file:///"):
                parent = Path(path[7:]).parent
            else:
                parent = None
        else:
            parent = Path(path).absolute().parent
        return cls.from_dict(config, parent=parent)

    @classmethod
    def read(cls: Type[T], path: PathUri, section: Optional[str] = None) -> T:
        r"""Load configuration from JSON or YAML file."""
        return cls.from_path(path, section=section)

    def write(self, path: PathUri) -> None:
        r"""Save configuration to JSON or YAML file."""
        config = self.asdict()
        write_config_dict(config, path)

    def asdict(self) -> Dict[str, Any]:
        r"""Get configuration dictionary."""
        return asdict(self)

    def _finalize(self: T, parent: Optional[Path] = None) -> None:
        r"""Finalize parameters after loading these from input file."""
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
        r"""Merge kwarg dictionaries in 'norm' or 'acti' sequence.

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
    r"""Merge kwarg dictionaries in 'norm' or 'acti' sequence.

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
        if not all(isinstance(item, (str, dict) if i == 0 else dict) for i, item in enumerate(arg)):
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


def read_config_dict(path: PathUri) -> Dict[str, Any]:
    r"""Read configuration from JSON or YAML file."""
    with StorageObject.from_path(path) as config_object:
        config_suffix = config_object.path.suffix
        if config_suffix.lower() not in (".json", ".yaml", ".yml"):
            raise ValueError(f"read_config_dict() 'path' has unsupported suffix {config_suffix}")
        config_text = config_object.read_text()
        if config_suffix == ".json":
            config_dict = json.loads(config_text)
        else:
            config_dict = YAML(typ="safe").load(config_text)
    if config_dict is None:
        config_dict = {}
    return config_dict


def write_config_dict(config: Dict[str, Any], path: PathUri) -> None:
    r"""Write configuration to JSON or YAML file."""
    with StorageObject.from_path(path) as config_object:
        config_suffix = config_object.path.suffix
        if config_suffix.lower() not in (".json", ".yaml", ".yml"):
            raise ValueError(f"write_config_dict() 'path' has unsupported suffix {config_suffix}")
        if config_suffix == ".json":
            config_text = json.dumps(config) + "\n"
        else:
            buffer = StringIO()
            YAML(typ="safe").dump(config, buffer)
            config_text = buffer.getvalue()
        config_object.write_text(config_text)
