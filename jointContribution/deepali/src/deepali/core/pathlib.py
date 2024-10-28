r"""File path utility functions."""
import os
import re
from errno import ENOTEMPTY
from pathlib import Path
from shutil import rmtree
from typing import Any
from typing import Optional
from typing import Union
from typing import overload
from urllib.parse import urlsplit

from .typing import PathStr
from .typing import PathUri
from .typing import is_path_str
from .typing import is_path_str_field
from .typing import is_path_str_type_hint

__all__ = (
    "Path",
    "PathStr",
    "PathUri",
    "abspath",
    "abspath_template",
    "delete",
    "is_absolute",
    "is_path_str",
    "is_path_str_field",
    "is_path_str_type_hint",
    "is_uri",
    "make_parent_dir",
    "norm_uri",
    "path_suffix",
    "path_uri",
    "to_uri",
    "unlink_or_mkdir",
)


@overload
def abspath(path: str, parent: Optional[PathStr] = None) -> str:
    ...


@overload
def abspath(path: Path, parent: Optional[PathStr] = None) -> Path:
    ...


def abspath(path: PathStr, parent: Optional[PathStr] = None) -> PathStr:
    r"""Make path absolute."""
    path_is_str = isinstance(path, str)
    if not path_is_str and not isinstance(path, Path):
        raise TypeError(f"abspath() 'path' must be pathlib.Path or str, not {type(path)}")
    path = (Path(parent or ".") / Path(path)).absolute()
    if path_is_str:
        path = path.as_posix()
    return path


@overload
def abspath_template(path: str, parent: Optional[Path] = None) -> str:
    ...


@overload
def abspath_template(path: Path, parent: Optional[Path] = None) -> Path:
    ...


def abspath_template(path: PathStr, parent: Optional[Path] = None) -> PathStr:
    r"""Make path format string absolute."""
    if not isinstance(path, (Path, str)):
        raise TypeError(f"abspath_template() 'path' must be pathlib.Path or str, not {type(path)}")
    if str(path).startswith("{"):
        return path
    return abspath(path, parent=parent)


def delete(path: PathStr, non_empty: bool = True) -> bool:
    r"""Remove file or (non-empty) directory."""
    path = Path(path)
    if path.is_dir():
        if non_empty:
            rmtree(path)
        else:
            try:
                rmdir(path)  # noqa
            except OSError as error:
                if error.errno == ENOTEMPTY:
                    return False
                raise
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            return False
    return True


def is_absolute(path: Union[Path, str]) -> bool:
    r"""Check whether given path string or URI is absolute."""
    if is_uri(path):
        return True
    return Path(path).is_absolute()


def is_uri(arg: Any) -> bool:
    r"""Check whether a given argument is a URI."""
    if isinstance(arg, Path):
        return False
    if isinstance(arg, str):
        # Windows path with drive letter
        if os.name == "nt" and re.match("([a-zA-Z]):[/\\\\](.*)", arg):
            return False
        return re.match("([a-zA-Z0-9]+)://(.*)", arg) is not None
    return False


def make_parent_dir(path: PathStr, parents: bool = True, exist_ok: bool = True) -> Path:
    r"""Make parent directory of file path."""
    parent = Path(path).absolute().parent
    parent.mkdir(parents=parents, exist_ok=exist_ok)
    return parent


def norm_uri(uri: str) -> str:
    r"""Normalize URI.

    Args:
        uri: A valid URI string.

    Returns:
        Normalized URI string.

    """
    match = re.match("(?P<scheme>[a-zA-Z0-9]+)://(?P<path>.*)", uri)
    if not match:
        raise ValueError(f"norm_uri() 'uri' is not a valid URI: {uri}")
    scheme = match["scheme"].lower()
    path = re.sub("^/+", "", re.sub("[/\\\\]{1,}", "/", match["path"]))
    # Local file URI
    if scheme == "file":
        if os.name != "nt" or re.match("(?P<drive>[a-zA-Z]):[/\\\\]", path) is None:
            path = "/" + path
        return "file://" + path
    # AWS S3 object URI
    if scheme == "s3":
        return "s3://" + path
    # Other URI
    return urlsplit(uri, scheme="file").geturl()


def path_suffix(path: PathUri) -> str:
    r"""Get filename suffix, including .gz, .bz2 if present."""
    m = re.search("(\\.[a-zA-Z0-9]+(\\.gz|\\.GZ|\\.bz2|\\.BZ2|\\.zst|\\.ZST)?)$", str(path))
    return m.group(1) if m else ""


def path_uri(arg: PathStr) -> str:
    r"""Create valid URI from local path.

    Unlike Path.as_uri(), this function does not escape special characters as used in format template strings.

    Args:
        arg: Local path.

    Returns:
        Valid URI.

    """
    uri = norm_uri(f"file://{Path(arg).absolute()}")
    if uri.endswith("/"):
        # Trailing forward slashes are removed by Path already, but trailing backward slashes are
        # only converted to a single forward slash by norm_uri(), not removed by Path. To produce
        # a consistent result regardless of whether forward or backward slashes are used, remove
        # any remaining trailing slash even if it could signify a directory.
        uri = uri[:-1]
    return uri


def to_uri(*args: Optional[PathUri]) -> str:
    r"""Create valid URI from resource paths.

    Args:
        args: Local path components or an already valid URI. The last absolute path or URI in this
            list of arguments is the base path or URI prefix for subsequent relative paths which
            are appended to this base to construct the URI. Any ``None`` values are ignored.

    Returns:
        Valid URI.

    """
    args = [arg for arg in args if arg is not None]
    for i, arg in enumerate(reversed(args)):
        if is_uri(arg):
            base = str(arg)
            args = args[len(args) - i :]
            break
        elif Path(arg).is_absolute():
            base = Path(arg)
            args = args[len(args) - i :]
            break
    else:
        base = Path.cwd()
    if isinstance(base, Path):
        uri = path_uri(base.joinpath(*args))
    else:
        uri = norm_uri(f"{base}/{'/'.join(args)}" if args else base)
    return uri


def unlink_or_mkdir(path: PathStr) -> Path:
    r"""Unlink existing file or make parent directory if non-existent.

    This function is useful when a script output file is managed by DVC using protected symbolic or hard links.
    Call this function before writing the new output file. It will unlike any existing output file, and ensure
    that the output directory exists.

    Args:
        path: File path.

    Returns:
        Absolute file path.

    """
    path = Path(path).absolute()
    try:
        path.unlink()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
