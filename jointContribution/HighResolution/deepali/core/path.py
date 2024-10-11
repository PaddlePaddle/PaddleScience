import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from tempfile import mkstemp
from typing import Generator
from typing import Optional
from typing import Union
from typing import overload

from .types import PathStr


@overload
def abspath(path: str, parent: Optional[PathStr] = None) -> str:
    ...


@overload
def abspath(path: Path, parent: Optional[PathStr] = None) -> Path:
    ...


def abspath(path: PathStr, parent: Optional[PathStr] = None) -> PathStr:
    """Make path absolute."""
    path_type = type(path)
    if path_type not in (Path, str):
        raise TypeError("abspath() 'path' must be pathlib.Path or str")
    path = (Path(parent or ".") / Path(path)).absolute()
    if path_type is str:
        path = path.as_posix()
    return path


@overload
def abspath_template(path: str, parent: Optional[Path] = None) -> str:
    ...


@overload
def abspath_template(path: Path, parent: Optional[Path] = None) -> str:
    ...


def abspath_template(path: PathStr, parent: Optional[Path] = None) -> PathStr:
    """Make path format string absolute."""
    if not isinstance(path, (Path, str)):
        raise TypeError("abspath_template() 'path' must be pathlib.Path or str")
    if str(path).startswith("{"):
        return path
    return abspath(path, parent=parent)


def delete(path: PathStr) -> bool:
    """Remove file or (non-empty) directory."""
    try:
        shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)


def filename_suffix(path: PathStr) -> str:
    """Get filename suffix, including .gz, .bz2 if present."""
    m = re.search("(\\.[a-zA-Z0-9]+(\\.gz|\\.GZ|\\.bz2|\\.BZ2)?)$", str(path))
    return m.group(1) if m else ""


def make_parent_dir(path: PathStr, parents: bool = True, exist_ok: bool = True) -> Path:
    """Make parent directory of file path."""
    parent = Path(path).absolute().parent
    parent.mkdir(parents=parents, exist_ok=exist_ok)
    return parent


def make_temp_file(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[PathStr] = None,
    text: bool = False,
) -> Path:
    """Make temporary file with mkstemp, but close open file handle immediately."""
    fp, path = mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
    os.close(fp)
    return Path(path)


@contextmanager
def temp_dir(
    suffix: str = None, prefix: str = None, dir: Union[Path, str] = None
) -> Generator[Path, None, None]:
    """Create temporary directory within context."""
    path = mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path)


@contextmanager
def temp_file(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: PathStr = None,
    text: bool = False,
) -> Generator[Path, None, None]:
    """Create temporary file within context."""
    path = make_temp_file(suffix=suffix, prefix=prefix, dir=dir, text=text)
    try:
        yield path
    finally:
        os.remove(path)


def unlink_or_mkdir(path: PathStr) -> Path:
    """Unlink existing file or make parent directory if non-existent.

    This function is useful when a script output file is managed by `DVC <https://dvc.org>` using
    protected symbolic or hard links. Call this function before writing the new output file. It will
    remove any existing output file, and ensure that the output directory exists.

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
