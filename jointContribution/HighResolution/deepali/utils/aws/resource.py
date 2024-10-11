from __future__ import annotations

import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Generator
from typing import Optional
from typing import TypeVar
from typing import Union
from urllib.parse import urlsplit

PathStr = Union[Path, str]
PathUri = Union[Path, str]
T = TypeVar("T", bound="Resource")


class Resource(object):
    """Interface for storage objects.

    This base class can be used for storage objects that are only stored locally.
    The base implementations of the ``Resource`` interface functions reflect this use
    case, where ``Resource("/path/to/file")`` represents such local path object.
    The factory function ``Resource.from_uri`` is recommended for creating concrete
    instances of ``Resource`` or one of its subclasses. To create a resource instance
    for a local file path, use ``Resource.from_uri("file:///path/to/file")``.
    An S3 object resource is created by ``Resource.from_uri("s3://bucket/key")``.
    By using the ``Resource`` interface when writing tools that read and write
    from either local, remote, or cloud storage, the tool CLI can create these
    resource instances from input argument URIs or local file path strings
    without URI scheme, i.e., without "file://" prefix. The consumer or producer
    of a resource object can either directly read/write the object data using
    the ``read_(bytes|text)`` and/or ``write_(bytes|text)`` functions, or
    download/upload the storage object to/from a local file path using the
    ``pull`` and ``push`` operations. Note that these operations directly interact
    with the local storage if the resource instance is of base type ``Resource``,
    rather than a remote or cloud storage specific subclass. The ``pull`` and ``push``
    operations should be preferred over ``read`` and ``write`` if the resource data
    is accessed multiple times, in order to take advantage of the local temporary
    copy of the resource object. Otherwise, system IO operations can be saved by
    using the direct ``read`` and ``write`` operations instead.

    Additionally, the ``Resource.release`` function should be called by tools when
    a resource is no longer required to indicate that the local copy of this resource
    can be removed. If the resource object itself represents a local ``Resource``,
    the release operation has no effect. To ensure that the ``release`` function is
    called also in case of an exception, the ``Resource`` class implements the context
    manager interface functions ``__enter__`` and ``__exit__``.

    Example usage with resource context:

    .. code-block:: python

        with Resource.from_uri("s3://bucket/key") as res:
            # request download to local storage
            path = res.pull().path
            # use local storage object referenced by path
        # local copy of storage object has been deleted

    The above is equivalent to using a try-finally block:

    .. code-block:: python

        res = Resource.from_uri("s3://bucket/key")
        try:
            path = res.pull().path
            # use local storage object referenced by path
        finally:
            # delete local copy of storage object
            res.release()

    Usage of the ``with`` statement is recommended.

    Nesting of contexts for a resource object is possible and post-pones the
    invocation of the ``release`` operation until the outermost context has
    been left. This is accomplished by using a counter that is increment by
    ``__enter__``, and decremented again by ``__exit__``.

    It should be noted that ``Resource`` operations are generally not thread-safe,
    and actual consumers of resource objects should require the main thread to
    deal with obtaining, downloading (if ``pull`` is used), and releasing a resource.
    For different resources from remote storage (e.g., AWS S3), when using multiple
    processes (threads), the main process (thread) must initialize the default client
    connection (e.g., using ``S3Client.init_default()``) before spawning processes.

    """

    def __init__(self: T, path: PathStr) -> None:
        """Initialize storage object.

        Args:
            path (str, pathlib.Path): Local path of storage object.

        """
        self._path = Path(path).absolute()
        self._depth = 0

    def __enter__(self: T) -> T:
        """Enter context."""
        self._depth = max(1, self._depth + 1)
        return self

    def __exit__(self: T, *exc) -> None:
        """Release resource when leaving outermost context."""
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self.release()

    @staticmethod
    def from_path(*args: Optional[PathStr]) -> Resource:
        """Create storage object from path or URI.

        Args:
            args: Path or URI components. The last absolute path or URI is the base to which
                subsequent arguments are appended. Any ``None`` value is ignored. See also ``to_uri()``.

        Returns:
            obj (Resource): Instance of concrete type representing the referenced storage object.

        """
        return Resource.from_uri(to_uri(args))

    @staticmethod
    def from_uri(uri: str) -> Resource:
        """Create storage object from URI.

        Args:
            uri: URI of storage object.

        Returns:
            obj (Resource): Instance of concrete type representing the referenced storage object.

        """
        res = urlsplit(uri, scheme="file")
        if res.scheme == "file":
            match = re.match("/+([a-zA-Z]:.*)", res.path)
            path = match.group(1) if match else res.path
            return Resource(Path("/" + res.netloc + "/" + path if res.netloc else path))
        if res.scheme == "s3":
            from .s3.object import S3Object

            return S3Object.from_uri(uri)
        raise ValueError("Invalid or unsupported storage object URI: %s", uri)

    @property
    def uri(self: T) -> str:
        """
        Returns:
            uri (str): URI of storage object.

        """
        return self.path.as_uri()

    @property
    def path(self: T) -> Path:
        """Get absolute local path of storage object."""
        return self._path

    @property
    def name(self: T) -> str:
        """Name of storage object including file name extension, excluding directory path."""
        return self.path.name

    def with_path(self: T, path) -> T:
        """Create copy of storage object reference with modified ``path``.

        Args:
            path (str, pathlib.Path): New local path of storage object.

        Returns:
            self: New storage object reference with modified ``path`` property.

        """
        obj = deepcopy(self)
        obj._path = Path(path).absolute()
        return obj

    def with_properties(self: T, **kwargs) -> T:
        """Create copy of storage object reference with modified properties.

        Args:
            **kwargs: New property values. Only specified properties are changed.

        Returns:
            self: New storage object reference with modified properties.

        """
        obj = deepcopy(self)
        for name, value in kwargs.items():
            setattr(obj, name, value)
        return obj

    def exists(self: T) -> bool:
        """Whether object exists in storage."""
        return self.path.exists()

    def is_file(self: T) -> bool:
        """Whether storage object represents a file."""
        return self.path.is_file()

    def is_dir(self: T) -> bool:
        """Whether storage object represents a directory."""
        return self.path.is_dir()

    def iterdir(self: T, prefix: Optional[str] = None) -> Generator[T, None, None]:
        """List storage objects within directory, excluding subfolder contents.

        Args:
            prefix: Name prefix.

        Returns:
            iterable: Generator of storage objects.

        """
        assert type(self) is Resource, "must be implemented by subclass"
        for path in self.path.iterdir():
            if not prefix or path.name.startswith(prefix):
                yield Resource(path)

    def pull(self: T, force: bool = False) -> T:
        """Download content of storage object to local path.

        Args:
            force (bool): Whether to force download even if local path already exists.

        Returns:
            self: This storage object.

        """
        return self

    def push(self: T, force: bool = False) -> T:
        """Upload content of local path to storage object.

        Args:
            force (bool): Whether to force upload even if storage object already exists.

        Returns:
            self: This storage object.

        """
        return self

    def read_bytes(self: T) -> bytes:
        """Read file content from local path if it exists, or referenced storage object otherwise.

        Returns:
            data (bytes): Binary file content of storage object.

        """
        return self.pull().path.read_bytes()

    def write_bytes(self: T, data: bytes) -> T:
        """Write bytes to storage object.

        Args:
            data (bytes): Binary data to write.

        Returns:
            self: This storage object.

        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)
        return self.push()

    def read_text(self: T, encoding: Optional[str] = None) -> str:
        """Read text file content from local path if it exists, or referenced storage object otherwise.

        Args:
            encoding (str): Text encoding.

        Returns:
            text (str): Decoded text file content of storage object.

        """
        return self.pull().path.read_text()

    def write_text(self: T, text: str, encoding: Optional[str] = None) -> T:
        """Write text to storage object.

        Args:
            text (str): Text to write.
            encoding (str): Text encoding.

        Returns:
            self: This storage object.

        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding=encoding)
        return self.push()

    def rmdir(self: T) -> T:
        """Remove directory both locally and from remote storage."""
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass
        return self

    def unlink(self: T) -> T:
        """Remove file both locally and from remote storage."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return self

    def delete(self: T) -> T:
        """Remove object both locally and from remote storage."""
        try:
            self.rmdir()
        except NotADirectoryError:
            self.unlink()
        return self

    def release(self: T) -> T:
        """Release local temporary copy of storage object.

        Only remove local copy of storage object. When the storage object
        is only stored locally, i.e., self is not a subclass of Resource,
        but of type ``Resource``, this operation does nothing.

        """
        if type(self) is not Resource:
            try:
                shutil.rmtree(self.path)
            except FileNotFoundError:
                pass
            except NotADirectoryError:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        return self

    def __str__(self: T) -> str:
        """Get human-readable string representation of storage object reference."""
        return self.uri

    def __repr__(self: T) -> str:
        """Get human-readable string representation of storage object reference."""
        return type(self).__name__ + "(path='{}')".format(self.path)


def is_absolute(path: Union[Path, str]) -> bool:
    """Check whether given path string or URI is absolute."""
    if is_uri(path):
        return True
    return Path(path).is_absolute()


def is_uri(arg: Any) -> bool:
    """Check whether a given argument is a URI."""
    if isinstance(arg, Path):
        return False
    if isinstance(arg, str):
        if os.name == "nt" and re.match("([a-zA-Z]):[/\\\\](.*)", arg):
            return False
        return re.match("([a-zA-Z0-9]+)://(.*)", arg) is not None
    return False


def to_uri(*args: Optional[PathStr]) -> str:
    """Create valid URI from resource paths.

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
        uri = local_path_uri(base.joinpath(*args))
    else:
        uri = norm_uri(f"{base}/{'/'.join(args)}" if args else base)
    return uri


def norm_uri(uri: str) -> str:
    """Normalize URI.

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
    if scheme == "file":
        if os.name != "nt" or re.match("(?P<drive>[a-zA-Z]):[/\\\\]", path) is None:
            path = "/" + path
        return "file://" + path
    if scheme == "s3":
        return "s3://" + path
    return urlsplit(uri, scheme="file").geturl()


def local_path_uri(arg: PathStr) -> str:
    """Create valid URI from local path.

    Unlike Path.as_uri(), this function does not escape special characters as used in format template strings.

    Args:
        arg: Local path.

    Returns:
        Valid URI.

    """
    uri = norm_uri(f"file://{Path(arg).absolute()}")
    if uri.endswith("/"):
        uri = uri[:-1]
    return uri
