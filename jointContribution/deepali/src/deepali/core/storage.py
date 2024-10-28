r"""Interface for resources stored locally, remotely, or in cloud storage."""

from __future__ import annotations  # noqa

import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Generator
from typing import Optional
from typing import TypeVar
from typing import cast
from urllib.parse import urlsplit

from .pathlib import PathStr
from .pathlib import PathUri
from .pathlib import to_uri
from .pathlib import unlink_or_mkdir

__all__ = (
    "LocalObject",
    "StorageObject",
    "copy_file",
)


TStorageObject = TypeVar("TStorageObject", bound="StorageObject")


class StorageObject(object):
    r"""Interface for storage objects.

    This base class can be used for storage objects that are only stored locally.
    The base implementations of the ``StorageObject`` interface functions reflect this
    use case, where ``StorageObject("/path/to/file")`` represents such local path object.
    The factory function ``StorageObject.from_uri`` is recommended for creating concrete
    instances of ``StorageObject`` or one of its subclasses. To create a resource instance
    for a local file path, use ``StorageObject.from_uri("file:///path/to/file")``.
    An S3 object resource is created by ``StorageObject.from_uri("s3://bucket/key")``.
    By using the ``StorageObject`` interface when writing tools that read and write
    from either local, remote, or cloud storage, the tool CLI can create these
    resource instances from input argument URIs or local file path strings
    without URI scheme, i.e., without "file://" prefix. The consumer or producer
    of a resource object can either directly read/write the object data using
    the ``read_(bytes|text)`` and/or ``write_(bytes|text)`` functions, or
    download/upload the storage object to/from a local file path using the
    ``pull`` and ``push`` operations. Note that these operations directly interact
    with the local storage if the resource instance is of base type ``StorageObject``,
    rather than a remote or cloud storage specific subclass. The ``pull`` and ``push``
    operations should be preferred over ``read`` and ``write`` if the resource data
    is accessed multiple times, in order to take advantage of the local temporary
    copy of the resource object. Otherwise, system IO operations can be saved by
    using the direct ``read`` and ``write`` operations instead.

    Additionally, the ``StorageObject.release()`` function should be called by tools
    when a resource is no longer required to indicate that the local copy of this
    resource can be removed. If the resource object itself represents a local
    ``StorageObject``, the release operation has no effect. To ensure that the ``release``
    function is called also in case of an exception, the ``StorageObject`` class
    implements the context manager interface functions ``__enter__`` and ``__exit__``.

    Example usage with storage object context:

    .. code-block:: python

        with StorageObject.from_uri("s3://bucket/key") as obj:
            # request download to local storage
            path = obj.pull().path
            # use local storage object referenced by path
        # local copy of storage object has been deleted

    The above is equivalent to using a try-finally block:

    .. code-block:: python

        obj = StorageObject.from_uri("s3://bucket/key")
        try:
            path = obj.pull().path
            # use local storage object referenced by path
        finally:
            # delete local copy of storage object
            obj.release()

    Usage of the ``with`` statement is recommended.

    Nesting of contexts for a resource object is possible and post-pones the
    invocation of the ``release`` operation until the outermost context has
    been left. This is accomplished by using a counter that is increment by
    ``__enter__``, and decremented again by ``__exit__``.

    It should be noted that ``StorageObject`` operations are generally not thread-safe,
    and actual consumers of resource objects should require the main thread to deal with
    obtaining, downloading (if ``pull`` is used), and releasing a resource. For different
    resources from remote storage (e.g., AWS S3), when using multiple processes (threads),
    the main process (thread) must initialize the default client connection (e.g., using
    ``S3Client.init_default()``) before spawning processes.

    """

    def __init__(self, path: PathStr) -> None:
        r"""Initialize storage object.

        Args:
            path (str, pathlib.Path): Local path of storage object.

        """
        self._path = Path(path).absolute()
        self._depth = 0

    def __enter__(self: TStorageObject) -> TStorageObject:
        r"""Enter context."""
        self._depth = max(1, self._depth + 1)
        return self

    def __exit__(self, *exc) -> None:
        r"""Release resource when leaving outermost context."""
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self.release()

    @staticmethod
    def from_path(*args: Optional[PathStr]) -> StorageObject:
        r"""Create storage object from path or URI.

        Args:
            args: Path or URI components. The last absolute path or URI is the base to which
                subsequent arguments are appended. Any ``None`` value is ignored. See also ``to_uri()``.

        Returns:
            obj (Resource): Instance of concrete type representing the referenced storage object.

        """
        return StorageObject.from_uri(to_uri(*args))

    @staticmethod
    def from_uri(uri: str) -> StorageObject:
        r"""Create storage object from URI.

        Args:
            uri: URI of storage object.

        Returns:
            Instance of concrete type representing the referenced storage object.

        """
        res = urlsplit(uri, scheme="file")
        if res.scheme == "file":
            match = re.match(r"/+([a-zA-Z]:.*)", res.path)
            path = match.group(1) if match else res.path
            return LocalObject(Path("/" + res.netloc + "/" + path if res.netloc else path))
        if res.scheme == "s3":
            # DO NOT import at module level to avoid cyclical import!
            try:
                from deepali.utils.aws.s3.object import S3Object
            except ImportError:
                raise ImportError("StorageObject.from_uri() requires AWS S3 deepali[utils]")
            return cast(StorageObject, S3Object.from_uri(uri))
        raise ValueError("Invalid or unsupported storage object URI: %s", uri)

    @property
    def uri(self) -> str:
        r"""
        Returns:
            uri (str): URI of storage object.

        """
        return self.path.as_uri()

    @property
    def path(self) -> Path:
        r"""Get absolute local path of storage object."""
        return self._path

    @property
    def name(self) -> str:
        r"""Name of storage object including file name extension, excluding directory path."""
        return self.path.name

    def with_path(self: TStorageObject, path) -> TStorageObject:
        r"""Create copy of storage object reference with modified ``path``.

        Args:
            path (str, pathlib.Path): New local path of storage object.

        Returns:
            New storage object reference with modified ``path`` property.

        """
        obj = deepcopy(self)
        obj._path = Path(path).absolute()
        return obj

    def with_properties(self: TStorageObject, **kwargs) -> TStorageObject:
        r"""Create copy of storage object reference with modified properties.

        Args:
            **kwargs: New property values. Only specified properties are changed.

        Returns:
            New storage object reference with modified properties.

        """
        obj = deepcopy(self)
        for name, value in kwargs.items():
            setattr(obj, name, value)
        return obj

    def exists(self) -> bool:
        r"""Whether object exists in storage."""
        return self.path.exists()

    def is_file(self) -> bool:
        r"""Whether storage object represents a file."""
        return self.path.is_file()

    def is_dir(self) -> bool:
        r"""Whether storage object represents a directory."""
        return self.path.is_dir()

    def iterdir(
        self: TStorageObject, prefix: Optional[str] = None
    ) -> Generator[TStorageObject, None, None]:
        r"""List storage objects within directory, excluding subfolder contents.

        Args:
            prefix: Name prefix.

        Returns:
            Generator of storage objects.

        """
        assert type(self) is StorageObject, "must be implemented by subclass"
        for path in self.path.iterdir():
            if not prefix or path.name.startswith(prefix):
                yield cast(TStorageObject, StorageObject(path))

    def pull(self: TStorageObject, force: bool = False) -> TStorageObject:
        r"""Download content of storage object to local path.

        Args:
            force (bool): Whether to force download even if local path already exists.

        Returns:
            This storage object.

        """
        return self

    def push(self: TStorageObject, force: bool = False) -> TStorageObject:
        r"""Upload content of local path to storage object.

        Args:
            force (bool): Whether to force upload even if storage object already exists.

        Returns:
            This storage object.

        """
        return self

    def read_bytes(self) -> bytes:
        r"""Read file content from local path if it exists, or referenced storage object otherwise.

        Returns:
            Binary file content of storage object.

        """
        return self.pull().path.read_bytes()

    def write_bytes(self: TStorageObject, data: bytes) -> TStorageObject:
        r"""Write bytes to storage object.

        Args:
            data: Binary data to write.

        Returns:
            This storage object.

        """
        try:
            self.path.unlink()
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)
        return self.push()

    def read_text(self, encoding: Optional[str] = None) -> str:
        r"""Read text file content from local path if it exists, or referenced storage object otherwise.

        Args:
            encoding: Text encoding.

        Returns:
            Decoded text file content of storage object.

        """
        return self.pull().path.read_text()

    def write_text(
        self: TStorageObject, text: str, encoding: Optional[str] = None
    ) -> TStorageObject:
        r"""Write text to storage object.

        Args:
            text: Text to write.
            encoding: Text encoding.

        Returns:
            This storage object.

        """
        try:
            self.path.unlink()
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding=encoding)
        return self.push()

    def rmdir(self: TStorageObject) -> TStorageObject:
        r"""Remove directory both locally and from remote storage."""
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass
        return self

    def unlink(self: TStorageObject) -> TStorageObject:
        r"""Remove file both locally and from remote storage."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return self

    def delete(self: TStorageObject) -> TStorageObject:
        r"""Remove object both locally and from remote storage."""
        try:
            self.rmdir()
        except NotADirectoryError:
            self.unlink()
        return self

    def release(self: TStorageObject) -> TStorageObject:
        r"""Release local temporary copy of storage object.

        Only remove local copy of storage object. When the storage object
        is only stored locally, i.e., self is not a subclass of StorageObject,
        but of type ``StorageObject``, this operation does nothing.

        """
        if type(self) is not StorageObject:
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

    def __str__(self) -> str:
        r"""Get human-readable string representation of storage object reference."""
        return self.uri

    def __repr__(self) -> str:
        r"""Get human-readable string representation of storage object reference."""
        return type(self).__name__ + "(path='{}')".format(self.path)


class LocalObject(StorageObject):
    r"""Local storage object."""

    def release(self: TStorageObject) -> TStorageObject:
        r"""Release local temporary copy of storage object."""
        return self


def copy_file(src: PathStr, dst: PathUri) -> StorageObject:
    r"""Copy local file to specified path or URI."""
    obj = StorageObject.from_uri(to_uri(dst))
    if isinstance(obj, LocalObject):
        path = unlink_or_mkdir(obj.path)
        shutil.copy2(src, path)
    else:
        obj.with_path(src).push(force=True)
    return obj
