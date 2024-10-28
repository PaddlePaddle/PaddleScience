r"""Reference objects stored in AWS Simple Storage Service (S3)."""

from __future__ import annotations  # noqa

import re
from copy import deepcopy
from pathlib import Path
from tempfile import gettempdir
from typing import Generator
from typing import Optional

from deepali.core.pathlib import PathStr
from deepali.core.pathlib import to_uri
from deepali.core.storage import StorageObject

from .client import S3Client


class S3Object(StorageObject):
    r"""Object stored in AWS Simple Storage Service (S3)."""

    def __init__(self, bucket: str, key: str, path: PathStr = None) -> None:
        r"""Initialize AWS S3 object.

        Args:
            bucket: Name of AWS S3 bucket containing this object.
            key: Key of object in AWS S3 bucket with forward slashes as path separators.
                When string ends with a forward slash, the S3 object represents a set
                of S3 objects which share this common key prefix. The local path is
                in this case a directory tree rather than a single file path.
            path: File path of local file or directory corresponding to AWS S3 object.
                When not specified, a temporary path in the tempfile.gettempdir() is
                constructed from the bucket name and S3 object key.

        """
        if bucket is None:
            raise TypeError("S3 bucket name must be str")
        if bucket == "":
            raise ValueError("S3 bucket name must not be an empty string")
        key = self.normkey(key)
        path = Path(path) if path else self.default_path(bucket=bucket, key=key)
        if not key.endswith("/") and path.is_dir():
            key += "/"
        super().__init__(path)
        self._bucket = bucket
        self._key = key

    @property
    def s3(self) -> S3Client:
        """
        Returns:
            client: Underlying connected S3 client.

        """
        return S3Client.default().connect()

    @staticmethod
    def normkey(key: str) -> str:
        r"""Normalize S3 object key string."""
        if not isinstance(key, str):
            raise TypeError("S3 object key must be str")
        if key == "":
            return "/"
        key = re.sub("[/\\\\]{1,}", "/", key)
        if len(key) > 1 and key.startswith("/"):
            key = key[1:]
        return key

    @staticmethod
    def default_path(bucket: str, key: str) -> Path:
        r"""Get default local path of S3 object copy.

        Args:
            bucket: Name of S3 bucket.
            key: Key of S3 object.

        Returns:
            path: Absolute path of local copy of S3 object.

        """
        key = S3Object.normkey(key)
        if key.startswith("/"):
            key = key[1:]
        return Path(gettempdir()).joinpath("deepali", "data", "s3", bucket, key)

    def reset_path(self) -> S3Object:
        r"""Reset ``path`` to ``default_path`` for given ``bucket`` and ``key``.

        Returns:
            self: This instance.

        """
        self._path = self.default_path(bucket=self.bucket, key=self.key)
        return self

    @staticmethod
    def from_path(*args: Optional[PathStr]) -> S3Object:
        r"""Create storage object from path or URI.

        Args:
            args: Path or URI components. The last absolute path or URI is the base to which
                subsequent arguments are appended. Any ``None`` value is ignored. See also ``to_uri()``.

        Returns:
            obj (Resource): Instance of concrete type representing the referenced storage object.

        """
        return S3Object.from_uri(to_uri(*args))

    @classmethod
    def from_uri(cls, uri: str) -> S3Object:
        r"""Create AWS S3 object from URI.

        Args:
            uri: URI of S3 object. Must start with 's3://' followed by the bucket name.
                The remainder of the URI represents the object key, excluding the forward
                slash separating the bucket name from the object key.

        Returns:
            obj: S3 object instance.

        """
        match = re.match("[sS]3://(?P<bucket>[^/]+)/(?P<key>.*)", uri)
        if not match:
            raise ValueError("Invalid AWS S3 object URI: %s", uri)
        return cls(bucket=match["bucket"], key=match["key"])

    @property
    def uri(self) -> str:
        r"""URI of storage object."""
        return "s3://" + self.bucket + "/" + self.key

    @property
    def bucket(self) -> str:
        r"""Name of S3 bucket containing storage object."""
        return self._bucket

    @property
    def key(self) -> str:
        r"""Name of S3 key corresponding to storage object referenced by this instance."""
        return self._key

    @property
    def name(self) -> str:
        r"""Name of storage object including file name extension, excluding directory path."""
        key = self.key
        if key.endswith("/"):
            key = key[:-1]
        return key.rsplit("/", 1)[-1]

    def with_bucket(self, bucket: str) -> S3Object:
        r"""Create copy of storage object reference with modified ``bucket``.

        Args:
            bucket: New bucket name.

        Returns:
            self: New storage object reference with modified ``bucket`` property.

        """
        if bucket is None:
            raise TypeError("Bucket name must be str")
        if bucket == "":
            raise ValueError("Invalid S3 bucket name")
        obj = deepcopy(self)
        obj._bucket = bucket
        return obj

    def with_key(self, key: str) -> S3Object:
        r"""Create copy of storage object reference with modified ``key``.

        Args:
            key: New S3 object key.

        Returns:
            self: New storage object reference with modified ``key`` property.

        """
        if key is None:
            raise TypeError("S3 object key must be str")
        if key == "":
            raise ValueError("Invalid S3 object key")
        obj = deepcopy(self)
        obj._key = key
        return obj

    def exists(self) -> bool:
        r"""Whether object exists in AWS S3."""
        return self.s3.exists(bucket=self.bucket, key=self.key)

    def is_file(self) -> bool:
        r"""Whether AWS S3 object exists and represents a file."""
        return not self.key.endswith("/") and self.exists()

    def is_dir(self) -> bool:
        r"""Whether AWS S3 object exists and represents a directory."""
        return self.key.endswith("/") and self.exists()

    def iterdir(self, prefix: str = None) -> Generator[S3Object, None, None]:
        r"""List S3 objects within directory, excluding subfolder contents.

        Args:
            prefix: Name prefix.

        Returns:
            iterable: Generator of S3 objects.

        """
        if self.key.endswith("/"):
            if prefix is None:
                prefix = ""
            if self.key != "/":
                prefix = self.key + prefix
            for key in self.s3.iterdir(bucket=self.bucket, prefix=prefix):
                yield S3Object(bucket=self.bucket, key=key)

    def pull(self, force: bool = False) -> S3Object:
        r"""Download content from AWS S3 to local path.

        Args:
            force: Whether to force download even if local path already exists.

        Returns:
            self: This instance.

        """
        if self.key.endswith("/"):
            self.s3.download_files(
                bucket=self.bucket, prefix=self.key, path=self.path, overwrite=force
            )
        else:
            try:
                self.s3.download_file(
                    bucket=self.bucket, key=self.key, path=self.path, overwrite=force
                )
            except FileExistsError:
                pass
        return self

    def push(self, force: bool = False) -> S3Object:
        r"""Upload content of local path to AWS S3.

        Args:
            force: Whether to force upload even if S3 object already exists.

        Returns:
            self: This instance.

        """
        if self.key.endswith("/"):
            self.s3.upload_files(
                bucket=self.bucket, prefix=self.key, path=self.path, overwrite=force
            )
        else:
            try:
                self.s3.upload_file(
                    bucket=self.bucket, key=self.key, path=self.path, overwrite=force
                )
            except FileExistsError:
                pass
        return self

    def read_bytes(self) -> bytes:
        r"""Read file content from local path if it exists, or corresponding S3 object otherwise.

        Returns:
            Binary file content of storage object.

        """
        assert not self.key.endswith("/"), "S3 object key must referrence a file object"
        if self.path.is_file():
            return self.path.read_bytes()
        return self.s3.read_bytes(bucket=self.bucket, key=self.key)

    def write_bytes(self, data: bytes) -> S3Object:
        r"""Write bytes directly to storage object.

        Args:
            data: Binary data to write.

        Returns:
            self: This storage object.

        """
        assert not self.key.endswith("/"), "S3 object key must referrence a file object"
        self.s3.write_bytes(bucket=self.bucket, key=self.key, data=data)
        return self

    def read_text(self, encoding: Optional[str] = None) -> str:
        r"""Read text file content from local path if it exists, or corresponding S3 object otherwise.

        Args:
            encoding: Text encoding.

        Returns:
            text: Decoded text file content of storage object.

        """
        data = self.read_bytes()
        return data.decode(encoding) if encoding else data.decode()

    def write_text(self, text: str, encoding: Optional[str] = None) -> S3Object:
        r"""Write text to storage object.

        Args:
            text: Text to write.
            encoding: Text encoding.

        Returns:
            self: This storage object.

        """
        self.write_bytes(text.encode(encoding) if encoding else text.encode())
        return self

    def rmdir(self) -> S3Object:
        r"""Remove S3 objects both locally and from cloud storage."""
        if not self.key.endswith("/"):
            raise NotADirectoryError("S3 object key prefix must end with '/' character")
        super().rmdir()
        self.s3.delete_files(bucket=self.bucket, prefix=self.key)
        return self

    def unlink(self) -> S3Object:
        r"""Remove S3 object both locally and from cloud storage."""
        assert not self.key.endswith("/"), "S3 object key must not end with '/' character"
        super().unlink()
        self.s3.delete_file(bucket=self.bucket, key=self.key)
        return self

    def __repr__(self) -> str:
        r"""Get human-readable string representation of storage object reference."""
        return type(self).__name__ + "(bucket='{}', key='{}', path='{}')".format(
            self.bucket, self.key, self.path
        )
