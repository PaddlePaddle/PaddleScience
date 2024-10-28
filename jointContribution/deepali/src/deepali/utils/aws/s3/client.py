r"""Client for AWS Simple Storage Service (S3)."""
from __future__ import annotations  # noqa

import io
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import boto3
from botocore.exceptions import ClientError

from .config import S3Config

S3ClientError = ClientError
PathStr = Union[Path, str]


def match_all(key: str) -> bool:
    r"""Default 'match' function used by 'S3Client.download_files()'."""
    return True


class S3Client(object):
    r"""Client for AWS Simple Storage Service (S3)."""

    class Operation(Enum):
        r"""Enumeration of permissible S3 operations.

        Note that these permissions restrict the set of allowed operations and
        is independent of S3 object permissions which are configured in AWS S3.

        """

        READ = "r"
        WRITE = "w"
        DELETE = "d"

    _default: List[S3Client] = []

    @classmethod
    def default(cls, client: Optional[S3Client] = None) -> S3Client:
        r"""Default client instance.

        This function is not thread-safe if not called first by the main thread.

        Args:
            client: If ``None``, a new client instance is created if no current default client exists.
                Otherwise, the given client instance replaces the current default instance.

        Returns:
            Current default client instance. If no default client instance has been set before,
                a new instance with default configuration is created.

        """
        if client is not None:
            if cls._default:
                cls._default[-1] = client
            else:
                cls._default.append(client)
        elif not cls._default:
            cls._default.append(cls())
        return cls._default[-1]

    @classmethod
    @contextmanager
    def init_default(cls, *args, **kwargs) -> Generator[S3Client, None, None]:
        r"""Set new default client to use in current context.

        This function is not thread-safe. Only the main thread may set the default client
        instance to use by worker threads. Any multi-threaded operations should share
        a client instance, a reference to which should be provided by the main thread.
        Nested ``with`` statements using this context manager function are supported,
        as long as these occur in the main thread only.

        Args:
            *args: Positional arguments for client ``__init__`` function call.
            **kwargs: Keyword arguments for client ``__init__`` function call.

        Returns:
            client (S3Client): Generator function that returns the connected default client
                instance and resets the default to the previously active one upon completion.

        """
        client = cls(*args, **kwargs)
        client.connect()
        cls._default.append(client)
        try:
            yield client
        finally:
            client = cls._default.pop()
            client.close()

    def __init__(self, config: Optional[S3Config] = None, **kwargs) -> None:
        r"""Initialize S3 client.

        Args:
            config: Client configuration.
            **kwargs: Individual client configuration settings.

        """
        if config is None:
            config = S3Config()
        self._config = config._replace(**kwargs)
        self._client = None
        self._depth = 0

    @property
    def exceptions(self):
        r"""Get boto3 exceptions for connected client."""
        if self._client is None:
            raise AssertionError(
                f"{type(self).__name__} must be connected to access exceptions. Mabye use botocore.exceptions module instead."
            )
        return self._client.exceptions

    @classmethod
    def from_arg(cls, arg: Union[S3Config, S3Client, Dict[str, Any], None]) -> S3Client:
        r"""Get client instance given function argument.

        Args:
            arg: Function argument.

        Returns:
            client: S3 client instance. If ``arg`` is an S3Client instance,
                this instance is returned. If ``arg`` is of type S3Config or dict, a new
                instance is returned, which has been initialized with this configuration.
                If ``arg`` is ``None``, the default client instance is returned.

        """
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, S3Config):
            return cls(arg)
        if isinstance(arg, dict):
            return cls(**arg)
        return cls.default()

    @property
    def config(self) -> S3Config:
        r"""Get client configuration object."""
        return self._config

    @property
    def ops(self) -> Set[S3Client.Operation]:
        r"""Get list of permissible client operations."""
        return set(S3Client.Operation(c) for c in self.config.mode)

    def connect(self) -> S3Client:
        r"""Establish connection with AWS S3."""
        if self._client is None:
            self._client = boto3.client(
                "s3", region_name=self.config.region, verify=self.config.verify
            )
        return self

    def close(self) -> None:
        r"""Close connection with AWS S3."""
        self._client = None

    def is_closed(self) -> bool:
        r"""Check if S3 connection is closed."""
        return self._client is None

    def is_open(self) -> bool:
        r"""Check if client connection is open."""
        return not self.is_closed()

    def __enter__(self) -> S3Client:
        r"""Ensure client connection is open when entering context.

        This function increments a context depth counter in a non-thread-safe way.
        Only the main thread should create client connection and pass these on to
        worker threads that require access to this resource.

        """
        self.connect()
        self._depth = max(1, self._depth + 1)
        return self

    def __exit__(self, *exc) -> None:
        r"""Close connection when exiting outermost context."""
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self.close()

    def exists(self, bucket: str, key: str) -> bool:
        r"""Check if specified S3 object exists.

        Args:
            bucket: Bucket name.
            key: S3 object key. To query the existence of a folder, the
                key must end with a forward slash character ('/').

        Returns:
            Whether S3 object exists in specified bucket.

        """
        assert not self.is_closed(), "client connection required"
        if key.endswith("/"):
            resp = self._client.list_objects_v2(Bucket=bucket, Prefix=key)
            flag = bool(resp.get("Contents", []))
        else:
            flag = False
            try:
                self._client.head_object(Bucket=bucket, Key=key)
                flag = True
            except self._client.exceptions.ClientError as error:
                error_info = getattr(error, "response", {}).get("Error", {})
                if error_info.get("Code") != "404":
                    raise error
            except self._client.exceptions.NoSuchKey:
                pass
        return flag

    def keys(self, bucket: str, prefix: Optional[str] = None) -> Generator[str, None, None]:
        r"""List S3 objects with specified key prefix.

        Args:
            bucket: Bucket name.
            prefix: Common key prefix.

        Returns:
            Generator of S3 object keys.

        Raises:
            PermissionError: If read operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        if S3Client.Operation.READ not in self.ops:
            raise PermissionError("S3 client has no read permissions")
        kwargs = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix
        while True:
            resp: dict = self._client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                yield obj["Key"]
            try:
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            except KeyError:
                break

    def iterdir(self, bucket: str, prefix: Optional[str] = None) -> Generator[str, None, None]:
        r"""List S3 objects with specified key prefix, excluding subfolder contents.

        Args:
            bucket: Bucket name.
            prefix: Common key prefix. To list the contents of a folder, the
                key prefix must end with a forward slash character ('/'), otherwise
                the result will only contain the folder itself.

        Returns:
            Generator of S3 object keys, where subfolders are represented
                by keys ending with a forward slash character ('/').

        Raises:
            PermissionError: If read operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        if S3Client.Operation.READ not in self.ops:
            raise PermissionError("S3 client has no read permissions")
        kwargs = {"Bucket": bucket, "Delimiter": "/"}
        if prefix:
            kwargs["Prefix"] = prefix
        while True:
            resp: dict = self._client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                yield obj["Key"]
            for obj in resp.get("CommonPrefixes", []):
                yield obj["Prefix"]
            try:
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            except KeyError:
                break

    def listdir(self, bucket: str, prefix: Optional[str] = None) -> Generator[str, None, None]:
        r"""List names of S3 objects whose keys match a given prefix, excluding subfolder contents.

        This convenience function uses ``iterdir`` to obtain the keys of the S3 objects matching
        a given ``prefix`` that correspond to the respective folder, and removes the folder key
        prefix from the resulting object keys. It thus behaves similar to ``os.listdir``.

        Args:
            bucket: Bucket name.
            prefix: Common key prefix. To list the contents of a folder, the
                key prefix must end with a forward slash character ('/'), otherwise
                the result will only contain the folder itself.

        Returns:
            Generator of S3 object keys, where subfolders are represented
                by keys ending with a forward slash character ('/').

        Raises:
            PermissionError: If read operations have not been enabled for this client.

        """
        skip = (prefix or "").rfind("/") + 1
        for key in self.iterdir(bucket, prefix):
            yield key[skip:]

    def read_bytes(self, bucket: str, key: str) -> bytes:
        r"""Download binary object data.

        Args:
            bucket: Bucket name.
            key: S3 object key.

        Returns:
            data: S3 object data.

        Raises:
            PermissionError: If read operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        assert key, "S3 object key must be specified"
        if S3Client.Operation.READ not in self.ops:
            raise PermissionError("S3 client has no read permissions")
        buffer = io.BytesIO()
        config = self._config.transfer_config()
        self._client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer, Config=config)
        return buffer.getvalue()

    def write_bytes(self, bucket: str, key: str, data: bytes) -> None:
        r"""Upload binary object data.

        Args:
            bucket: Bucket name.
            key: S3 object key.
            data: S3 object data.

        Raises:
            PermissionError: If write operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        assert key, "S3 object key must be specified"
        if S3Client.Operation.WRITE not in self.ops:
            raise PermissionError("S3 client has no write permissions")
        buffer = io.BytesIO(data)
        config = self._config.transfer_config()
        self._client.upload_fileobj(Fileobj=buffer, Bucket=bucket, Key=key, Config=config)

    def read_text(self, bucket: str, key: str, encoding: Optional[str] = None) -> str:
        r"""Download text file content.

        Args:
            bucket: Bucket name.
            key: S3 object key.
            encoding: Text encoding.

        Returns:
            text: Decoded text.

        Raises:
            PermissionError: If read operations have not been enabled for this client.

        """
        data = self.read_bytes(bucket, key)
        return data.decode(encoding) if encoding else data.decode()

    def write_text(self, bucket: str, key: str, text: str, encoding: Optional[str] = None) -> None:
        r"""Upload text file content.

        Args:
            bucket: Bucket name.
            key: S3 object key.
            text: S3 object data.
            encoding: Text encoding.

        Raises:
            PermissionError: If write operations have not been enabled for this client.

        """
        data = text.encode(encoding) if encoding else text.encode()
        self.write_bytes(bucket=bucket, key=key, data=data)

    def download_file(self, bucket: str, key: str, path: PathStr, overwrite: bool = True) -> None:
        r"""Download S3 object to local file.

        Args:
            bucket: Bucket name.
            key: S3 object key.
            path: Local file path or path of existing output directory.
            overwrite: Whether to overwrite existing local file.

        Raises:
            PermissionError: If read operations have not been enabled for this client.
            FileExistsError: If local ``path`` exists and ``overwrite=False``.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        assert key, "S3 object key must be specified"
        if S3Client.Operation.READ not in self.ops:
            raise PermissionError("S3 client has no read permissions")
        path = Path(path).absolute()
        if path.is_dir():
            path = path.joinpath(key.rsplit("/", 1)[-1])
        if not overwrite and path.is_file():
            raise FileExistsError(
                "Use overwrite=True to force overwriting existing local file '{}'".format(path)
            )
        data = self.read_bytes(bucket, key)
        try:
            path.unlink()
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(data)
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as e:
            try:
                path.unlink()
            except Exception:
                pass
            raise e

    def download_files(
        self,
        bucket: str,
        prefix: str,
        path: PathStr,
        match: Optional[Callable[[str], bool]] = None,
        overwrite: bool = True,
    ) -> Tuple[int, int]:
        r"""Download S3 objects to local directory.

        Args:
            bucket: Bucket name.
            prefix: Common key prefix.
            path: Local directory path. If the directory exists, downloaded
                files are merged into this existing directory and subdirectories.
            match: Filter function that takes a subkey without ``prefix`` and
                returns either True or False. If False, the corresponding file is skipped.
            overwrite: Whether to overwrite existing local files.

        Returns:
            total: Number of matching S3 objects.
            count: Number of actually downloaded files.

        Raises:
            PermissionError: If read operations have not been enabled for this client.

        """
        total = 0
        count = 0
        skip = prefix.rfind("/") + 1
        if match is None:
            match = match_all
        for key in self.keys(bucket=bucket, prefix=prefix):
            subkey = key[skip:]
            if subkey and match(subkey):
                total += 1
                try:
                    self.download_file(
                        bucket=bucket, key=key, path=path.joinpath(subkey), overwrite=overwrite
                    )
                    count += 1
                except FileExistsError:
                    pass
        return total, count

    def upload_file(self, path: PathStr, bucket: str, key: str, overwrite: bool = True) -> None:
        r"""Upload local file to S3.

        Args:
            path: Local file path.
            bucket: Bucket name.
            key: S3 object key.
            overwrite: Whether to overwrite existing S3 object.

        Raises:
            PermissionError: If write operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        assert key, "S3 object key must be specified"
        assert not key.endswith("/"), "S3 object key must not end with forward slash"
        if S3Client.Operation.WRITE not in self.ops:
            raise PermissionError("S3 client has no write permissions")
        if not overwrite and self.exists(bucket=bucket, key=key):
            raise FileExistsError(
                "Use overwrite=True to force overwriting "
                + "existing S3 object '{k}' in bucket {b}".format(k=key, b=bucket)
            )
        path = Path(path).absolute()
        self.write_bytes(bucket=bucket, key=key, data=path.read_bytes())

    def upload_files(
        self, path: PathStr, bucket: str, prefix: Optional[str] = None, overwrite: bool = True
    ) -> Tuple[int, int]:
        r"""Upload local directory to S3.

        Args:
            path: Local directory path.
            bucket: Bucket name.
            prefix: Common S3 object key prefix. If ``None`` or empty string,
                the content of the local directory is uploaded to the specified
                ``bucket`` without any object key prefix, i.e., the files will
                be located in S3 directly underneath the destination bucket.
            overwrite: Whether to overwrite existing S3 objects.

        Returns:
            total: Number of local files.
            count: Number of uploaded files.

        Raises:
            PermissionError: If write operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        total = 0
        count = 0
        if prefix is None:
            prefix = "/"
        if not prefix.endswith("/"):
            prefix += "/"
        path = Path(path).absolute()
        for child in path.iterdir():
            key = prefix + child.name
            if child.is_dir():
                tot, cnt = self.upload_files(
                    path=child, bucket=bucket, prefix=key, overwrite=overwrite
                )
                total += tot
                count += cnt
            else:
                try:
                    self.upload_file(path=child, bucket=bucket, key=key, overwrite=overwrite)
                    count += 1
                except FileExistsError:
                    pass
                total += 1
        return total, count

    def delete_file(self, bucket: str, key: str) -> int:
        r"""Delete S3 object.

        Args:
            bucket: Bucket name.
            key: S3 object key.

        Returns:
            count: Number of deleted S3 objects (0 or 1).

        Raises:
            PermissionError: If delete operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        if S3Client.Operation.DELETE not in self.ops:
            raise PermissionError("S3 client has no permission to delete objects")
        resp = self._client.delete_object(Bucket=bucket, Key=key)
        return 1 if "DeleteMarker" in resp else 0

    def delete_files(self, bucket: str, prefix: str) -> int:
        r"""Delete S3 objects.

        Args:
            bucket: Bucket name.
            prefix: Common key prefix.

        Returns
            count: Number of deleted S3 objects.

        Raises:
            PermissionError: If delete operations have not been enabled for this client.

        """
        assert not self.is_closed(), "client connection required"
        assert bucket, "S3 bucket must be specified"
        if S3Client.Operation.DELETE not in self.ops:
            raise PermissionError("S3 client has no permission to delete objects")
        count = 0
        list_kwargs = {"Bucket": bucket, "Prefix": prefix}
        del_kwargs = {"Bucket": bucket, "Delete": {"Objects": []}}
        while True:
            list_resp = self._client.list_objects_v2(**list_kwargs)
            del_kwargs["Delete"]["Objects"] = [
                {"Key": obj["Key"]} for obj in list_resp.get("Contents", [])
            ]
            if del_kwargs["Delete"]["Objects"]:
                count += len(self._client.delete_objects(**del_kwargs)["Deleted"])
            try:
                list_resp["ContinuationToken"] = list_resp["NextContinuationToken"]
            except KeyError:
                break
        return count
