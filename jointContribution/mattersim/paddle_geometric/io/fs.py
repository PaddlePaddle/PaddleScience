import io
import os.path as osp
import pickle
import re
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Union, overload
from uuid import uuid4

import fsspec
import paddle

DEFAULT_CACHE_PATH = '/tmp/paddle_simplecache'


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    r"""Get filesystem backend given a path URI to the resource.

    Common example paths and dispatch result:

    * :obj:`"/home/file"` -> LocalFileSystem
    * :obj:`"memory://home/file"` -> MemoryFileSystem
    * :obj:`"https://home/file"` -> HTTPFileSystem
    * :obj:`"gs://home/file"` -> GCSFileSystem
    * :obj:`"s3://home/file"` -> S3FileSystem

    Args:
        path (str): The URI to the filesystem location, *e.g.*,
            :obj:`"gs://home/me/file"`, :obj:`"s3://..."`.
    """
    return fsspec.core.url_to_fs(path)[0]


def normpath(path: str) -> str:
    if isdisk(path):
        return osp.normpath(path)
    return path


def exists(path: str) -> bool:
    return get_fs(path).exists(path)


def makedirs(path: str, exist_ok: bool = True) -> None:
    return get_fs(path).makedirs(path, exist_ok)


def isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def isfile(path: str) -> bool:
    return get_fs(path).isfile(path)


def isdisk(path: str) -> bool:
    return 'file' in get_fs(path).protocol


def islocal(path: str) -> bool:
    return isdisk(path) or 'memory' in get_fs(path).protocol


@overload
def ls(path: str, detail: Literal[False] = False) -> List[str]:
    pass


@overload
def ls(path: str, detail: Literal[True]) -> List[Dict[str, Any]]:
    pass


def ls(
    path: str,
    detail: bool = False,
) -> Union[List[str], List[Dict[str, Any]]]:
    fs = get_fs(path)
    outputs = fs.ls(path, detail=detail)

    if not isdisk(path):
        if detail:
            for output in outputs:
                output['name'] = fs.unstrip_protocol(output['name'])
        else:
            outputs = [fs.unstrip_protocol(output) for output in outputs]

    return outputs


def cp(
    path1: str,
    path2: str,
    extract: bool = False,
    log: bool = True,
    use_cache: bool = True,
    clear_cache: bool = True,
) -> None:
    kwargs: Dict[str, Any] = {}

    is_path1_dir = isdir(path1)
    is_path2_dir = isdir(path2)

    # Cache result if the protocol is not local:
    cache_dir: Optional[str] = None
    if not islocal(path1):
        if log and 'pytest' not in sys.modules:
            print(f'Downloading {path1}', file=sys.stderr)

        if extract and use_cache:
            cache_dir = osp.join(DEFAULT_CACHE_PATH, uuid4().hex)
            kwargs.setdefault('simplecache', dict(cache_storage=cache_dir))
            path1 = f'simplecache::{path1}'

    # Handle automatic extraction:
    multiple_files = False
    if extract and path1.endswith('.tar.gz'):
        kwargs.setdefault('tar', dict(compression='gzip'))
        path1 = f'tar://**::{path1}'
        multiple_files = True
    elif extract and path1.endswith('.zip'):
        path1 = f'zip://**::{path1}'
        multiple_files = True
    elif extract and path1.endswith('.gz'):
        kwargs.setdefault('compression', 'infer')
    elif extract:
        raise NotImplementedError(
            f"Automatic extraction of '{path1}' not yet supported")

    # Perform the copy:
    for open_file in fsspec.open_files(path1, **kwargs):
        with open_file as f_from:
            if not multiple_files:
                if is_path2_dir:
                    basename = osp.basename(path1)
                    if extract and path1.endswith('.gz'):
                        basename = '.'.join(basename.split('.')[:-1])
                    to_path = osp.join(path2, basename)
                else:
                    to_path = path2
            else:
                common_path = osp.commonprefix(
                    [fsspec.core.strip_protocol(path1), open_file.path])
                to_path = osp.join(path2, open_file.path[len(common_path):])
            with fsspec.open(to_path, 'wb') as f_to:
                while True:
                    chunk = f_from.read(10 * 1024 * 1024)
                    if not chunk:
                        break
                    f_to.write(chunk)

    if use_cache and clear_cache and cache_dir is not None:
        try:
            rm(cache_dir)
        except Exception:
            pass


def rm(path: str, recursive: bool = True) -> None:
    get_fs(path).rm(path, recursive)


def mv(path1: str, path2: str) -> None:
    fs1 = get_fs(path1)
    fs2 = get_fs(path2)
    assert fs1.protocol == fs2.protocol
    fs1.mv(path1, path2)


def glob(path: str) -> List[str]:
    fs = get_fs(path)
    paths = fs.glob(path)

    if not isdisk(path):
        paths = [fs.unstrip_protocol(path) for path in paths]

    return paths


def paddle_save(data: Any, path: str) -> None:
    buffer = io.BytesIO()
    paddle.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())


def paddle_load(path: str, map_location: Any = None) -> Any:
    with fsspec.open(path, 'rb') as f:
        return paddle.load(f, map_location=map_location)
