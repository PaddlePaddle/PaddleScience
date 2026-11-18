# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Atomic, shared, read-through cache for the public ERA5 Zarr v2 store."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from pathlib import PurePosixPath


class ERA5Cache(MutableMapping):
    """Cache original compressed objects; never expose an unfinished download.

    Training and a background prefetch process may share this directory. Missing
    objects are fetched from GCS, rather than silently interpreted as NaN chunks.
    """

    def __init__(self, source: str, cache_dir: str | Path):
        import gcsfs

        self.source = source.removeprefix("gs://").rstrip("/")
        digest = hashlib.sha256(self.source.encode()).hexdigest()[:16]
        self.root = Path(cache_dir).resolve() / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.fs = gcsfs.GCSFileSystem(token="anon", timeout=120)
        metadata_bytes = self[".zmetadata"]
        self.metadata = json.loads(metadata_bytes)["metadata"]
        # Consolidated metadata also supplies individual array/group metadata.
        for key, value in self.metadata.items():
            path = self._path(key)
            if not path.exists():
                self._publish(path, json.dumps(value).encode())

    def _path(self, key):
        parts = PurePosixPath(key)
        if parts.is_absolute() or ".." in parts.parts:
            raise KeyError(key)
        return self.root.joinpath(*parts.parts)

    @staticmethod
    def _publish(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def __getitem__(self, key):
        path = self._path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            pass
        for attempt in range(5):
            try:
                content = self.fs.cat_file(f"{self.source}/{key}")
                self._publish(path, content)
                return content
            except FileNotFoundError as error:
                raise KeyError(key) from error
            except (OSError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 16))
        raise RuntimeError("Unreachable download state")

    def __setitem__(self, key, value):
        raise TypeError("ERA5 cache is read-only")

    def __delitem__(self, key):
        raise TypeError("ERA5 cache is read-only")

    def __iter__(self):
        yield ".zmetadata"
        yield from self.metadata
        for key, value in self.metadata.items():
            if not key.endswith("/.zarray"):
                continue
            name = key.removesuffix("/.zarray")
            axes = [
                range(math.ceil(n / c)) for n, c in zip(value["shape"], value["chunks"])
            ]
            separator = value.get("dimension_separator", ".")
            for index in itertools.product(*axes):
                yield f"{name}/{separator.join(map(str, index))}"

    def __len__(self):
        return (
            1
            + len(self.metadata)
            + sum(
                math.prod(
                    math.ceil(n / c) for n, c in zip(value["shape"], value["chunks"])
                )
                for key, value in self.metadata.items()
                if key.endswith("/.zarray")
            )
        )

    def cached(self, key):
        return self._path(key).is_file()

    def object_keys(self, variables, first_time_index, last_time_index):
        """Enumerate complete native chunks intersecting an inclusive time range."""
        for name in variables:
            info = self.metadata[f"{name}/.zarray"]
            dims = self.metadata[f"{name}/.zattrs"]["_ARRAY_DIMENSIONS"]
            axes = []
            for dim, size, chunk in zip(dims, info["shape"], info["chunks"]):
                if dim == "time" and name != "time":
                    axes.append(
                        range(first_time_index // chunk, last_time_index // chunk + 1)
                    )
                else:
                    axes.append(range(math.ceil(size / chunk)))
            separator = info.get("dimension_separator", ".")
            for index in itertools.product(*axes):
                yield f"{name}/{separator.join(map(str, index))}"
