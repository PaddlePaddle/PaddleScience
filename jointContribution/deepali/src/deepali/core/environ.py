r"""Utilities for parsing and setting up of common environment variables."""

import os
import re
from typing import Optional
from typing import Tuple


def cuda_visible_devices() -> Tuple[int, ...]:
    r"""Get IDs of GPUs specified by CUDA_VISIBLE_DEVICES environment variable."""
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpus:
        return ()
    gpus = [x for x in gpus.split(",") if x]
    gpu_ids = []
    RE_RANGE = re.compile(r"^(?P<start>[0-9]+)-(?P<end>[0-9]+)$")
    for gpu in gpus:
        match = RE_RANGE.match(gpu)
        if match is None:
            try:
                gpu_ids.append(int(gpu))
            except TypeError:
                raise TypeError(f"CUDA_VISIBLE_DEVICES contains invalid value {gpu}")
        else:
            gpu_ids.extend(range(int(match.group("start")), int(match.group("end")) + 1))
    for gpu_id in gpu_ids:
        if gpu_id < 0:
            raise ValueError("CUDA_VISIBLE_DEVICES contains negative GPU ID")
    return gpu_ids


def check_cuda_visible_devices(num: Optional[int] = None) -> int:
    r"""Check if CUDA_VISIBLE_DEVICES environment variable is set."""
    gpu_ids = cuda_visible_devices()
    if num and len(gpu_ids) != num:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be set to {num} GPUs")
    return len(gpu_ids)


def init_num_threads(threads: int) -> None:
    r"""Set environment variables used to limit number of process threads."""
    if threads < 0:
        raise ValueError("init_num_threads() 'threads' must not be negative")
    threads = max(1, int(threads))
    for env in (
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[env] = str(threads)
    try:
        import SimpleITK as sitk

        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    except ImportError:
        pass


def init_omp_num_threads(threads: Optional[int] = None, default: int = 1) -> int:
    r"""Set environment variable OMP_NUM_THREADS to limit number of process threads."""
    if threads is None or threads < 0:
        threads = os.environ.get("OMP_NUM_THREADS", default)
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    return threads
