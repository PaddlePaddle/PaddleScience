import multiprocessing as mp

from psutil import virtual_memory


def cpu_count() -> int:
    r"""Get number of available CPUs.

    If running inside a Linux Docker container, this returns the limit set when creating the container.

    """
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())
        # For physical machine, the `cfs_quota_us` could be '-1'
        num_cpus = cfs_quota_us // cfs_period_us
    except FileNotFoundError:
        num_cpus = -1
    if num_cpus < 1:
        num_cpus = mp.cpu_count()
    return num_cpus


def memory_limit() -> int:
    r"""Get memory limit in bytes.

    If running inside a Linux Docker container, this returns the limit set when creating the container.

    """
    limit_in_bytes = virtual_memory().total
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as fp:
            limit_in_bytes = min(int(fp.read()), limit_in_bytes)
    except FileNotFoundError:
        pass
    return limit_in_bytes
