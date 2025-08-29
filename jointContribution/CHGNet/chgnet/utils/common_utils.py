from __future__ import annotations

import json
import os

import numpy as np
import nvidia_smi
import paddle


def determine_device(
    use_device: (str | None) = None, *, check_cuda_mem: bool = False
) -> str:
    """Determine the device to use for model.

    Args:
        use_device (str): User specify device name
        check_cuda_mem (bool): Whether to return cuda with most available memory
            Default = False

    Returns:
        device (str): device name to be passed to model.to(device)
    """
    use_device = use_device or os.getenv("CHGNET_DEVICE")
    if use_device in {"gpu", None} or paddle.is_compiled_with_cuda():
        device = "gpu"
    else:
        device = "cpu"
    return device


def cuda_devices_sorted_by_free_mem() -> list[int]:
    """List available CUDA devices sorted by increasing available memory.

    To get the device with the most free memory, use the last list item.

    Returns:
        list[int]: CUDA device numbers sorted by increasing free memory.
    """
    if not paddle.device.cuda.device_count() >= 1:
        return []
    free_memories = []
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    for idx in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(idx)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free_memories.append(info.free)
    nvidia_smi.nvmlShutdown()
    return sorted(range(len(free_memories)), key=lambda x: free_memories[x])


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        """Initialize the meter."""
        self.reset()

    def reset(self) -> None:
        """Reset the meter value, average, sum and count to 0."""
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter value, average, sum and count.

        Args:
            val (float): New value to be added to the running average.
            n (int, optional): Number of times the value is added. Default = 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


def mae(prediction: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Computes the mean absolute error between prediction and target.

    Args:
        prediction: Tensor (N, 1)
        target: Tensor (N, 1).

    Returns:
        tensor
    """
    return paddle.mean(x=paddle.abs(x=target - prediction))


def read_json(filepath: str) -> dict:
    """Read the JSON file.

    Args:
        filepath (str): file name of JSON to read.

    Returns:
        dict: data stored in filepath
    """
    with open(filepath) as file:
        return json.load(file)


def write_json(dct: dict, filepath: str) -> dict:
    """Write the JSON file.

    Args:
        dct (dict): dictionary to write
        filepath (str): file name of JSON to write.
    """

    def handler(obj: object) -> (int | object):
        """Convert numpy int64 to int.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(filepath, mode="w") as file:
        json.dump(dct, file, default=handler)


def mkdir(path: str) -> str:
    """Make directory.

    Args:
        path (str): directory name

    Returns:
        path
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path
