import warnings
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

import paddle
from paddle.io import DataLoader

class DeviceHelper:
    def __init__(self, device: Optional[str] = None):
        with_cuda = paddle.is_compiled_with_cuda()

        if device is None:
            if with_cuda:
                device = 'gpu'
            else:
                device = 'cpu'

        self.device = paddle.get_device(device)
        self.is_gpu = self.device.startswith('gpu')

        if self.is_gpu and not with_cuda:
            warnings.warn(f"Requested device '{self.device}' is not available, falling back to CPU")
            self.device = 'cpu'

        self.stream = None
        self.stream_context = nullcontext

    def maybe_init_stream(self) -> None:
        # Paddle does not support streams directly, so we can omit stream initialization.
        pass

    def maybe_wait_stream(self) -> None:
        # Paddle does not support stream management as PyTorch does, so this can be omitted.
        pass


class PrefetchLoader:
    r"""A GPU prefetcher class for asynchronously transferring data of a
    :class:`paddle.io.DataLoader` from host memory to device memory.

    Args:
        loader (paddle.io.DataLoader): The data loader.
        device (str, optional): The device to load the data to.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[str] = None,
    ):
        self.loader = loader
        self.device_helper = DeviceHelper(device)

    def non_blocking_transfer(self, batch: Any) -> Any:
        if not self.device_helper.is_gpu:
            return batch
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        # In Paddle, we use `to()` method to move tensors to the correct device.
        batch = paddle.to_tensor(batch)  # Ensure it's a tensor
        return batch

    def __iter__(self) -> Any:
        first = True
        self.device_helper.maybe_init_stream()

        batch = None
        for next_batch in self.loader:
            # Transfer data to the correct device in a non-blocking way
            next_batch = self.non_blocking_transfer(next_batch)

            if not first:
                yield batch
            else:
                first = False

            self.device_helper.maybe_wait_stream()

            batch = next_batch

        yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
