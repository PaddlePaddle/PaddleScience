import logging
import warnings
from itertools import chain

import paddle
from paddle import nn
from paddle_geometric.data import Batch
from paddle_geometric.utils import cumsum

if paddle.distributed.get_world_size() > 1:
    from paddle.distributed import parallel_apply


class DataParallel(nn.Layer):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting a list of :class:`paddle_geometric.data.Data` objects and copying
    them as :class:`paddle_geometric.data.Batch` objects to each device.
    In the forward pass, the module is replicated on each device, and each
    replica handles a portion of the input.
    During the backwards pass, gradients from each replica are summed into the
    original module.

    The batch size should be larger than the number of GPUs used.

    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or paddle.device): CUDA devices.
            (default: all devices)
        output_device (int or paddle.device): Device location of output.
            (default: :obj:`device_ids[0]`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): Will exclude each key in the
            list. (default: :obj:`None`)
    """

    def __init__(self, module, device_ids=None, output_device=None,
                 follow_batch=None, exclude_keys=None):
        super(DataParallel, self).__init__()

        warnings.warn("'DataParallel' is usually much slower than "
                      "'DistributedDataParallel' even on a single machine. "
                      "Please consider switching to 'DistributedDataParallel' "
                      "for multi-GPU training.")

        self.module = module
        self.device_ids = device_ids or paddle.device.get_device()
        self.output_device = output_device or self.device_ids[0]
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []

    def forward(self, data_list):
        """"""  # noqa: D419
        if len(data_list) == 0:
            logging.warning('DataParallel received an empty data list, which '
                            'may result in unexpected behavior.')
            return None

        if len(self.device_ids) == 1:  # Fallback
            data = Batch.from_data_list(
                data_list, follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys).to(self.device_ids[0])
            return self.module(data)

        # Check if the model is on the correct device
        for param in self.module.parameters():
            if param.device != self.device_ids[0]:
                raise RuntimeError(
                    f"Module must have its parameters on device "
                    f"'{self.device_ids[0]}' but found one of them on device "
                    f"'{param.device}'")

        inputs = self.scatter(data_list, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs)
        return self.gather(outputs, self.output_device)

    def scatter(self, data_list, device_ids):
        num_devices = min(len(device_ids), len(data_list))

        count = paddle.to_tensor([data.num_nodes for data in data_list], dtype='int64')
        ptr = cumsum(count)
        device_id = num_devices * ptr.cast('float32') / ptr[-1].item()
        device_id = (device_id[:-1] + device_id[1:]) / 2.0
        device_id = device_id.cast('int64')  # round.
        split = cumsum(device_id.bincount())
        split = paddle.unique(split)
        split = paddle.sort(split)
        split = split.tolist()

        return [
            Batch.from_data_list(data_list[split[i]:split[i + 1]],
                                 follow_batch=self.follow_batch,
                                 exclude_keys=self.exclude_keys).to(
                f'gpu:{device_ids[i]}')
            for i in range(len(split) - 1)
        ]

    def replicate(self, module, device_ids):
        return nn.LayerList([module.copy().to(f'cuda:{device_ids[i]}') for i in range(len(device_ids))])

    def parallel_apply(self, replicas, inputs):
        # We use paddle.distributed for parallel_apply if available
        if paddle.distributed.get_world_size() > 1:
            return parallel_apply(replicas, inputs)
        else:
            # Fallback for single GPU
            return [replica(*input) for replica, input in zip(replicas, inputs)]

    def gather(self, outputs, output_device):
        # Gather the outputs from all devices
        return outputs[0]  # Assuming a single output

    def __repr__(self):
        return f"{self.__class__.__name__}(device_ids={self.device_ids}, output_device={self.output_device})"
