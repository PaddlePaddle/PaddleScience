from typing import List, Optional, Union
import paddle
from paddle import Tensor
from paddle.io import Sampler, Dataset, DataLoader
from paddle_geometric.data import Data, InMemoryDataset


class ImbalancedSampler(Sampler):
    r"""A weighted random sampler that randomly samples elements according to
    class distribution.
    As such, it will either remove samples from the majority class
    (under-sampling) or add more examples from the minority class
    (over-sampling).
    """

    def __init__(
        self,
        dataset: Union[Dataset, Data, List[Data], Tensor],
        input_nodes: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
    ):
        if isinstance(dataset, Data):
            y = dataset.y.flatten()
            assert dataset.num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, Tensor):
            y = dataset.flatten()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.y.flatten()
            assert len(dataset) == y.numel()

        else:
            ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = paddle.concat(ys, axis=0).flatten()
            else:
                y = paddle.to_tensor(ys).flatten()
            assert len(dataset) == y.numel()

        assert y.dtype == paddle.int64  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1. / paddle.bincount(y)
        weight = class_weight[y]

        # Sample the elements with replacement based on the computed weight.
        self.weight = weight
        self.num_samples = num_samples

    def __iter__(self):
        return iter(paddle.randperm(self.num_samples, dtype=paddle.int64).numpy())

    def __len__(self):
        return self.num_samples
