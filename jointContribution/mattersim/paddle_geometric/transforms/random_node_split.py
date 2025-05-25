import random
from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.data.storage import NodeStorage
from paddle_geometric.transforms import BaseTransform


@functional_transform('random_node_split')
class RandomNodeSplit(BaseTransform):
    r"""Performs a node-level random split by adding :obj:`train_mask`,
    :obj:`val_mask` and :obj:`test_mask` attributes to the
    :class:`~paddle_geometric.data.Data` or
    :class:`~paddle_geometric.data.HeteroData` object
    (functional name: :obj:`random_node_split`).

    Args:
        split (str, optional): The type of dataset split (:obj:`"train_rest"`,
            :obj:`"test_rest"`, :obj:`"random"`).
            If set to :obj:`"train_rest"`, all nodes except those in the
            validation and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"test_rest"`, all nodes except those in the
            training and validation sets will be used for test (as in the
            `"Pitfalls of Graph Neural Network Evaluation"
            <https://arxiv.org/abs/1811.05868>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test` (as in the `"Semi-supervised
            Classification with Graph Convolutional Networks"
            <https://arxiv.org/abs/1609.02907>`_ paper).
            (default: :obj:`"train_rest"`)
        num_splits (int, optional): The number of splits to add. If bigger
            than :obj:`1`, the shape of masks will be
            :obj:`[num_nodes, num_splits]`, and :obj:`[num_nodes]` otherwise.
            (default: :obj:`1`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"test_rest"` and :obj:`"random"` split.
            (default: :obj:`20`)
        num_val (int or float, optional): The number of validation samples.
            If float, it represents the ratio of samples to include in the
            validation set. (default: :obj:`500`)
        num_test (int or float, optional): The number of test samples in case
            of :obj:`"train_rest"` and :obj:`"random"` split. If float, it
            represents the ratio of samples to include in the test set.
            (default: :obj:`1000`)
        key (str, optional): The name of the attribute holding ground-truth
            labels. By default, will only add node-level splits for node-level
            storages in which :obj:`key` is present. (default: :obj:`"y"`).
    """
    def __init__(
        self,
        split: str = "train_rest",
        num_splits: int = 1,
        num_train_per_class: int = 20,
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        key: Optional[str] = "y",
    ) -> None:
        assert split in ['train_rest', 'test_rest', 'random']
        self.split = split
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = key

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = paddle.stack(train_masks, axis=-1).squeeze(-1)
            store.val_mask = paddle.stack(val_masks, axis=-1).squeeze(-1)
            store.test_mask = paddle.stack(test_masks, axis=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = store.num_nodes
        assert num_nodes is not None

        train_mask = paddle.zeros([num_nodes], dtype=paddle.bool)
        val_mask = paddle.zeros([num_nodes], dtype=paddle.bool)
        test_mask = paddle.zeros([num_nodes], dtype=paddle.bool)

        num_val = int(num_nodes * self.num_val) if isinstance(self.num_val, float) else self.num_val
        num_test = int(num_nodes * self.num_test) if isinstance(self.num_test, float) else self.num_test

        if self.split == 'train_rest':
            perm = paddle.randperm(num_nodes)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True
        else:
            assert self.key is not None
            y = getattr(store, self.key)
            num_classes = int(paddle.max(y).item()) + 1
            for c in range(num_classes):
                idx = paddle.nonzero(y == c).flatten()
                idx = idx[paddle.randperm(len(idx))]
                train_mask[idx[:self.num_train_per_class]] = True

            remaining = paddle.nonzero(~train_mask).flatten()
            remaining = remaining[paddle.randperm(len(remaining))]

            val_mask[remaining[:num_val]] = True

            if self.split == 'test_rest':
                test_mask[remaining[num_val:]] = True
            elif self.split == 'random':
                test_mask[remaining[num_val:num_val + num_test]] = True

        return train_mask, val_mask, test_mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(split={self.split})'
