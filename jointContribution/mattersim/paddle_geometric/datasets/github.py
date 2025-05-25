from typing import Callable, Optional

import numpy as np
import paddle
from paddle import Tensor

from paddle_geometric.data import Data, InMemoryDataset, download_url


class GitHub(InMemoryDataset):
    r"""The GitHub Web and ML Developers dataset introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent developers on :obj:`github:`GitHub` and edges are mutual
    follower relationships.
    It contains 37,300 nodes, 578,006 edges, 128 node features and 2 classes.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 37,700
          - 578,006
          - 0
          - 2
    """
    url = 'https://graphmining.ai/datasets/ptg/github.npz'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'github.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = paddle.to_tensor(data['features'], dtype='float32')
        y = paddle.to_tensor(data['target'], dtype='int64')
        edge_index = paddle.to_tensor(data['edges'], dtype='int64')
        edge_index = edge_index.transpose([1, 0])

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
