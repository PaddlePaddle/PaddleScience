import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from paddle_geometric.utils import coalesce


class Reddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

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
    """

    url = 'https://data.dgl.ai/dataset/reddit.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import scipy.sparse as sp

        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = paddle.to_tensor(data['feature'], dtype=paddle.float32)
        y = paddle.to_tensor(data['label'], dtype=paddle.int64)
        split = paddle.to_tensor(data['node_types'], dtype=paddle.int64)

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = paddle.to_tensor(adj.row, dtype=paddle.int64)
        col = paddle.to_tensor(adj.col, dtype=paddle.int64)
        edge_index = paddle.stack([row, col], axis=0)
        edge_index = coalesce(edge_index, num_nodes=x.shape[0])

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
