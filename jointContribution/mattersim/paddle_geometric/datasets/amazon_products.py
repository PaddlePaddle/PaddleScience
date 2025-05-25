import json
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_google_url


class AmazonProducts(InMemoryDataset):
    r"""The Amazon dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing products and its categories.

    Args:
        root: Root directory where the dataset should be saved.
        transform: A function/transform that takes in an
            :class:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in a
            :class:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
        force_reload: Whether to re-process the dataset.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 1,569,960
          - 264,339,468
          - 200
          - 107
    """
    adj_full_id = '17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa'
    feats_id = '10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj'
    class_map_id = '1LIl4kimLfftj4-7NmValuWyCQE8AaE7P'
    role_id = '1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K'

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
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, 'adj_full.npz')
        download_google_url(self.feats_id, self.raw_dir, 'feats.npy')
        download_google_url(self.class_map_id, self.raw_dir, 'class_map.json')
        download_google_url(self.role_id, self.raw_dir, 'role.json')

    def process(self) -> None:
        import scipy.sparse as sp

        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = paddle.to_tensor(adj.row, dtype='int64')
        col = paddle.to_tensor(adj.col, dtype='int64')
        edge_index = paddle.stack([row, col], axis=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = paddle.to_tensor(x, dtype='float32')

        ys = [-1] * x.shape[0]
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = paddle.to_tensor(ys, dtype='int64')

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = paddle.zeros((x.shape[0],), dtype='bool')
        train_mask[paddle.to_tensor(role['tr'], dtype='int64')] = True

        val_mask = paddle.zeros((x.shape[0],), dtype='bool')
        val_mask[paddle.to_tensor(role['va'], dtype='int64')] = True

        test_mask = paddle.zeros((x.shape[0],), dtype='bool')
        test_mask[paddle.to_tensor(role['te'], dtype='int64')] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
