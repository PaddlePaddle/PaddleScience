import json
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import paddle
from paddle_geometric.data import Data, InMemoryDataset, download_google_url


class Yelp(InMemoryDataset):
    r"""The Yelp dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing customer reviewers and their friendship.
    """

    adj_full_id = '1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA'
    feats_id = '1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM'
    class_map_id = '1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU'
    role_id = '1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_'

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

        train_mask = paddle.zeros([x.shape[0]], dtype='bool')
        train_mask[paddle.to_tensor(role['tr'])] = True

        val_mask = paddle.zeros([x.shape[0]], dtype='bool')
        val_mask[paddle.to_tensor(role['va'])] = True

        test_mask = paddle.zeros([x.shape[0]], dtype='bool')
        test_mask[paddle.to_tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
