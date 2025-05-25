import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import HeteroData, InMemoryDataset
from paddle_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class DBLP(InMemoryDataset):
    r"""A subset of the DBLP computer science bibliography website, as
    collected in the `"MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    DBLP is a heterogeneous graph containing four types of entities - authors
    (4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences
    (20 nodes).
    The authors are divided into four research areas (database, data mining,
    artificial intelligence, information retrieval).
    Each author is described by a bag-of-words representation of their paper
    keywords.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npy',
            'labels.npy', 'node_types.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import scipy.sparse as sp

        data = HeteroData()

        node_types = ['author', 'paper', 'term', 'conference']
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = paddle.to_tensor(x.todense(), dtype='float32')

        x = np.load(osp.join(self.raw_dir, 'features_2.npy'))
        data['term'].x = paddle.to_tensor(x, dtype='float32')

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = paddle.to_tensor(node_type_idx, dtype='int64')
        data['conference'].num_nodes = int((node_type_idx == 3).sum())

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['author'].y = paddle.to_tensor(y, dtype='int64')

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = paddle.to_tensor(idx, dtype='int64')
            mask = paddle.zeros([data['author'].num_nodes], dtype='bool')
            mask[idx] = True
            data['author'][f'{name}_mask'] = mask

        s = {}
        N_a = data['author'].num_nodes
        N_p = data['paper'].num_nodes
        N_t = data['term'].num_nodes
        N_c = data['conference'].num_nodes
        s['author'] = (0, N_a)
        s['paper'] = (N_a, N_a + N_p)
        s['term'] = (N_a + N_p, N_a + N_p + N_t)
        s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = paddle.to_tensor(A_sub.row, dtype='int64')
                col = paddle.to_tensor(A_sub.col, dtype='int64')
                data[src, dst].edge_index = paddle.stack([row, col], axis=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save(data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
