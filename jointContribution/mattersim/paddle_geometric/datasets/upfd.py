import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_google_url,
    extract_zip,
)
from paddle_geometric.io import read_txt_array
from paddle_geometric.utils import coalesce, cumsum


class UPFD(InMemoryDataset):
    r"""The tree-structured fake news propagation graph classification dataset
    from the `"User Preference-aware Fake News Detection"
    <https://arxiv.org/abs/2104.12259>`_ paper.
    It includes two sets of tree-structured fake & real news propagation graphs
    extracted from Twitter.
    For a single graph, the root node represents the source news, and leaf
    nodes represent Twitter users who retweeted the same root news.
    A user node has an edge to the news node if and only if the user retweeted
    the root news directly.
    Two user nodes have an edge if and only if one user retweeted the root news
    from the other user.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the graph set (:obj:`"politifact"`, :obj:`"gossipcop"`).
        feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`,
            :obj:`"bert"`, :obj:`"content"`).
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
        pre_filter (callable, optional): A function that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset.
        force_reload (bool, optional): Whether to re-process the dataset.
    """
    file_ids = {
        'politifact': '1KOmSrlGcC50PjkvRVbyb_WoWHVql06J-',
        'gossipcop': '1VskhAQ92PrT4sWEKQ2v2-AJhEcpp4A81',
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in ['politifact', 'gossipcop']
        assert split in ['train', 'val', 'test']

        self.root = root
        self.name = name
        self.feature = feature

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed', self.feature)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
            'val_idx.npy', 'test_idx.npy', f'new_{self.feature}_feature.npz'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        file_id = self.file_ids[self.name]
        path = download_google_url(file_id, self.raw_dir, 'data.zip')
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import scipy.sparse as sp

        x = sp.load_npz(osp.join(self.raw_dir, f'new_{self.feature}_feature.npz'))
        x = paddle.to_tensor(x.todense(), dtype=paddle.float32)

        edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',', dtype=paddle.int64).t()
        edge_index = coalesce(edge_index, num_nodes=x.shape[0])

        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = paddle.to_tensor(y, dtype=paddle.int64)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = paddle.to_tensor(batch, dtype=paddle.int64)

        node_slice = cumsum(batch.bincount())
        edge_slice = cumsum(batch[edge_index[0]].bincount())
        graph_slice = paddle.arange(y.shape[0] + 1)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice
        }

        edge_index -= node_slice[batch[edge_index[0]]].reshape([1, -1])
        self.data = Data(x=x, edge_index=edge_index, y=y)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, path)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)}, name={self.name}, feature={self.feature})'
