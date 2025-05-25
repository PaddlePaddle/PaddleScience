import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url
from paddle_geometric.io import fs
from paddle_geometric.utils import one_hot


class LINKXDataset(InMemoryDataset):
    r"""A variety of non-homophilous graph datasets from the `"Large Scale
    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
    Methods" <https://arxiv.org/abs/2110.14446>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"penn94"`, :obj:`"reed98"`,
            :obj:`"amherst41"`, :obj:`"cornell5"`, :obj:`"johnshopkins55"`).
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
    github_url = ('https://github.com/CUAI/Non-Homophily-Large-Scale/'
                  'raw/master/data')
    gdrive_url = 'https://drive.usercontent.google.com/download?confirm=t'

    facebook_datasets = [
        'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55'
    ]

    datasets = {
        'penn94': {
            'data.mat': f'{github_url}/facebook100/Penn94.mat'
        },
        'reed98': {
            'data.mat': f'{github_url}/facebook100/Reed98.mat'
        },
        'amherst41': {
            'data.mat': f'{github_url}/facebook100/Amherst41.mat',
        },
        'cornell5': {
            'data.mat': f'{github_url}/facebook100/Cornell5.mat'
        },
        'johnshopkins55': {
            'data.mat': f'{github_url}/facebook100/Johns%20Hopkins55.mat'
        },
        'genius': {
            'data.mat': f'{github_url}/genius.mat'
        },
        'wiki': {
            'wiki_views2M.pt':
            f'{gdrive_url}&id=1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',
            'wiki_edges2M.pt':
            f'{gdrive_url}&id=14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',
            'wiki_features2M.pt':
            f'{gdrive_url}&id=1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
        }
    }

    splits = {
        'penn94': f'{github_url}/splits/fb100-Penn94-splits.npy',
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = list(self.datasets[self.name].keys())
        if self.name in self.splits:
            names += [self.splits[self.name].split('/')[-1]]
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for filename, path in self.datasets[self.name].items():
            download_url(path, self.raw_dir, filename=filename)
        if self.name in self.splits:
            download_url(self.splits[self.name], self.raw_dir)

    def _process_wiki(self) -> Data:
        paths = {x.split('/')[-1]: x for x in self.raw_paths}
        x = fs.load(paths['wiki_features2M.pt'])
        edge_index = fs.load(paths['wiki_edges2M.pt']).transpose([1, 0])
        y = fs.load(paths['wiki_views2M.pt'])

        return Data(x=x, edge_index=edge_index, y=y)

    def _process_facebook(self) -> Data:
        from scipy.io import loadmat

        mat = loadmat(self.raw_paths[0])

        A = mat['A'].tocsr().tocoo()
        row = paddle.to_tensor(A.row, dtype='int64')
        col = paddle.to_tensor(A.col, dtype='int64')
        edge_index = paddle.stack([row, col], axis=0)

        metadata = paddle.to_tensor(mat['local_info'].astype('int64'))

        xs = []
        y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        x = paddle.concat([metadata[:, :1], metadata[:, 2:]], axis=-1)
        for i in range(x.shape[1]):
            _, out = paddle.unique(x[:, i], return_inverse=True)
            xs.append(one_hot(out))
        x = paddle.concat(xs, axis=-1)

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.name in self.splits:
            splits = np.load(self.raw_paths[1], allow_pickle=True)
            assert data.num_nodes is not None
            sizes = (data.num_nodes, len(splits))
            data.train_mask = paddle.zeros(sizes, dtype='bool')
            data.val_mask = paddle.zeros(sizes, dtype='bool')
            data.test_mask = paddle.zeros(sizes, dtype='bool')

            for i, split in enumerate(splits):
                data.train_mask[:, i][paddle.to_tensor(split['train'])] = True
                data.val_mask[:, i][paddle.to_tensor(split['valid'])] = True
                data.test_mask[:, i][paddle.to_tensor(split['test'])] = True

        return data

    def _process_genius(self) -> Data:
        from scipy.io import loadmat

        mat = loadmat(self.raw_paths[0])
        edge_index = paddle.to_tensor(mat['edge_index'], dtype='int64')
        x = paddle.to_tensor(mat['node_feat'], dtype='float32')
        y = paddle.to_tensor(mat['label'], dtype='int64').squeeze()

        return Data(x=x, edge_index=edge_index, y=y)

    def process(self) -> None:
        if self.name in self.facebook_datasets:
            data = self._process_facebook()
        elif self.name == 'genius':
            data = self._process_genius()
        elif self.name == 'wiki':
            data = self._process_wiki()
        else:
            raise NotImplementedError(
                f"Chosen dataset '{self.name}' is not implemented")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}({len(self)})'
