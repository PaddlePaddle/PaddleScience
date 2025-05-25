import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import HeteroData, InMemoryDataset, download_url, extract_zip
from paddle_geometric.io import fs


class OGB_MAG(InMemoryDataset):
    r"""The :obj:`ogbn-mag` dataset from the `"Open Graph Benchmark: Datasets
    for Machine Learning on Graphs" <https://arxiv.org/abs/2005.00687>`_ paper.
    """

    url = 'http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip'
    urls = {
        'metapath2vec': ('https://data.pyg.org/datasets/'
                         'mag_metapath2vec_emb.zip'),
        'transe': ('https://data.pyg.org/datasets/'
                   'mag_transe_emb.zip'),
    }

    def __init__(
        self,
        root: str,
        preprocess: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def num_classes(self) -> int:
        assert isinstance(self._data, HeteroData)
        return int(self._data['paper'].y.max().item()) + 1

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'mag', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'mag', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv.gz'
        ]

        if self.preprocess is not None:
            file_names += [f'mag_{self.preprocess}_emb.pdparams']

        return file_names

    @property
    def processed_file_names(self) -> str:
        if self.preprocess is not None:
            return f'data_{self.preprocess}.pdparams'
        else:
            return 'data.pdparams'

    def download(self) -> None:
        if not all([osp.exists(f) for f in self.raw_paths[:5]]):
            path = download_url(self.url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            for file_name in ['node-feat', 'node-label', 'relations']:
                path = osp.join(self.raw_dir, 'mag', 'raw', file_name)
                shutil.move(path, self.raw_dir)
            path = osp.join(self.raw_dir, 'mag', 'split')
            shutil.move(path, self.raw_dir)
            path = osp.join(self.raw_dir, 'mag', 'raw', 'num-node-dict.csv.gz')
            shutil.move(path, self.raw_dir)
            fs.rm(osp.join(self.raw_dir, 'mag'))
            os.remove(osp.join(self.raw_dir, 'mag.zip'))
        if self.preprocess is not None:
            path = download_url(self.urls[self.preprocess], self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Load paper features
        path = osp.join(self.raw_dir, 'node-feat', 'paper', 'node-feat.csv.gz')
        x_paper = pd.read_csv(path, compression='gzip', header=None, dtype=np.float32).values
        data['paper'].x = paddle.to_tensor(x_paper)

        # Load paper years
        path = osp.join(self.raw_dir, 'node-feat', 'paper', 'node_year.csv.gz')
        year_paper = pd.read_csv(path, compression='gzip', header=None, dtype=np.int64).values
        data['paper'].year = paddle.to_tensor(year_paper).reshape([-1])

        # Load paper labels
        path = osp.join(self.raw_dir, 'node-label', 'paper', 'node-label.csv.gz')
        y_paper = pd.read_csv(path, compression='gzip', header=None, dtype=np.int64).values.flatten()
        data['paper'].y = paddle.to_tensor(y_paper)

        # Load structural features if specified
        if self.preprocess is None:
            path = osp.join(self.raw_dir, 'num-node-dict.csv.gz')
            num_nodes_df = pd.read_csv(path, compression='gzip')
            for node_type in ['author', 'institution', 'field_of_study']:
                data[node_type].num_nodes = num_nodes_df[node_type].tolist()[0]
        else:
            emb_dict = fs.paddle_load(self.raw_paths[-1])
            for key, value in emb_dict.items():
                if key != 'paper':
                    data[key].x = paddle.to_tensor(value)

        # Load edges
        for edge_type in [('author', 'affiliated_with', 'institution'),
                          ('author', 'writes', 'paper'),
                          ('paper', 'cites', 'paper'),
                          ('paper', 'has_topic', 'field_of_study')]:
            f = '___'.join(edge_type)
            path = osp.join(self.raw_dir, 'relations', f, 'edge.csv.gz')
            edge_index = pd.read_csv(path, compression='gzip', header=None, dtype=np.int64).values
            edge_index = paddle.to_tensor(edge_index).transpose([1, 0])
            data[edge_type].edge_index = edge_index

        # Load train/val/test splits
        for f, v in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            path = osp.join(self.raw_dir, 'split', 'time', 'paper', f'{f}.csv.gz')
            idx = pd.read_csv(path, compression='gzip', header=None, dtype=np.int64).values.flatten()
            idx = paddle.to_tensor(idx)
            mask = paddle.zeros([data['paper'].num_nodes], dtype=paddle.bool)
            mask[idx] = True
            data['paper'][f'{v}_mask'] = mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return 'ogbn-mag()'
