import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Union

import fsspec
import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset
from paddle_geometric.io import fs
from paddle_geometric.utils import coalesce


class EgoData(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        # Adjusts the 'circle' attribute based on the number of nodes in the graph
        if key == 'circle':
            return self.num_nodes
        elif key == 'circle_batch':
            return int(value.max()) + 1 if value.numel() > 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def read_ego(files: List[str], name: str) -> List[EgoData]:
    # Reads ego networks from files
    import pandas as pd
    import tqdm

    files = sorted(files)

    all_featnames = []
    files = [
        x for x in files if x.split('.')[-1] in
        ['circles', 'edges', 'egofeat', 'feat', 'featnames']
    ]
    for i in range(4, len(files), 5):
        featnames_file = files[i]
        with fsspec.open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            all_featnames += featnames
    all_featnames = sorted(list(set(all_featnames)))
    all_featnames_dict = {key: i for i, key in enumerate(all_featnames)}

    data_list = []
    for i in tqdm.tqdm(range(0, len(files), 5)):
        circles_file = files[i]
        edges_file = files[i + 1]
        egofeat_file = files[i + 2]
        feat_file = files[i + 3]
        featnames_file = files[i + 4]

        x = None
        if name != 'gplus':  # Skips reading features for the gplus dataset
            x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,
                                dtype=np.float32)
            x_ego = paddle.to_tensor(x_ego.values)

            x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
            x = paddle.to_tensor(x.values)[:, 1:]

            x_all = paddle.concat([x, x_ego], axis=0)

            # Reorders `x` according to `featnames` ordering
            x_all = paddle.zeros([x.shape[0], len(all_featnames)])
            with fsspec.open(featnames_file, 'r') as f:
                featnames = f.read().split('\n')[:-1]
                featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            indices = [all_featnames_dict[featname] for featname in featnames]
            x_all[:, paddle.to_tensor(indices)] = x
            x = x_all

            if x.shape[1] > 100_000:
                x = x.to_sparse_csr()

        idx = pd.read_csv(feat_file, sep=' ', header=None, dtype=str,
                          usecols=[0]).squeeze()

        idx_assoc: Dict[str, int] = {}
        for i, j in enumerate(idx):
            idx_assoc[j] = i

        circles: List[int] = []
        circles_batch: List[int] = []
        with fsspec.open(circles_file, 'r') as f:
            for i, line in enumerate(f.read().split('\n')[:-1]):
                circle_indices = [idx_assoc[c] for c in line.split()[1:]]
                circles += circle_indices
                circles_batch += [i] * len(circle_indices)
        circle = paddle.to_tensor(circles)
        circle_batch = paddle.to_tensor(circles_batch)

        try:
            row = pd.read_csv(edges_file, sep=' ', header=None, dtype=str,
                              usecols=[0]).squeeze()
            col = pd.read_csv(edges_file, sep=' ', header=None, dtype=str,
                              usecols=[1]).squeeze()
        except Exception:
            continue

        row = paddle.to_tensor([idx_assoc[i] for i in row])
        col = paddle.to_tensor([idx_assoc[i] for i in col])

        N = max(int(row.max()), int(col.max())) + 2
        N = x.shape[0] if x is not None else N

        row_ego = paddle.full([N - 1], N - 1, dtype='int64')
        col_ego = paddle.arange(N - 1, dtype='int64')

        # Connects ego node to every other node
        row = paddle.concat([row, row_ego, col_ego], axis=0)
        col = paddle.concat([col, col_ego, row_ego], axis=0)
        edge_index = paddle.stack([row, col], axis=0)
        edge_index = coalesce(edge_index, num_nodes=N)

        data = EgoData(x=x, edge_index=edge_index, circle=circle,
                       circle_batch=circle_batch)

        data_list.append(data)

    return data_list


def read_soc(files: List[str], name: str) -> List[Data]:
    # Reads social network datasets
    import pandas as pd

    skiprows = 4
    if name == 'pokec':
        skiprows = 0

    edge_index = pd.read_csv(files[0], sep='\t', header=None,
                             skiprows=skiprows, dtype=np.int64)
    edge_index = paddle.to_tensor(edge_index.values).t()
    num_nodes = edge_index.max().item() + 1
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


def read_wiki(files: List[str], name: str) -> List[Data]:
    # Reads Wikipedia network datasets
    import pandas as pd

    edge_index = pd.read_csv(files[0], sep='\t', header=None, skiprows=4,
                             dtype=np.int64)
    edge_index = paddle.to_tensor(edge_index.values).t()

    idx = paddle.unique(edge_index.flatten())
    idx_assoc = paddle.full([edge_index.max() + 1], -1, dtype='int64')
    idx_assoc[idx] = paddle.arange(idx.shape[0], dtype='int64')

    edge_index = idx_assoc[edge_index]
    num_nodes = edge_index.max().item() + 1
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


class SNAPDataset(InMemoryDataset):
    # A variety of graph datasets collected from SNAP at Stanford University

    url = 'https://snap.stanford.edu/data'

    available_datasets = {
        'ego-facebook': ['facebook.tar.gz'],
        'ego-gplus': ['gplus.tar.gz'],
        'ego-twitter': ['twitter.tar.gz'],
        'soc-ca-astroph': ['ca-AstroPh.txt.gz'],
        'soc-ca-grqc': ['ca-GrQc.txt.gz'],
        'soc-epinions1': ['soc-Epinions1.txt.gz'],
        'soc-livejournal1': ['soc-LiveJournal1.txt.gz'],
        'soc-pokec': ['soc-pokec-relationships.txt.gz'],
        'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'],
        'soc-slashdot0922': ['soc-Slashdot0902.txt.gz'],
        'wiki-vote': ['wiki-Vote.txt.gz'],
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        # Initialize dataset properties
        self.name = name.lower()
        assert self.name in self.available_datasets.keys()
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        # Directory for raw data
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        # Directory for processed data
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        # Processed data file name
        return 'data.pt'

    def _download(self) -> None:
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def download(self) -> None:
        # Download dataset files from SNAP
        for name in self.available_datasets[self.name]:
            fs.cp(f'{self.url}/{name}', self.raw_dir, extract=True)

    def process(self) -> None:
        # Process raw files into graph data
        raw_dir = self.raw_dir
        filenames = fs.ls(self.raw_dir)
        if len(filenames) == 1 and fs.isdir(filenames[0]):
            raw_dir = filenames[0]

        raw_files = fs.ls(raw_dir)

        data_list: Union[List[Data], List[EgoData]]
        if self.name[:4] == 'ego-':
            data_list = read_ego(raw_files, self.name[4:])
        elif self.name[:4] == 'soc-':
            data_list = read_soc(raw_files, self.name[:4])
        elif self.name[:5] == 'wiki-':
            data_list = read_wiki(raw_files, self.name[5:])
        else:
            raise NotImplementedError

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        # Display dataset name and length
        return f'SNAP-{self.name}({len(self)})'
