import os
import os.path as osp
from typing import Callable, Dict, List, Optional

import paddle

from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)
from paddle_geometric.io import fs


class MalNetTiny(InMemoryDataset):
    data_url = ('http://malnet.cc.gatech.edu/'
                'graph-data/malnet-graphs-tiny.tar.gz')
    split_url = 'http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip'
    splits = ['train', 'val', 'test']

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if split not in {'train', 'val', 'trainval', 'test', None}:
            raise ValueError(f'Split "{split}" found, but expected either '
                             f'"train", "val", "trainval", "test" or None')
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

        if split is not None:
            split_slices = fs.load(self.processed_paths[1])
            if split == 'train':
                self._indices = range(split_slices[0], split_slices[1])
            elif split == 'val':
                self._indices = range(split_slices[1], split_slices[2])
            elif split == 'trainval':
                self._indices = range(split_slices[0], split_slices[2])
            elif split == 'test':
                self._indices = range(split_slices[2], split_slices[3])

    @property
    def raw_file_names(self) -> List[str]:
        return ['malnet-graphs-tiny', osp.join('split_info_tiny', 'type')]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pdparams', 'split_slices.pdparams']

    def download(self) -> None:
        path = download_url(self.data_url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

        path = download_url(self.split_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        y_map: Dict[str, int] = {}
        data_list = []
        split_slices = [0]

        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_paths[1], f'{split}.txt')) as f:
                filenames = f.read().split('\n')[:-1]
                split_slices.append(split_slices[-1] + len(filenames))

            for filename in filenames:
                path = osp.join(self.raw_paths[0], f'{filename}.edgelist')
                malware_type = filename.split('/')[0]
                y = y_map.setdefault(malware_type, len(y_map))

                with open(path) as f:
                    edges = f.read().split('\n')[5:-1]

                edge_indices = [[int(s) for s in e.split()] for e in edges]
                edge_index = paddle.to_tensor(edge_indices, dtype='int64').transpose([1, 0])
                num_nodes = int(edge_index.max()) + 1
                data = Data(edge_index=edge_index, y=paddle.to_tensor([y], dtype='int64'),
                            num_nodes=num_nodes)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        paddle.save(split_slices, self.processed_paths[1])
