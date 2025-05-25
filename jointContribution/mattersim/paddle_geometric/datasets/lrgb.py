import os
import os.path as osp
import pickle
from typing import Callable, Dict, List, Optional

import paddle
from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from paddle_geometric.io import fs


class LRGBDataset(InMemoryDataset):
    names = [
        'pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func',
        'peptides-struct'
    ]

    urls = {
        'pascalvoc-sp':
        'https://www.dropbox.com/s/8x722ai272wqwl4/pascalvocsp.zip?dl=1',
        'coco-sp':
        'https://www.dropbox.com/s/r6ihg1f4pmyjjy0/cocosp.zip?dl=1',
        'pcqm-contact':
        'https://www.dropbox.com/s/qdag867u6h6i60y/pcqmcontact.zip?dl=1',
        'peptides-func':
        'https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1',
        'peptides-struct':
        'https://www.dropbox.com/s/zgv4z8fcpmknhs8/peptidesstruct.zip?dl=1'
    }

    dwnld_file_name = {
        'pascalvoc-sp': 'voc_superpixels_edge_wt_region_boundary',
        'coco-sp': 'coco_superpixels_edge_wt_region_boundary',
        'pcqm-contact': 'pcqmcontact',
        'peptides-func': 'peptidesfunc',
        'peptides-struct': 'peptidesstruct'
    }

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.names
        assert split in ['train', 'val', 'test']

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = osp.join(self.processed_dir, f'{split}.pdparams')
        self.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name.split('-')[1] == 'sp':
            return ['train.pickle', 'val.pickle', 'test.pickle']
        else:
            return ['train.pdparams', 'val.pdparams', 'test.pdparams']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pdparams', 'val.pdparams', 'test.pdparams']

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, self.dwnld_file_name[self.name]),
                  self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        if self.name == 'pcqm-contact':
            self.process_pcqm_contact()
        else:
            if self.name == 'coco-sp':
                label_map = self.label_remap_coco()

            for split in ['train', 'val', 'test']:
                if self.name.split('-')[1] == 'sp':
                    with open(osp.join(self.raw_dir, f'{split}.pickle'),
                              'rb') as f:
                        graphs = pickle.load(f)
                elif self.name.split('-')[0] == 'peptides':
                    graphs = fs.load(osp.join(self.raw_dir, f'{split}.pdparams'))

                data_list = []
                for graph in graphs:
                    if self.name.split('-')[1] == 'sp':
                        x = paddle.to_tensor(graph[0], dtype='float32')
                        edge_attr = paddle.to_tensor(graph[1], dtype='float32')
                        edge_index = paddle.to_tensor(graph[2], dtype='int64')
                        y = paddle.to_tensor(graph[3], dtype='int64')
                    elif self.name.split('-')[0] == 'peptides':
                        x = graph[0]
                        edge_attr = graph[1]
                        edge_index = graph[2]
                        y = graph[3]

                    if self.name == 'coco-sp':
                        for i, label in enumerate(y):
                            y[i] = label_map[int(label.numpy())]

                    data = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_attr, y=y)

                    if self.pre_filter is not None and not self.pre_filter(
                            data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                path = osp.join(self.processed_dir, f'{split}.pdparams')
                self.save(data_list, path)

    def label_remap_coco(self) -> Dict[int, int]:
        original_label_idx = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

        label_map = {}
        for i, key in enumerate(original_label_idx):
            label_map[key] = i

        return label_map

    def process_pcqm_contact(self) -> None:
        for split in ['train', 'val', 'test']:
            graphs = fs.load(osp.join(self.raw_dir, f'{split}.pdparams'))

            data_list = []
            for graph in graphs:
                x = graph[0]
                edge_attr = graph[1]
                edge_index = graph[2]
                edge_label_index = graph[3]
                edge_label = graph[4]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, osp.join(self.processed_dir, f'{split}.pdparams'))
