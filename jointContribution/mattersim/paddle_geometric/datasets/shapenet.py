import json
import os
import os.path as osp
from typing import Callable, List, Optional, Union

import paddle

from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from paddle_geometric.io import fs, read_txt_array


class ShapeNet(InMemoryDataset):
    url = ('https://shapenet.cs.stanford.edu/media/'
           'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }

    def __init__(
        self,
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'trainval':
            path = self.processed_paths[3]
        else:
            raise ValueError(f'Split {split} found, but expected either '
                             'train, val, trainval or test')

        self.load(path)

        assert isinstance(self._data, Data)
        self._data.x = self._data.x if include_normals else None

        self.y_mask = paddle.zeros((len(self.seg_classes.keys()), 50),
                                   dtype='bool')
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

    @property
    def num_classes(self) -> int:
        return self.y_mask.shape[-1]

    @property
    def raw_file_names(self) -> List[str]:
        return list(self.category_ids.values()) + ['train_test_split']

    @property
    def processed_file_names(self) -> List[str]:
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            osp.join(f'{cats}_{split}.pt')
            for split in ['train', 'val', 'test', 'trainval']
        ]

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        name = self.url.split('/')[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process_filenames(self, filenames: List[str]) -> List[Data]:
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        for name in filenames:
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue

            tensor = read_txt_array(osp.join(self.raw_dir, name))
            pos = tensor[:, :3]
            x = tensor[:, 3:6]
            y = tensor[:, -1].astype('int64')
            data = Data(pos=pos, x=x, y=y, category=cat_idx[cat])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self) -> None:
        trainval = []
        for i, split in enumerate(['train', 'val', 'test']):
            path = osp.join(self.raw_dir, 'train_test_split',
                            f'shuffled_{split}_file_list.json')
            with open(path) as f:
                filenames = [
                    osp.sep.join(name.split('/')[1:]) + '.txt'
                    for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == 'train' or split == 'val':
                trainval += data_list
            self.save(data_list, self.processed_paths[i])
        self.save(trainval, self.processed_paths[3])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'categories={self.categories})')
