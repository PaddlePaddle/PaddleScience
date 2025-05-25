import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import paddle

from paddle_geometric.data import InMemoryDataset, download_url, extract_zip
from paddle_geometric.io import fs, read_off, read_txt_array


class SHREC2016(InMemoryDataset):
    train_url = ('http://www.dais.unive.it/~shrec2016/data/'
                 'shrec2016_PartialDeformableShapes.zip')
    test_url = ('http://www.dais.unive.it/~shrec2016/data/'
                'shrec2016_PartialDeformableShapes_TestSet.zip')

    categories = [
        'cat', 'centaur', 'david', 'dog', 'horse', 'michael', 'victoria',
        'wolf'
    ]
    partialities = ['holes', 'cuts']

    def __init__(
        self,
        root: str,
        partiality: str,
        category: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert partiality.lower() in self.partialities
        self.part = partiality.lower()
        assert category.lower() in self.categories
        self.cat = category.lower()
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.__ref__ = fs.paddle_load(self.processed_paths[0])
        path = self.processed_paths[1] if train else self.processed_paths[2]
        self.load(path)

    @property
    def ref(self) -> str:
        ref = self.__ref__
        if self.transform is not None:
            ref = self.transform(ref)
        return ref

    @property
    def raw_file_names(self) -> List[str]:
        return ['training', 'test']

    @property
    def processed_file_names(self) -> List[str]:
        name = f'{self.part}_{self.cat}.pt'
        return [f'{i}_{name}' for i in ['ref', 'training', 'test']]

    def download(self) -> None:
        path = download_url(self.train_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir, 'shrec2016_PartialDeformableShapes')
        os.rename(path, osp.join(self.raw_dir, 'training'))

        path = download_url(self.test_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir,
                        'shrec2016_PartialDeformableShapes_TestSet')
        os.rename(path, osp.join(self.raw_dir, 'test'))

    def process(self) -> None:
        ref_data = read_off(
            osp.join(self.raw_paths[0], 'null', f'{self.cat}.off'))

        train_list = []
        name = f'{self.part}_{self.cat}_*.off'
        paths = glob.glob(osp.join(self.raw_paths[0], self.part, name))
        paths = [path[:-4] for path in paths]
        paths = sorted(paths, key=lambda e: (len(e), e))

        for path in paths:
            data = read_off(f'{path}.off')
            y = read_txt_array(f'{path}.baryc_gt')
            data.y = y[:, 0].astype('int64') - 1
            data.y_baryc = y[:, 1:]
            train_list.append(data)

        test_list = []
        name = f'{self.part}_{self.cat}_*.off'
        paths = glob.glob(osp.join(self.raw_paths[1], self.part, name))
        paths = [path[:-4] for path in paths]
        paths = sorted(paths, key=lambda e: (len(e), e))

        for path in paths:
            test_list.append(read_off(f'{path}.off'))

        if self.pre_filter is not None:
            train_list = [d for d in train_list if self.pre_filter(d)]
            test_list = [d for d in test_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            ref_data = self.pre_transform(ref_data)
            train_list = [self.pre_transform(d) for d in train_list]
            test_list = [self.pre_transform(d) for d in test_list]

        paddle.save(ref_data, self.processed_paths[0])
        self.save(train_list, self.processed_paths[1])
        self.save(test_list, self.processed_paths[2])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'partiality={self.part}, category={self.cat})')
