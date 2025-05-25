import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import paddle
from tqdm import tqdm
from paddle_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from paddle_geometric.io import fs


class ZINC(InMemoryDataset):
    r"""The ZINC dataset from the `ZINC database
    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
    `"Automatic Chemical Design Using a Data-Driven Continuous Representation
    of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about
    250,000 molecular graphs with up to 38 heavy atoms.
    """

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self) -> None:
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = list(range(len(mols)))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index')) as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = paddle.to_tensor(mol['atom_type'], dtype='int64').reshape([-1, 1])
                y = paddle.to_tensor(mol['logP_SA_cycle_normalized'], dtype='float32')

                adj = mol['bond_type']
                edge_index = paddle.nonzero(adj).t()
                edge_attr = paddle.to_tensor(adj[edge_index[0], edge_index[1]], dtype='int64')

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            self.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))
