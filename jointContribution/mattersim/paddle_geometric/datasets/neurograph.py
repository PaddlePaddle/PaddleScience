import os
import os.path as osp
from typing import Callable, List, Optional

from paddle_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from paddle_geometric.io import fs


class NeuroGraphDataset(InMemoryDataset):
    r"""The NeuroGraph benchmark datasets from the
    `"NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics"
    <https://arxiv.org/abs/2306.06202>`_ paper.
    :class:`NeuroGraphDataset` holds a collection of five neuroimaging graph
    learning datasets that span multiple categories of demographics, mental
    states, and cognitive traits.
    """

    url = 'https://vanderbilt.box.com/shared/static'
    filenames = {
        'HCPGender': 'r6hlz2arm7yiy6v6981cv2nzq3b0meax.zip',
        'HCPTask': '8wzz4y17wpxg2stip7iybtmymnybwvma.zip',
        'HCPAge': 'lzzks4472czy9f9vc8aikp7pdbknmtfe.zip',
        'HCPWM': 'xtmpa6712fidi94x6kevpsddf9skuoxy.zip',
        'HCPFI': 'g2md9h9snh7jh6eeay02k1kr9m4ido9f.zip',
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
        assert name in self.filenames.keys()
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return 'data.pdparams'

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        url = f'{self.url}/{self.filenames[self.name]}'
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        os.rename(
            osp.join(self.raw_dir, self.name, 'processed', f'{self.name}.pdparams'),
            osp.join(self.raw_dir, 'data.pdparams'))
        fs.rm(osp.join(self.raw_dir, self.name))

    def process(self) -> None:
        data, slices = fs.paddle_load(self.raw_paths[0])

        num_samples = slices['x'].shape[0] - 1
        data_list: List[Data] = []
        for i in range(num_samples):
            x = data.x[slices['x'][i]:slices['x'][i + 1]]
            start = slices['edge_index'][i]
            end = slices['edge_index'][i + 1]
            edge_index = data.edge_index[:, start:end]
            sample = Data(x=x, edge_index=edge_index, y=data.y[i])

            if self.pre_filter is not None and not self.pre_filter(sample):
                continue

            if self.pre_transform is not None:
                sample = self.pre_transform(sample)

            data_list.append(sample)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'NeuroGraphDataset({self.name}, {len(self)})'
