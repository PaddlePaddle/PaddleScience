import os.path as osp
from typing import Callable, List, Optional

import paddle

from paddle_geometric.data import Data, InMemoryDataset
from paddle_geometric.io import fs, read_tu_data


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets from TU Dortmund University
    (e.g., :obj:`"IMDB-BINARY"`, :obj:`"REDDIT-BINARY"`, :obj:`"PROTEINS"`).

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The dataset name.
        transform (callable, optional): A function/transform that takes in a
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to disk.
        pre_filter (callable, optional): A function that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset.
        force_reload (bool, optional): Whether to re-process the dataset.
        use_node_attr (bool, optional): If :obj:`True`, the dataset will contain
            additional continuous node attributes.
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will contain
            additional continuous edge attributes.
        cleaned (bool, optional): If :obj:`True`, the dataset will contain only
            non-isomorphic graphs.
    """
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            force_reload: bool = False,
            use_node_attr: bool = False,
            use_edge_attr: bool = False,
            cleaned: bool = False,
    ) -> None:
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        out = fs.paddle_load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of Paddle Geometric. "
                "If this error occurred while loading an existing dataset, remove the "
                "'processed/' directory in the dataset's root folder and try again.")

        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = self.cleaned_url if self.cleaned else self.url
        fs.cp(f'{url}/{self.name}.zip', self.raw_dir, extract=True)
        for filename in fs.ls(osp.join(self.raw_dir, self.name)):
            fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
        fs.rm(osp.join(self.raw_dir, self.name))

    def process(self) -> None:
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.paddle_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
