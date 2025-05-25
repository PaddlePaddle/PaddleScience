import os
from typing import Callable, List, Optional

import pandas as pd
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url


class EmailEUCore(InMemoryDataset):
    r"""An e-mail communication network of a large European research
    institution, taken from the `"Local Higher-order Graph Clustering"
    <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>`_ paper.
    Nodes indicate members of the institution.
    An edge between a pair of members indicates that they exchanged at least
    one email.
    Node labels indicate membership to one of the 42 departments.

    Args:
        root (str): Root directory where the dataset should be saved.
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

    urls = [
        'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['email-Eu-core.txt', 'email-Eu-core-department-labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        for url in self.urls:
            path = download_url(url, self.raw_dir)
            os.system(f"gunzip -f {path}")

    def process(self) -> None:
        edge_index = pd.read_csv(self.raw_paths[0], sep=' ', header=None)
        edge_index = paddle.to_tensor(edge_index.values.T, dtype='int64')

        y = pd.read_csv(self.raw_paths[1], sep=' ', header=None, usecols=[1])
        y = paddle.to_tensor(y.values.flatten(), dtype='int64')

        data = Data(edge_index=edge_index, y=y, num_nodes=y.shape[0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
