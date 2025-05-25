import pickle
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url


class BA2MotifDataset(InMemoryDataset):
    r"""The synthetic BA-2motifs graph classification dataset for evaluating
    explainability algorithms, as described in the `"Parameterized Explainer
    for Graph Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.
    :class:`~paddle_geometric.datasets.BA2MotifDataset` contains 1000 random
    Barabasi-Albert (BA) graphs.
    Half of the graphs are attached with a
    :class:`~paddle_geometric.datasets.motif_generator.HouseMotif`, and the rest
    are attached with a five-node
    :class:`~paddle_geometric.datasets.motif_generator.CycleMotif`.
    The graphs are assigned to one of the two classes according to the type of
    attached motifs.

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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 1000
          - 25
          - ~51.0
          - 10
          - 2
    """
    url = 'https://github.com/flyingdoog/PGExplainer/raw/master/dataset'
    filename = 'BA-2motif.pkl'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(f'{self.url}/{self.filename}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], 'rb') as f:
            adj, x, y = pickle.load(f)

        adjs = paddle.to_tensor(adj, dtype='int64')
        xs = paddle.to_tensor(x, dtype='float32')
        ys = paddle.to_tensor(y, dtype='int64')

        data_list: List[Data] = []
        for i in range(xs.shape[0]):
            edge_index = paddle.nonzero(adjs[i]).t()
            x = xs[i]
            y = int(paddle.nonzero(ys[i]))

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
