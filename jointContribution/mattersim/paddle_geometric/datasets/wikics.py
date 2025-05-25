import json
import warnings
from itertools import chain
from typing import Callable, List, Optional

import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url
from paddle_geometric.utils import to_undirected


class WikiCS(InMemoryDataset):
    r"""The semi-supervised Wikipedia-based dataset from the
    `"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks"
    <https://arxiv.org/abs/2007.02901>`_ paper, containing 11,701 nodes,
    216,123 edges, 10 classes and 20 different training splits.

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
        is_undirected (bool, optional): Whether the graph is undirected.
            (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        is_undirected: Optional[bool] = None,
        force_reload: bool = False,
    ) -> None:
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behavior.")
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.json']

    @property
    def processed_file_names(self) -> str:
        return 'data_undirected.pt' if self.is_undirected else 'data.pt'

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            data = json.load(f)

        x = paddle.to_tensor(data['features'], dtype='float32')
        y = paddle.to_tensor(data['labels'], dtype='int64')

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = paddle.to_tensor(edges, dtype='int64').t()
        if self.is_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.shape[0])

        train_mask = paddle.to_tensor(data['train_masks'], dtype='bool').t()

        val_mask = paddle.to_tensor(data['val_masks'], dtype='bool').t()

        test_mask = paddle.to_tensor(data['test_mask'], dtype='bool')

        stopping_mask = paddle.to_tensor(data['stopping_masks'], dtype='bool').t()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    stopping_mask=stopping_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
