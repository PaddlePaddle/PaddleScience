import os.path as osp
from typing import Callable, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url


class Twitch(InMemoryDataset):
    r"""The Twitch Gamer networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent gamers on Twitch and edges are followerships between them.
    Node features represent embeddings of games played by the Twitch users.
    The task is to predict whether a user streams mature content.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"DE"`, :obj:`"EN"`,
            :obj:`"ES"`, :obj:`"FR"`, :obj:`"PT"`, :obj:`"RU"`).
        transform (callable, optional): A function/transform that takes in a
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in a
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to disk.
        force_reload (bool, optional): Whether to re-process the dataset.

    **STATS:**

    +------+--------+--------+----------+---------+
    | Name | #nodes | #edges | #features| #classes|
    +------+--------+--------+----------+---------+
    |  DE  | 9,498  |315,774 |   128    |    2    |
    |  EN  | 7,126  | 77,774 |   128    |    2    |
    |  ES  | 4,648  |123,412 |   128    |    2    |
    |  FR  | 6,551  |231,883 |   128    |    2    |
    |  PT  | 1,912  | 64,510 |   128    |    2    |
    |  RU  | 4,385  | 78,993 |   128    |    2    |
    +------+--------+--------+----------+---------+
    """

    url = 'https://graphmining.ai/datasets/ptg/twitch'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self) -> None:
        data = np.load(self.raw_paths[0], allow_pickle=True)
        x = paddle.to_tensor(data['features'], dtype=paddle.float32)
        y = paddle.to_tensor(data['target'], dtype=paddle.int64)

        edge_index = paddle.to_tensor(data['edges'], dtype=paddle.int64)
        edge_index = paddle.transpose(edge_index, perm=[1, 0])

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
