from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_url
from paddle_geometric.utils import coalesce


class Actor(InMemoryDataset):
    r"""The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in terms of words of
    actor's Wikipedia.

    Args:
        root: Root directory where the dataset should be saved.
        transform: A function/transform that takes in a
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in a
            :class:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk.
        force_reload: Whether to re-process the dataset.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 7,600
          - 30,019
          - 932
          - 5
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

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
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            node_data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, line, _ in node_data:
                indices = [int(x) for x in line.split(',')]
                rows += [int(n_id)] * len(indices)
                cols += indices
            row, col = paddle.to_tensor(rows, dtype='int64'), paddle.to_tensor(cols, dtype='int64')

            x = paddle.zeros([int(row.max()) + 1, int(col.max()) + 1], dtype='float32')
            x[row, col] = 1.0

            y = paddle.empty([len(node_data)], dtype='int64')
            for n_id, _, label in node_data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1]) as f:
            edge_data = f.read().split('\n')[1:-1]
            edge_indices = [[int(v) for v in r.split('\t')] for r in edge_data]
            edge_index = paddle.to_tensor(edge_indices, dtype='int64').transpose([1, 0])
            edge_index = coalesce(edge_index, num_nodes=x.shape[0])

        train_masks, val_masks, test_masks = [], [], []
        for path in self.raw_paths[2:]:
            tmp = np.load(path)
            train_masks += [paddle.to_tensor(tmp['train_mask'], dtype='bool')]
            val_masks += [paddle.to_tensor(tmp['val_mask'], dtype='bool')]
            test_masks += [paddle.to_tensor(tmp['test_mask'], dtype='bool')]
        train_mask = paddle.stack(train_masks, axis=1)
        val_mask = paddle.stack(val_masks, axis=1)
        test_mask = paddle.stack(test_masks, axis=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
