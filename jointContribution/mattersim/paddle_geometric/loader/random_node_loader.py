import math
from typing import Union

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.hetero_data import to_homogeneous_edge_index


class RandomNodeLoader(paddle.io.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~paddle_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/paddle_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (paddle_geometric.data.Data or paddle_geometric.data.HeteroData):
            The :class:`~paddle_geometric.data.Data` or
            :class:`~paddle_geometric.data.HeteroData` graph object.
        num_parts (int): The number of partitions.
        **kwargs (optional): Additional arguments of
            :class:`paddle.io.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        **kwargs,
    ):
        self.data = data
        self.num_parts = num_parts

        if isinstance(data, HeteroData):
            edge_index, node_dict, edge_dict = to_homogeneous_edge_index(data)
            self.node_dict, self.edge_dict = node_dict, edge_dict
        else:
            edge_index = data.edge_index

        self.edge_index = edge_index
        self.num_nodes = data.num_nodes

        super().__init__(
            range(self.num_nodes),
            batch_size=math.ceil(self.num_nodes / num_parts),
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, index):
        if not isinstance(index, paddle.Tensor):
            index = paddle.to_tensor(index)

        if isinstance(self.data, Data):
            return self.data.subgraph(index)

        elif isinstance(self.data, HeteroData):
            node_dict = {
                key: index[(index >= start) & (index < end)] - start
                for key, (start, end) in self.node_dict.items()
            }
            return self.data.subgraph(node_dict)
