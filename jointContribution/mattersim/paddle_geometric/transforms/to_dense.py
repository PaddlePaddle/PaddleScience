from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('to_dense')
class ToDense(BaseTransform):
    r"""Converts a sparse adjacency matrix to a dense adjacency matrix with
    shape :obj:`[num_nodes, num_nodes, *]` (functional name: :obj:`to_dense`).

    Args:
        num_nodes (int, optional): The number of nodes. If set to :obj:`None`,
            the number of nodes will get automatically inferred.
            (default: :obj:`None`)
    """
    def __init__(self, num_nodes: Optional[int] = None) -> None:
        self.num_nodes = num_nodes

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes
        assert orig_num_nodes is not None

        num_nodes = self.num_nodes or orig_num_nodes
        if self.num_nodes is not None:
            assert orig_num_nodes <= self.num_nodes

        if data.edge_attr is None:
            edge_attr = paddle.ones([data.edge_index.shape[1]], dtype='float32')
        else:
            edge_attr = data.edge_attr

        size = [num_nodes, num_nodes] + list(edge_attr.shape[1:])
        adj = paddle.sparse.sparse_coo_tensor(data.edge_index, edge_attr, shape=size)
        data.adj = adj.to_dense()
        data.edge_index = None
        data.edge_attr = None

        data.mask = paddle.zeros([num_nodes], dtype='bool')
        data.mask[:orig_num_nodes] = True

        if data.x is not None:
            _size = [num_nodes - data.x.shape[0]] + list(data.x.shape[1:])
            data.x = paddle.concat([data.x, paddle.zeros(_size, dtype=data.x.dtype)], axis=0)

        if data.pos is not None:
            _size = [num_nodes - data.pos.shape[0]] + list(data.pos.shape[1:])
            data.pos = paddle.concat([data.pos, paddle.zeros(_size, dtype=data.pos.dtype)], axis=0)

        if data.y is not None and isinstance(data.y, Tensor) and data.y.shape[0] == orig_num_nodes:
            _size = [num_nodes - data.y.shape[0]] + list(data.y.shape[1:])
            data.y = paddle.concat([data.y, paddle.zeros(_size, dtype=data.y.dtype)], axis=0)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_nodes={self.num_nodes})' if self.num_nodes else f'{self.__class__.__name__}()'
