import copy

import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('virtual_node')
class VirtualNode(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph that is connected
    to all other nodes, as described in the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper
    (functional name: :obj:`virtual_node`).
    The virtual node serves as a global scratch space that each node both reads
    from and writes to in every step of message passing.
    This allows information to travel long distances during the propagation
    phase.

    Node and edge features of the virtual node are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from the virtual node.
    """
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        edge_type = data.get('edge_type', paddle.zeros_like(row))
        num_nodes = data.num_nodes
        assert num_nodes is not None

        arange = paddle.arange(num_nodes, dtype=row.dtype)
        full = paddle.full([num_nodes], num_nodes, dtype=row.dtype)
        row = paddle.concat([row, arange, full], axis=0)
        col = paddle.concat([col, full, arange], axis=0)
        edge_index = paddle.stack([row, col], axis=0)

        num_edge_types = int(edge_type.max().item()) if edge_type.numel() > 0 else 0
        new_type = paddle.full([num_nodes], num_edge_types + 1, dtype=edge_type.dtype)
        edge_type = paddle.concat([edge_type, new_type, new_type + 1], axis=0)

        old_data = copy.copy(data)
        for key, value in old_data.items():
            if key in {'edge_index', 'edge_type'}:
                continue

            if isinstance(value, Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.shape)

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes
                    fill_value = 1.
                elif key == 'batch':
                    size[dim] = 1
                    fill_value = int(value[0].item())
                elif old_data.is_edge_attr(key):
                    size[dim] = 2 * num_nodes
                    fill_value = 0.
                elif old_data.is_node_attr(key):
                    size[dim] = 1
                    fill_value = 0.

                if fill_value is not None:
                    new_value = paddle.full(size, fill_value, dtype=value.dtype)
                    data[key] = paddle.concat([value, new_value], axis=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes + 1

        return data
