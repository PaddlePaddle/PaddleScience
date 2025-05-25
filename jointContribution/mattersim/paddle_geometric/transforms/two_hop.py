import paddle

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import coalesce, remove_self_loops


@functional_transform('two_hop')
class TwoHop(BaseTransform):
    r"""Adds the two-hop edges to the edge indices
    (functional name: :obj:`two_hop`).
    """

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        # Convert edge_index to a dense representation for multiplication
        edge_index_dense = paddle.to_tensor(edge_index.numpy())
        edge_index_dense_sorted = paddle.sort(edge_index_dense, axis=0)[0]

        # Perform two-hop connection calculation
        edge_index2 = paddle.matmul(edge_index_dense_sorted, edge_index_dense_sorted)

        # Convert back to sparse format and remove self-loops
        edge_index2, _ = remove_self_loops(edge_index2)

        # Concatenate original edges with two-hop edges
        edge_index = paddle.concat([edge_index_dense, edge_index2], axis=1)

        if edge_attr is not None:
            # Newly added edges will have zero features
            edge_attr2 = paddle.zeros([edge_index2.shape[1]] + list(edge_attr.shape[1:]), dtype=edge_attr.dtype)
            edge_attr = paddle.concat([edge_attr, edge_attr2], axis=0)

        # Coalesce to handle duplicates and finalize
        data.edge_index, data.edge_attr = coalesce(edge_index, edge_attr, N)

        return data
