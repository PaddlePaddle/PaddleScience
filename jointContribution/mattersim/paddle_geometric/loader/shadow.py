import copy
import math
from typing import Optional

import paddle
from paddle import Tensor
import paddle.sparse as sparse
from paddle.io import DataLoader

from paddle_geometric.data import Data, Batch
from paddle_geometric.typing import WITH_PADDLE_SPARSE


class ShaDowKHopSampler(DataLoader):
    r"""The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
    Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_ paper.
    Given a graph in a :obj:`data` object, the sampler will create shallow,
    localized subgraphs.
    A deep GNN on this local graph then smooths the informative local signals.

    Args:
        data (paddle_geometric.data.Data): The graph data object.
        depth (int): The depth/number of hops of the localized subgraph.
        num_neighbors (int): The number of neighbors to sample for each node in
            each hop.
        node_idx (LongTensor or BoolTensor, optional): The nodes that should be
            considered for creating mini-batches.
            If set to :obj:`None`, all nodes will be considered.
        replace (bool, optional): If set to :obj:`True`, will sample neighbors
            with replacement. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`paddle.io.DataLoader`, such as :obj:`batch_size` or
            :obj:`num_workers`.
    """

    def __init__(self, data: Data, depth: int, num_neighbors: int,
                 node_idx: Optional[Tensor] = None, replace: bool = False,
                 **kwargs):

        if not WITH_PADDLE_SPARSE:
            raise ImportError(
                f"'{self.__class__.__name__}' requires 'paddle-sparse'")

        self.data = copy.copy(data)
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace

        if data.edge_index is not None:
            self.is_sparse_tensor = False
            row, col = data.edge_index.cpu()
            self.adj_t = sparse.SparseCooTensor(
                indices=paddle.concat([row.unsqueeze(0), col.unsqueeze(0)], axis=0),
                values=paddle.arange(col.shape[0]),
                shape=(data.num_nodes, data.num_nodes),
            )
        else:
            self.is_sparse_tensor = True
            self.adj_t = data.adj_t.cpu()

        if node_idx is None:
            node_idx = paddle.arange(self.adj_t.shape[0])
        elif node_idx.dtype == paddle.bool:
            node_idx = paddle.nonzero(node_idx).squeeze(1)
        self.node_idx = node_idx

        super().__init__(
            range(self.num_nodes),
            batch_size=math.ceil(self.num_nodes / num_parts),
            collate_fn=self.__collate__,
            **kwargs,
        )

    def __collate__(self, n_id):
        n_id = paddle.to_tensor(n_id)

        # Convert adj_t to the COO format
        rowptr, col, value = self.adj_t.csr()

        # Assuming paddle_sparse has an equivalent function
        out = paddle.ops.paddle_sparse.ego_k_hop_sample_adj(
            rowptr, col, n_id, self.depth, self.num_neighbors, self.replace)

        rowptr, col, n_id, e_id, ptr, root_n_id = out

        adj_t = sparse.SparseCooTensor(
            indices=paddle.concat([rowptr.unsqueeze(0), col.unsqueeze(0)], axis=0),
            values=value[e_id] if value is not None else None,
            shape=(n_id.numel(), n_id.numel())
        )

        batch = Batch(batch=paddle.ops.paddle_sparse.ptr2ind(ptr, n_id.numel()),
                      ptr=ptr)
        batch.root_n_id = root_n_id

        if self.is_sparse_tensor:
            batch.adj_t = adj_t
        else:
            row, col, e_id = adj_t.t().coo()
            batch.edge_index = paddle.concat([row.unsqueeze(0), col.unsqueeze(0)], axis=0)

        for k, v in self.data:
            if k in ['edge_index', 'adj_t', 'num_nodes', 'batch', 'ptr']:
                continue
            if k == 'y' and v.shape[0] == self.data.num_nodes:
                batch[k] = v[n_id][root_n_id]
            elif isinstance(v, Tensor) and v.shape[0] == self.data.num_nodes:
                batch[k] = v[n_id]
            elif isinstance(v, Tensor) and v.shape[0] == self.data.num_edges:
                batch[k] = v[e_id]
            else:
                batch[k] = v

        return batch
