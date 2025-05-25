from typing import Union

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import to_undirected


@functional_transform('to_undirected')
class ToUndirected(BaseTransform):
    r"""Converts a homogeneous or heterogeneous graph to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge
    :math:`(i,j) \in \mathcal{E}` (functional name: :obj:`to_undirected`).
    In heterogeneous graphs, will add "reverse" connections for *all* existing
    edge types.

    Args:
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        merge (bool, optional): If set to :obj:`False`, will create reverse
            edge types for connections pointing to the same source and target
            node type.
            If set to :obj:`True`, reverse edges will be merged into the
            original relation.
            This option only has effects in
            :class:`~paddle_geometric.data.HeteroData` graph data.
            (default: :obj:`True`)
    """
    def __init__(self, reduce: str = "add", merge: bool = True):
        self.reduce = reduce
        self.merge = merge

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.shape[1]

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key

                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = paddle.stack([col, row], axis=0)

                inv_store = data[(dst, f'rev_{rel}', src)]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.shape[0] == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce)

                for key, value in zip(keys, values):
                    store[key] = value

        return data
