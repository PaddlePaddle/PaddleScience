import copy
from collections import defaultdict
from typing import Union

import paddle

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('remove_isolated_nodes')
class RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes from the graph
    (functional name: :obj:`remove_isolated_nodes`).
    """
    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        # Gather all nodes that occur in at least one edge (across all types):
        n_ids_dict = defaultdict(list)
        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            n_ids_dict[src].append(edge_store.edge_index[0])
            n_ids_dict[dst].append(edge_store.edge_index[1])

        n_id_dict = {k: paddle.unique(paddle.concat(v)) for k, v in n_ids_dict.items()}

        n_map_dict = {}
        for node_store in data.node_stores:
            if node_store._key not in n_id_dict:
                n_id_dict[node_store._key] = paddle.to_tensor([], dtype='int64')

            idx = n_id_dict[node_store._key]
            mapping = paddle.zeros([data.num_nodes], dtype='int64')
            mapping[idx] = paddle.arange(idx.shape[0], dtype='int64')
            n_map_dict[node_store._key] = mapping

        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            row = n_map_dict[src][edge_store.edge_index[0]]
            col = n_map_dict[dst][edge_store.edge_index[1]]
            edge_store.edge_index = paddle.stack([row, col], axis=0)

        old_data = copy.copy(data)
        for out, node_store in zip(data.node_stores, old_data.node_stores):
            for key, value in node_store.items():
                if key == 'num_nodes':
                    out.num_nodes = n_id_dict[node_store._key].shape[0]
                elif node_store.is_node_attr(key):
                    out[key] = value[n_id_dict[node_store._key]]

        return data
