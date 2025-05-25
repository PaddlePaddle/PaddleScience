from typing import Any, Dict

import numpy as np
import paddle
import scipy.sparse as sp

from paddle_geometric.data import Data
from paddle_geometric.utils import remove_self_loops
from paddle_geometric.utils import to_undirected as to_undirected_fn


def read_npz(path: str, to_undirected: bool = True) -> Data:
    with np.load(path) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Data:
    # 读取属性矩阵并转换为稀疏矩阵格式
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = paddle.to_tensor(x, dtype='float32')
    x = paddle.where(x > 0, paddle.ones_like(x), paddle.zeros_like(x))

    # 读取邻接矩阵并转换为稀疏矩阵格式
    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    row = paddle.to_tensor(adj.row, dtype='int64')
    col = paddle.to_tensor(adj.col, dtype='int64')
    edge_index = paddle.stack([row, col], axis=0)
    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.shape[0])

    y = paddle.to_tensor(f['labels'], dtype='int64')

    return Data(x=x, edge_index=edge_index, y=y)
