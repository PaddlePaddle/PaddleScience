import os.path as osp
import warnings
from itertools import repeat
from typing import Dict, List, Optional

import fsspec
import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.io import read_txt_array
from paddle_geometric.utils import (
    coalesce,
    index_to_mask,
    remove_self_loops,
    to_paddle_csr_tensor,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder: str, prefix: str) -> Data:
    # List of data items to load
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = paddle.arange(y.shape[0], dtype='int64')
    val_index = paddle.arange(y.shape[0], y.shape[0] + 500, dtype='int64')
    sorted_test_index = paddle.sort(test_index)

    if prefix.lower() == 'citeseer':
        # Handle isolated nodes in Citeseer dataset with missing test indices
        len_test_indices = int(test_index.max() - test_index.min()) + 1

        tx_ext = paddle.zeros([len_test_indices, tx.shape[1]], dtype=tx.dtype)
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = paddle.zeros([len_test_indices, ty.shape[1]], dtype=ty.dtype)
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == 'nell.0.001':
        tx_ext = paddle.zeros([len(graph) - allx.shape[0], x.shape[1]])
        tx_ext[sorted_test_index - allx.shape[0]] = tx

        ty_ext = paddle.zeros([len(graph) - ally.shape[0], y.shape[1]])
        ty_ext[sorted_test_index - ally.shape[0]] = ty

        tx, ty = tx_ext, ty_ext

        x = paddle.concat([allx, tx], axis=0)
        x[test_index] = x[sorted_test_index]

        row, col = paddle.nonzero(x, as_tuple=True)
        value = x[row, col]

        mask = ~index_to_mask(test_index, size=len(graph))
        mask[:allx.shape[0]] = False
        isolated_idx = paddle.nonzero(mask).flatten()

        row = paddle.concat([row, isolated_idx])
        col = paddle.concat([col, paddle.arange(isolated_idx.shape[0]) + x.shape[1]])
        value = paddle.concat([value, paddle.ones(isolated_idx.shape[0], dtype=value.dtype)])

        x = to_paddle_csr_tensor(
            edge_index=paddle.stack([row, col], axis=0),
            edge_attr=value,
            size=(x.shape[0], isolated_idx.shape[0] + x.shape[1]),
        )
    else:
        x = paddle.concat([allx, tx], axis=0)
        x[test_index] = x[sorted_test_index]

    y = paddle.concat([ally, ty], axis=0).argmax(axis=1)
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.shape[0])
    val_mask = index_to_mask(val_index, size=y.shape[0])
    test_mask = index_to_mask(test_index, size=y.shape[0])

    edge_index = edge_index_from_dict(
        graph_dict=graph,
        num_nodes=y.shape[0],
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder: str, prefix: str, name: str) -> Tensor:
    # Load data file and return as Paddle tensor
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path, dtype='int64')

    with fsspec.open(path, 'rb') as f:
        warnings.filterwarnings('ignore', '.*`scipy.sparse.csr` name.*')
        out = pickle.load(f, encoding='latin1')

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = paddle.to_tensor(out, dtype='float32')
    return out


def edge_index_from_dict(
    graph_dict: Dict[int, List[int]],
    num_nodes: Optional[int] = None,
) -> Tensor:
    rows: List[int] = []
    cols: List[int] = []
    for key, value in graph_dict.items():
        rows += repeat(key, len(value))
        cols += value
    row = paddle.to_tensor(rows, dtype='int64')
    col = paddle.to_tensor(cols, dtype='int64')
    edge_index = paddle.stack([row, col], axis=0)

    edge_index = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes, sort_by_row=False)

    return edge_index
