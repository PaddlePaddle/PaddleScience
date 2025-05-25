import math
import paddle
import paddle_geometric
from paddle_geometric.deprecation import deprecated
from paddle_geometric.utils import to_undirected

@deprecated("use 'transforms.RandomLinkSplit' instead")
def train_test_split_edges(
    data: 'paddle_geometric.data.Data',
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
) -> 'paddle_geometric.data.Data':
    r"""将 :class:`paddle_geometric.data.Data` 对象的边分成正负训练/验证/测试边。
    它会替换 :obj:`edge_index` 属性为
    :obj:`train_pos_edge_index`, :obj:`train_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` 和
    :obj:`test_pos_edge_index` 属性。
    如果 :obj:`data` 包含名为 :obj:`edge_attr` 的边特征，
    则也会添加 :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` 和
    :obj:`test_pos_edge_attr`。

    警告:
        :meth:`~paddle_geometric.utils.train_test_split_edges` 已弃用，并将在未来的发布中删除。
        请使用 :class:`paddle_geometric.transforms.RandomLinkSplit` 代替。

    参数:
        data (Data): 数据对象。
        val_ratio (float, optional): 正样本验证边的比例。
            (默认: :obj:`0.05`)
        test_ratio (float, optional): 正样本测试边的比例。
            (默认: :obj:`0.1`)

    :返回类型: :class:`paddle_geometric.data.Data`
    """
    assert 'batch' not in data  # 不支持批量模式。
    assert data.num_nodes is not None
    assert data.edge_index is not None

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    del data.edge_index
    del data.edge_attr

    # 返回上三角部分。
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.shape[0]))
    n_t = int(math.floor(test_ratio * row.shape[0]))

    # 正样本边。
    perm = paddle.randperm(row.shape[0])
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = paddle.stack([r, c], axis=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = paddle.stack([r, c], axis=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = paddle.stack([r, c], axis=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # 负样本边。
    neg_adj_mask = paddle.ones([num_nodes, num_nodes], dtype='uint8')
    neg_adj_mask = paddle.triu(neg_adj_mask.astype('int32'), diagonal=1).astype('bool')
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = paddle.nonzero(neg_adj_mask).t()
    perm = paddle.randperm(neg_row.shape[0])[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = paddle.stack([row, col], axis=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = paddle.stack([row, col], axis=0)

    return data
