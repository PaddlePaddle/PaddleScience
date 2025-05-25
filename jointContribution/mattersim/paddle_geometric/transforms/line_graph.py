import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import coalesce, cumsum, remove_self_loops, scatter


@functional_transform('line_graph')
class LineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-graph
    (functional name: :obj:`line_graph`).

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    For undirected graphs, the maximum line-graph node index is
    :obj:`(data.edge_index.size(1) // 2) - 1`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed: bool = False) -> None:
        self.force_directed = force_directed

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=N)
        row, col = edge_index

        if self.force_directed or data.is_directed():
            i = paddle.arange(row.shape[0], dtype=paddle.int64)

            count = scatter(paddle.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            ptr = cumsum(count)

            cols = [i[ptr[col[j]]:ptr[col[j] + 1]] for j in range(col.shape[0])]
            rows = [paddle.full((c.shape[0],), j, dtype=paddle.int64) for j, c in enumerate(cols)]

            row, col = paddle.concat(rows, axis=0), paddle.concat(cols, axis=0)

            data.edge_index = paddle.stack([row, col], axis=0)
            data.x = data.edge_attr
            data.num_nodes = edge_index.shape[1]

        else:
            mask = row < col
            row, col = row[mask], col[mask]
            i = paddle.arange(row.shape[0], dtype=paddle.int64)

            (row, col), i = coalesce(
                paddle.stack([
                    paddle.concat([row, col], axis=0),
                    paddle.concat([col, row], axis=0)
                ], axis=0),
                paddle.concat([i, i], axis=0),
                N,
            )

            count = scatter(paddle.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            joints = list(paddle.split(i, count.tolist()))

            def generate_grid(x: Tensor) -> Tensor:
                row = x.unsqueeze(-1).expand([x.shape[0], x.shape[0]]).flatten()
                col = x.expand([x.shape[0], x.shape[0]]).flatten()
                return paddle.stack([row, col], axis=0)

            joints = [generate_grid(joint) for joint in joints]
            joint = paddle.concat(joints, axis=1)
            joint, _ = remove_self_loops(joint)
            N = row.shape[0] // 2
            joint = coalesce(joint, num_nodes=N)

            if edge_attr is not None:
                data.x = scatter(edge_attr, i, dim=0, dim_size=N, reduce='sum')
            data.edge_index = joint
            data.num_nodes = edge_index.shape[1] // 2

        data.edge_attr = None

        return data
