import paddle

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import degree


@functional_transform('local_degree_profile')
class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper
    (functional name: :obj:`local_degree_profile`).

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """
    def __init__(self) -> None:
        from paddle_geometric.nn.aggr.fused import FusedAggregation
        self.aggr = FusedAggregation(['min', 'max', 'mean', 'std'])

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        num_nodes = data.num_nodes

        deg = degree(row, num_nodes, dtype='float32').reshape([-1, 1])
        xs = [deg] + self.aggr(deg[col], row, dim_size=num_nodes)

        if data.x is not None:
            data.x = data.x.reshape([-1, 1]) if data.x.ndim == 1 else data.x
            data.x = paddle.concat([data.x] + xs, axis=-1)
        else:
            data.x = paddle.concat(xs, axis=-1)

        return data
