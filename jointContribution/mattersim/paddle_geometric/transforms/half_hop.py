import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('half_hop')
class HalfHop(BaseTransform):
    r"""The graph upsampling augmentation from the
    `"Half-Hop: A Graph Upsampling Approach for Slowing Down Message Passing"
    <https://openreview.net/forum?id=lXczFIwQkv>`_ paper.
    The graph is augmented by adding artificial slow nodes between neighbors
    to slow down message propagation. (functional name: :obj:`half_hop`).

    .. note::
        :class:`HalfHop` augmentation is not supported if :obj:`data` has
        :attr:`edge_weight` or :attr:`edge_attr`.

    Args:
        alpha (float, optional): The interpolation factor
            used to compute slow node features
            :math:`x = \alpha*x_src + (1-\alpha)*x_dst` (default: :obj:`0.5`)
        p (float, optional): The probability of half-hopping
            an edge. (default: :obj:`1.0`)

    .. code-block:: python

        import paddle_geometric.transforms as T

        transform = T.HalfHop(alpha=0.5)
        data = transform(data)  # Apply transformation.
        out = model(data.x, data.edge_index)  # Feed-forward.
        out = out[~data.slow_node_mask]  # Get rid of slow nodes.
    """
    def __init__(self, alpha: float = 0.5, p: float = 1.0) -> None:
        if alpha < 0. or alpha > 1.:
            raise ValueError(f"Interpolation factor has to be between 0 and 1 "
                             f"(got '{alpha}'")
        if p < 0. or p > 1.:
            raise ValueError(f"Ratio of half-hopped edges has to be between "
                             f"0 and 1 (got '{p}'")

        self.p = p
        self.alpha = alpha

    def forward(self, data: Data) -> Data:
        if data.edge_weight is not None or data.edge_attr is not None:
            raise ValueError("'HalfHop' augmentation is not supported if "
                             "'data' contains 'edge_weight' or 'edge_attr'")

        assert data.x is not None
        assert data.edge_index is not None
        x, edge_index = data.x, data.edge_index
        num_nodes = data.num_nodes
        assert num_nodes is not None

        # isolate self loops which are not half-hopped
        self_loop_mask = edge_index[0] == edge_index[1]
        edge_index_self_loop = edge_index[:, self_loop_mask]
        edge_index = edge_index[:, ~self_loop_mask]

        # randomly sample nodes and half-hop their edges
        node_mask = paddle.rand([num_nodes]) < self.p
        edge_mask = paddle.gather(node_mask, edge_index[1])
        edge_index_to_halfhop = edge_index[:, edge_mask]
        edge_index_to_keep = edge_index[:, ~edge_mask]

        # add new slow nodes of which features are initialized
        # by linear interpolation
        num_halfhop_edges = edge_index_to_halfhop.shape[1]
        slow_node_ids = paddle.arange(num_halfhop_edges) + num_nodes
        x_src = paddle.gather(x, edge_index_to_halfhop[0])
        x_dst = paddle.gather(x, edge_index_to_halfhop[1])
        x_slow_node = self.alpha * x_src + (1 - self.alpha) * x_dst
        new_x = paddle.concat([x, x_slow_node], axis=0)

        # add new edges between slow nodes and the original nodes
        edge_index_slow = [
            paddle.stack([edge_index_to_halfhop[0], slow_node_ids]),
            paddle.stack([slow_node_ids, edge_index_to_halfhop[1]]),
            paddle.stack([edge_index_to_halfhop[1], slow_node_ids])
        ]
        new_edge_index = paddle.concat(
            [edge_index_to_keep, edge_index_self_loop] + edge_index_slow,
            axis=1)

        # prepare a mask that distinguishes between original nodes & slow nodes
        slow_node_mask = paddle.concat(
            [paddle.zeros([x.shape[0]]), paddle.ones([slow_node_ids.shape[0]])]
        ).astype('bool')

        data.x, data.edge_index = new_x, new_edge_index
        data.slow_node_mask = slow_node_mask

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, p={self.p})'
