from paddle import Tensor
import paddle
import paddle_geometric
from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import is_paddle_sparse_tensor, to_paddle_csc_tensor


@functional_transform('feature_propagation')
class FeaturePropagation(BaseTransform):
    r"""The feature propagation operator from the `"On the Unreasonable
    Effectiveness of Feature propagation in Learning on Graphs with Missing
    Node Features" <https://arxiv.org/abs/2111.12128>`_ paper
    (functional name: :obj:`feature_propagation`).

    .. math::
        \mathbf{X}^{(0)} &= (1 - \mathbf{M}) \cdot \mathbf{X}

        \mathbf{X}^{(\ell + 1)} &= \mathbf{X}^{(0)} + \mathbf{M} \cdot
        (\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \mathbf{X}^{(\ell)})

    where missing node features are inferred by known features via propagation.

    .. code-block:: python

        from paddle_geometric.transforms import FeaturePropagation

        transform = FeaturePropagation(missing_mask=paddle.isnan(data.x))
        data = transform(data)

    Args:
        missing_mask (paddle.Tensor): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{N\times F}` indicating missing
            node features.
        num_iterations (int, optional): The number of propagations.
            (default: :obj:`40`)
    """
    def __init__(self, missing_mask: Tensor, num_iterations: int = 40) -> None:
        self.missing_mask = missing_mask
        self.num_iterations = num_iterations

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        assert data.edge_index is not None or data.adj_t is not None

        assert data.x.shape == self.missing_mask.shape
        gcn_norm = paddle_geometric.nn.conv.gcn_conv.gcn_norm

        missing_mask = self.missing_mask.cast('bool')
        known_mask = ~missing_mask

        if data.edge_index is not None:
            edge_weight = data.edge_attr
            if 'edge_weight' in data:
                edge_weight = data.edge_weight
            adj_t = to_paddle_csc_tensor(
                edge_index=data.edge_index,
                edge_attr=edge_weight,
                size=data.num_nodes,
            ).t()
            adj_t, _ = gcn_norm(adj_t, add_self_loops=False)
        elif is_paddle_sparse_tensor(data.adj_t):
            adj_t, _ = gcn_norm(data.adj_t, add_self_loops=False)
        else:
            adj_t = gcn_norm(data.adj_t, add_self_loops=False)

        x = data.x.clone()
        x[missing_mask] = 0.

        out = x
        for _ in range(self.num_iterations):
            out = paddle.sparse.sparse_matmul(adj_t, out)
            out = paddle.where(known_mask, x, out)  # Reset.
        data.x = out

        return data

    def __repr__(self) -> str:
        na_values = (self.missing_mask.sum().item() / self.missing_mask.numel().item()) * 100
        return (f'{self.__class__.__name__}('
                f'missing_features={na_values:.1f}%, '
                f'num_iterations={self.num_iterations})')