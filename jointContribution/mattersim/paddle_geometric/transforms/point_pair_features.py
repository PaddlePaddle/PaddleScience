import paddle

import paddle_geometric
from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('point_pair_features')
class PointPairFeatures(BaseTransform):
    r"""Computes the rotation-invariant Point Pair Features
    (functional name: :obj:`point_pair_features`).

    .. math::
        \left( \| \mathbf{d_{j,i}} \|, \angle(\mathbf{n}_i, \mathbf{d_{j,i}}),
        \angle(\mathbf{n}_j, \mathbf{d_{j,i}}), \angle(\mathbf{n}_i,
        \mathbf{n}_j) \right)

    of linked nodes in its edge attributes, where :math:`\mathbf{d}_{j,i}`
    denotes the difference vector between, and :math:`\mathbf{n}_i` and
    :math:`\mathbf{n}_j` denote the surface normals of node :math:`i` and
    :math:`j` respectively.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, cat: bool = True):
        self.cat = cat

    def forward(self, data: Data) -> Data:
        ppf_func = paddle_geometric.nn.conv.ppf_conv.point_pair_features

        assert data.edge_index is not None
        assert data.pos is not None and data.norm is not None
        assert data.pos.shape[-1] == 3
        assert data.pos.shape == data.norm.shape

        row, col = data.edge_index
        pos, norm, pseudo = data.pos, data.norm, data.edge_attr

        ppf = ppf_func(pos[row], pos[col], norm[row], norm[col])

        if pseudo is not None and self.cat:
            pseudo = pseudo.reshape([-1, 1]) if pseudo.ndim == 1 else pseudo
            data.edge_attr = paddle.concat([pseudo, ppf.astype(pseudo.dtype)], axis=-1)
        else:
            data.edge_attr = ppf

        return data
