from typing import Tuple

import paddle

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import scatter


@functional_transform('local_cartesian')
class LocalCartesian(BaseTransform):
    r"""Saves the relative Cartesian coordinates of linked nodes in its edge
    attributes (functional name: :obj:`local_cartesian`). Each coordinate gets
    *neighborhood-normalized* to a specified interval
    (:math:`[0, 1]` by default).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized. (default: :obj:`True`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval ((float, float), optional): A tuple specifying the lower and
            upper bound for normalization. (default: :obj:`(0.0, 1.0)`)
    """
    def __init__(
            self,
            norm: bool = True,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[row] - pos[col]
        cart = cart.reshape([-1, 1]) if cart.ndim == 1 else cart

        if self.norm:
            max_value = scatter(cart.abs(), col, 0, pos.shape[0], reduce='max')
            max_value = max_value.max(axis=-1, keepdim=True)

            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            cart = length * cart / (2 * max_value[col]) + center

        if pseudo is not None and self.cat:
            pseudo = pseudo.reshape([-1, 1]) if pseudo.ndim == 1 else pseudo
            data.edge_attr = paddle.concat([pseudo, cart.astype(pseudo.dtype)], axis=-1)
        else:
            data.edge_attr = cart

        return data
