from math import pi as PI
from typing import Optional
import paddle

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('spherical')
class Spherical(BaseTransform):
    r"""Saves the spherical coordinates of linked nodes in its edge attributes
    (functional name: :obj:`spherical`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`{[0, 1]}^3`.
            (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        row, col = data.edge_index
        pos, pseudo = data.pos, data.edge_attr
        assert pos.shape[1] == 3

        cart = pos[col] - pos[row]

        rho = paddle.norm(cart, p=2, axis=-1).reshape([-1, 1])

        theta = paddle.atan2(cart[..., 1], cart[..., 0]).reshape([-1, 1])
        theta = theta + (theta < 0).astype(theta.dtype) * (2 * PI)

        phi = paddle.acos(cart[..., 2] / rho.reshape([-1])).reshape([-1, 1])

        if self.norm:
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * PI)
            phi = phi / PI

        spher = paddle.concat([rho, theta, phi], axis=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.reshape([-1, 1]) if pseudo.ndim == 1 else pseudo
            data.edge_attr = paddle.concat([pseudo, spher.astype(pos.dtype)], axis=-1)
        else:
            data.edge_attr = spher

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
