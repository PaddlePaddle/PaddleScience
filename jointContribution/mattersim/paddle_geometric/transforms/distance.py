from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('distance')
class Distance(BaseTransform):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes
    (functional name: :obj:`distance`). Each distance gets globally normalized
    to a specified interval (:math:`[0, 1]` by default).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval ((float, float), optional): A tuple specifying the lower and
            upper bound for normalization. (default: :obj:`(0.0, 1.0)`)
    """

    def __init__(
            self,
            norm: bool = True,
            max_value: Optional[float] = None,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None, "Node positions ('pos') must be provided in data."
        assert data.edge_index is not None, "Edge indices ('edge_index') must be provided in data."

        row, col = data.edge_index
        pos, pseudo = data.pos, data.edge_attr

        dist = paddle.norm(pos[col] - pos[row], p=2, axis=-1).reshape([-1, 1])

        if self.norm and dist.numel() > 0:
            max_val = float(dist.max()) if self.max is None else self.max

            length = self.interval[1] - self.interval[0]
            dist = length * (dist / max_val) + self.interval[0]

        if pseudo is not None and self.cat:
            pseudo = pseudo.reshape([-1, 1]) if pseudo.ndim == 1 else pseudo
            data.edge_attr = paddle.concat([pseudo, dist.astype(pseudo.dtype)], axis=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
