import re
from typing import List, Optional, Union

import paddle
from paddle import Tensor

import paddle_geometric
from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import one_hot, scatter


@functional_transform('grid_sampling')
class GridSampling(BaseTransform):
    r"""Clusters points into fixed-sized voxels
    (functional name: :obj:`grid_sampling`).
    Each cluster returned is a new point based on the mean of all points
    inside the given cluster.

    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        size: Union[float, List[float], Tensor],
        start: Optional[Union[float, List[float], Tensor]] = None,
        end: Optional[Union[float, List[float], Tensor]] = None,
    ) -> None:
        self.size = size
        self.start = start
        self.end = end

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes

        assert data.pos is not None
        c = paddle_geometric.nn.voxel_grid(data.pos, self.size, data.batch,
                                           self.start, self.end)
        c, perm = paddle_geometric.nn.pool.consecutive.consecutive_cluster(c)

        for key, item in data.items():
            if bool(re.search('edge', key)):
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support coarsening of edges")

            if isinstance(item, Tensor) and item.shape[0] == num_nodes:
                if key == 'y':
                    item = scatter(one_hot(item), c, dim=0, reduce='sum')
                    data[key] = item.argmax(axis=-1)
                elif key == 'batch':
                    data[key] = paddle.gather(item, perm)
                else:
                    data[key] = scatter(item, c, dim=0, reduce='mean')

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'
