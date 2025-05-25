from typing import List, Optional, Union

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.utils.repeat import repeat

# Assuming grid_cluster is either implemented or you have an alternative implementation
grid_cluster = None  # Replace with the correct implementation if available

def voxel_grid(
    pos: Tensor,
    size: Union[float, List[float], Tensor],
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, List[float], Tensor]] = None,
    end: Optional[Union[float, List[float], Tensor]] = None,
) -> Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (paddle.Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        batch (paddle.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`paddle.Tensor`
    """
    if grid_cluster is None:
        raise ImportError('`voxel_grid` requires `grid_cluster` implementation.')

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    dim = pos.shape[1]

    if batch is None:
        batch = paddle.zeros([pos.shape[0]], dtype='int64')

    pos = paddle.concat([pos, batch.unsqueeze(-1).to(pos.dtype)], axis=-1)

    if not isinstance(size, Tensor):
        size = paddle.to_tensor(size, dtype=pos.dtype, device=pos.place)
    size = repeat(size, dim)
    size = paddle.concat([size, paddle.ones([1], dtype=size.dtype)], axis=0)  # Add additional batch dim.

    if start is not None:
        if not isinstance(start, Tensor):
            start = paddle.to_tensor(start, dtype=pos.dtype, device=pos.place)
        start = repeat(start, dim)
        start = paddle.concat([start, paddle.zeros([1], dtype=start.dtype)], axis=0)

    if end is not None:
        if not isinstance(end, Tensor):
            end = paddle.to_tensor(end, dtype=pos.dtype, device=pos.place)
        end = repeat(end, dim)
        end = paddle.concat([end, batch.max().unsqueeze(0)], axis=0)

    # Assuming grid_cluster is defined in paddle_geometric or implemented by the user
    return grid_cluster(pos, size, start, end)
