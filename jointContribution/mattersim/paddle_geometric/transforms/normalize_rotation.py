import paddle
import paddle.nn.functional as F

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('normalize_rotation')
class NormalizeRotation(BaseTransform):
    r"""Rotates all points according to the eigenvectors of the point cloud
    (functional name: :obj:`normalize_rotation`).
    If the data additionally holds normals saved in :obj:`data.normal`, these
    will be rotated accordingly.

    Args:
        max_points (int, optional): If set to a value greater than :obj:`0`,
            only a random number of :obj:`max_points` points are sampled and
            used to compute eigenvectors. (default: :obj:`-1`)
        sort (bool, optional): If set to :obj:`True`, will sort eigenvectors
            according to their eigenvalues. (default: :obj:`False`)
    """
    def __init__(self, max_points: int = -1, sort: bool = False) -> None:
        self.max_points = max_points
        self.sort = sort

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        pos = data.pos

        if self.max_points > 0 and pos.shape[0] > self.max_points:
            perm = paddle.randperm(pos.shape[0])
            pos = paddle.index_select(pos, perm[:self.max_points])

        pos = pos - pos.mean(axis=0, keepdim=True)
        C = paddle.matmul(pos.t(), pos)
        e, v = paddle.linalg.eig(C)
        e, v = paddle.real(e), paddle.real(v)

        if self.sort:
            indices = paddle.argsort(e, descending=True)
            v = paddle.index_select(v, indices, axis=1)

        data.pos = paddle.matmul(data.pos, v)

        if 'normal' in data:
            data.normal = F.normalize(paddle.matmul(data.normal, v))
            data.normal = paddle.round(data.normal * 10000) / 10000

        return data
