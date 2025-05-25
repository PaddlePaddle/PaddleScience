import paddle

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('delaunay')
class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`).
    """
    def forward(self, data: Data) -> Data:
        import scipy.spatial

        assert data.pos is not None

        if data.pos.shape[0] < 2:
            data.edge_index = paddle.to_tensor([], dtype='int64').reshape([2, 0])
        elif data.pos.shape[0] == 2:
            data.edge_index = paddle.to_tensor([[0, 1], [1, 0]], dtype='int64')
        elif data.pos.shape[0] == 3:
            data.face = paddle.to_tensor([[0], [1], [2]], dtype='int64')
        elif data.pos.shape[0] > 3:
            pos = data.pos.numpy()
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            face = paddle.to_tensor(tri.simplices, dtype='int64')

            data.face = face.t()

        return data
