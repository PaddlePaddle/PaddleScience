import paddle

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import to_undirected


@functional_transform('face_to_edge')
class FaceToEdge(BaseTransform):
    r"""Converts mesh faces :obj:[3, num_faces] to edge indices
    :obj:[2, num_edges] (functional name: :obj:face_to_edge).

    Args:
        remove_faces (bool, optional): If set to :obj:False, the face tensor
            will not be removed.
    """
    def __init__(self, remove_faces: bool = True) -> None:
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'face'):
            assert data.face is not None
            face = data.face
            edge_index = paddle.concat([face[:2], face[1:], face[::2]], axis=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data
